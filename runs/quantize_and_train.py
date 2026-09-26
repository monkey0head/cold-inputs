"""Train and evaluate item-ID or semantic-ID sequential recommenders.

SID recipes quantize embeddings before training and pass semantic IDs in memory.
Item-ID recipes skip quantization. All metrics are logged to a single experiment.
"""

import inspect
import logging
import os

import hydra
import pandas as pd
import torch
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf

from runs._resolvers import register_resolvers
from runs.train import (create_dataloaders, create_model, evaluate, load_for_eval,
                        predict, prepare_data, training)
from src.preprocess.training_exposure import (ExposureStats, compute_item_collision_props,
                                              compute_training_exposure, log_capacity_stats,
                                              log_exposure_stats, log_subset_stats)
from src.run_utils import (count_parameters, fix_seeds, log_peak_memory_stats,
                           reset_peak_memory_stats)
from src.trackers import ExperimentTracker


register_resolvers()


@hydra.main(version_base=None, config_path="config", config_name="quantize_and_train")
def main(config: DictConfig):

    print(OmegaConf.to_yaml(config, resolve=True))

    dataset_class = get_class(config.dataset_params.train_dataset._target_)
    needs_sids = "semantic_ids" in inspect.signature(dataset_class).parameters
    if config.quantization.enabled != needs_sids:
        raise ValueError("quantization.enabled must match the selected training dataset's SID requirement")
    if config.quantization_only and not config.quantization.enabled:
        raise ValueError("quantization_only requires quantization.enabled=true")

    if hasattr(config, "cuda_visible_devices"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda_visible_devices)

    reset_peak_memory_stats()

    experiment: ExperimentTracker = instantiate(config.experiment_tracker)
    experiment.log_config(config)
    logging.info(f"Started experiment {experiment.experiment_id}.")

    fix_seeds(config.random_state)

    semantic_ids = run_quantization(config, experiment) if config.quantization.enabled else None

    if config.quantization_only:
        log_peak_memory_stats(experiment)
        experiment.finalize()
        return

    metrics_val = run_training(config, experiment, semantic_ids)

    log_peak_memory_stats(experiment)
    experiment.finalize()

    if metrics_val is not None:
        return metrics_val[config.optimize_metric]


def run_quantization(config: DictConfig, experiment: ExperimentTracker):

    from runs.quantize import (evaluate_sids, fit_quantizer, generate_sids,
                               load_categories, load_embeddings)

    X = load_embeddings(config.quantization)
    categories = load_categories(
        config.quantization, config.dataset, num_embedding_rows=X.shape[0]
    )
    quantizer = fit_quantizer(X, config, experiment)
    semantics_mapping, raw_sids, sids, collision_codebook_size = generate_sids(
        X, quantizer, config.quantization, experiment
    )
    evaluate_sids(
        raw_sids, X, quantizer, collision_codebook_size, config.quantization, experiment,
        categories=categories,
    )

    return sids


def run_training(config: DictConfig, experiment: ExperimentTracker, semantic_ids: torch.Tensor | None = None):

    data, item_count, semantic_ids = prepare_data(config, semantic_ids=semantic_ids)
    if semantic_ids is not None:
        exposure, item_props = log_data_stats(config, experiment, data, semantic_ids)
    else:
        log_subset_stats(data, experiment, catalog_size=int(item_count))

    train_loader, eval_loader = create_dataloaders(
        data["train"], data["validation_input"], data["validation_target"], config, semantic_ids)

    model = create_model(config, item_count, semantic_ids)
    if config.eval_checkpoint:
        trainer, seqrec_module = load_for_eval(model, config, experiment, semantic_ids)
    else:
        trainer, seqrec_module = training(model, train_loader, eval_loader, config, experiment, semantic_ids)

    total_params, _ = count_parameters(seqrec_module)
    strata = {}
    if semantic_ids is not None:
        log_capacity_stats(
            experiment,
            total_params=total_params,
            vocab_size=item_count + 1,
            catalog_size=int(semantic_ids.shape[0]) - 1,
            exposure=exposure,
            best_epoch=getattr(seqrec_module, "_best_epoch", None),
        )
        # SID exposure and collision strata apply only to the SID path.
        strata = {
            "warm_items": exposure.reachable_items,
            "target_exposure": exposure.item_expected_occ,
            "item_props": item_props,
        }
    else:
        experiment.log_scalars({"capacity/total_params": total_params,
                                "capacity/vocab_size": int(item_count) + 1})

    metrics_val = None
    if config.calc_val_metrics:
        validation_input, validation_target = data["validation_input"], data["validation_target"]
        recs_val = predict(trainer, seqrec_module, validation_input, config, experiment, semantic_ids, prefix="val")
        metrics_val = evaluate(recs_val, validation_target, data["train"], experiment, config, prefix="val", **strata)

    if config.calc_test_metrics:
        test_input, test_target = data["test_input"], data["test_target"]
        recs_test = predict(trainer, seqrec_module, test_input, config, experiment, semantic_ids, prefix="test")
        metrics_test = evaluate(recs_test, test_target, data["train"], experiment, config, prefix="test", **strata)

    return metrics_val


def log_data_stats(config: DictConfig, experiment: ExperimentTracker, data: dict,
                   semantic_ids: torch.Tensor) -> tuple[ExposureStats, pd.DataFrame]:
    """Log per-subset, training-exposure, and item-collision stats.

    Returns the exposure accounting and the per-item collision properties, both of which
    the stratified evaluation in ``runs/train.py`` consumes.

    The token count is read off the SID tensor rather than the config: it is the number of
    SID columns the model consumes, which is ``num_tokens_per_item`` and differs from
    ``quantizer.num_codebooks`` for every collision solver that appends a token.
    """
    catalog_size = int(semantic_ids.shape[0]) - 1  # row 0 is the padding SID
    log_subset_stats(data, experiment, catalog_size=catalog_size)

    num_tokens = int(semantic_ids.shape[1])
    if num_tokens != int(config.num_tokens_per_item):
        raise ValueError(
            f"SID tensor has {num_tokens} columns but num_tokens_per_item="
            f"{config.num_tokens_per_item}; exposure stats would be off by a codebook"
        )

    # Item ids index the SID rows directly, so an offset would misalign the two.
    max_item_id = int(data["train"]["item_id"].max())
    if max_item_id > catalog_size:
        raise ValueError(
            f"train item_id {max_item_id} exceeds the {catalog_size} catalog rows of the SID "
            f"tensor (item_id_offset={config.item_id_offset}); items and SIDs are misaligned"
        )

    train_dataset_config = config.dataset_params.train_dataset
    exposure = compute_training_exposure(
        data["train"], semantic_ids,
        window_items=int(train_dataset_config.max_length) // num_tokens,
        num_codebooks=num_tokens,
        random_slice=bool(OmegaConf.select(train_dataset_config, "random_slice", default=False)),
    )
    log_exposure_stats(experiment, exposure)

    return exposure, compute_item_collision_props(semantic_ids, num_tokens)


if __name__ == "__main__":

    main()

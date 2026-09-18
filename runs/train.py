"""Train and evaluate sequential recommender from Hydra config.
"""

import os
import time
import inspect

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate, get_class
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from src.datasets import PaddingCollateFn
from src.metrics import Evaluator
from src.postprocess import preds2recs
from src.preprocess.data_stats import base_stats
from src.run_utils import count_parameters
from src.trackers import ExperimentTracker

# Corpus-level metrics are undefined on a user subset (they depend on the whole
# recommended-item population), so they are excluded from cold/warm stratification
# and reported only on the full population by the existing evaluate() path.
CORPUS_METRICS = {"Coverage", "Surprisal", "Novelty"}

# `train_last_days` is expressed in days, and the split parquets store second-resolution
# timestamps (see the project CLAUDE.md on Yambda's uint32 seconds).
SECONDS_PER_DAY = 86400


def prepare_data(config: DictConfig, semantic_ids=None):

    seq_rec_data_path = os.getenv("SEQ_REC_DATA_PATH")
    data_path = os.path.join(seq_rec_data_path, 'split', config.dataset.name, config.split_name)

    data = {}
    for subset in ("train", "validation", "test_input", "test_target"):
        subset_path = os.path.join(data_path, f'{subset}.parquet')
        data[subset] = pd.read_parquet(subset_path)

        if config.item_id_offset is not None:
            data[subset].item_id += config.item_id_offset

        print(f'{subset} shape', data[subset].shape)
        print(f'{subset} stats\n', base_stats(data[subset], extended=False))

    if semantic_ids is not None:
        item_count = int(semantic_ids.max())
    else:
        item_count = data['train'].item_id.max()

    print(f'Item count: {item_count}')

    # Trailing-window training set. Validation and test are untouched, and `item_count` is
    # taken above so the vocabulary stays the full catalog whatever the window keeps.
    if config.train_last_days is not None:
        train = data['train']
        cutoff = train['timestamp'].max() - int(config.train_last_days) * SECONDS_PER_DAY
        data['train'] = train[train['timestamp'] >= cutoff].reset_index(drop=True)
        print(f"train_last_days={config.train_last_days}: kept {len(data['train']):,} of "
              f"{len(train):,} interactions, {data['train'].user_id.nunique():,} of "
              f"{train.user_id.nunique():,} users")

    return data, item_count, semantic_ids


def create_dataloaders(train, validation, config, semantic_ids=None):

    validation_size = config.dataloader.validation_size
    validation_users = validation.user_id.unique()
    if validation_size and (validation_size < len(validation_users)):
        validation_users = np.random.choice(validation_users, size=validation_size, replace=False)
        validation = validation[validation.user_id.isin(validation_users)]
        print(f'validation-it-training stats', base_stats(validation, extended=False))

    extra_kwargs = {'semantic_ids': semantic_ids} if semantic_ids is not None else {}

    train_kwargs = filter_kwargs(config.dataset_params.train_dataset._target_, **extra_kwargs)
    train_dataset = instantiate(config.dataset_params.train_dataset, df=train, random_state=int(config.random_state), **train_kwargs)

    eval_kwargs = filter_kwargs(config.dataset_params.predict_dataset._target_, **extra_kwargs)
    eval_dataset = instantiate(config.dataset_params.predict_dataset, df=validation, validation_mode=True, **eval_kwargs)

    left_padding = getattr(config.model, "left_padding", False)
    train_loader = DataLoader(train_dataset, batch_size=config.dataloader.batch_size,
                              shuffle=True, num_workers=config.dataloader.num_workers,
                              collate_fn=PaddingCollateFn(left_padding=left_padding))
    eval_loader = DataLoader(eval_dataset, batch_size=config.dataloader.test_batch_size,
                             shuffle=False, num_workers=config.dataloader.num_workers,
                             collate_fn=PaddingCollateFn(left_padding=left_padding))

    return train_loader, eval_loader


def create_model(config, item_count, semantic_ids=None):

    kwargs = {}
    if hasattr(config.model, "vocab_size_field") and config.model.vocab_size_field is not None:
        kwargs[str(config.model.vocab_size_field)] = item_count + int(
            getattr(config.model, "vocab_size_offset", 1)
        )

    if semantic_ids is not None:
        kwargs['semantic_ids'] = semantic_ids

    model_node = config.model.specification if hasattr(config.model, "specification") else config.model
    kwargs = filter_kwargs(model_node._target_, **kwargs)
    model = instantiate(model_node, **kwargs)

    if config.compile:
        model = torch.compile(model)
        if hasattr(model, "_forward_with_kv_cache"):
            model._forward_with_kv_cache = torch.compile(model._forward_with_kv_cache, dynamic=True)

    return model


def training(model, train_loader, eval_loader, config, experiment: ExperimentTracker, semantic_ids=None):

    extra_kwargs = {'semantic_ids': semantic_ids} if semantic_ids is not None else {}
    extra_kwargs = filter_kwargs(config.seqrec_module._target_, **extra_kwargs)
    seqrec_module = instantiate(config.seqrec_module, model=model, **extra_kwargs)

    total_params, trainable_params = count_parameters(seqrec_module)
    print(f"Total parameters: {total_params:,} (trainable: {trainable_params:,})")
    experiment.log_scalar("num_params_total", total_params)
    experiment.log_scalar("num_params_trainable", trainable_params)

    model_checkpoint = hydra.utils.instantiate(
        config.model_checkpoint,
        dirpath=os.path.join(config.checkpoint_dir, experiment.experiment_id),
    )
    callbacks = hydra.utils.instantiate(config.callbacks)
    callbacks.append(model_checkpoint)

    trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks)

    start_time = time.time()
    try:
        trainer.fit(model=seqrec_module,
                    train_dataloaders=train_loader,
                    val_dataloaders=eval_loader)
    except RuntimeError as e:
        remove_best_checkpoint_if_cuda_oom(e, model_checkpoint)
        raise
    finally:
        finalize_training_checkpoint(trainer, seqrec_module, model_checkpoint, experiment)

    training_time = time.time() - start_time
    print('Training time', training_time)
    experiment.log_scalar('training_time', training_time)

    return trainer, seqrec_module


def load_for_eval(model, config, experiment: ExperimentTracker, semantic_ids=None):
    """Rebuild a trained module from ``config.eval_checkpoint`` instead of fitting it.

    Returns the same ``(trainer, seqrec_module)`` pair as ``training``, so prediction,
    evaluation and the stratified breakdowns run unchanged. The checkpoint must match
    the model and item-ID mapping. SID recipes also require the original quantization
    so that each embedding row represents the same semantic ID.
    """
    extra_kwargs = {'semantic_ids': semantic_ids} if semantic_ids is not None else {}
    extra_kwargs = filter_kwargs(config.seqrec_module._target_, **extra_kwargs)
    seqrec_module = instantiate(config.seqrec_module, model=model, **extra_kwargs)

    total_params, trainable_params = count_parameters(seqrec_module)
    print(f"Total parameters: {total_params:,} (trainable: {trainable_params:,})")
    experiment.log_scalar("num_params_total", total_params)
    experiment.log_scalar("num_params_trainable", trainable_params)

    checkpoint = torch.load(config.eval_checkpoint)
    seqrec_module.load_state_dict(checkpoint["state_dict"])
    best_epoch = checkpoint["epoch"]
    seqrec_module._best_epoch = best_epoch  # consumed by the compute-budget capacity stat
    experiment.log_scalar("num_best_epochs", best_epoch)
    print(f"Loaded checkpoint for evaluation: {config.eval_checkpoint} (epoch {best_epoch})")

    trainer = hydra.utils.instantiate(config.trainer, callbacks=[])

    return trainer, seqrec_module


def remove_best_checkpoint_if_cuda_oom(error: RuntimeError, model_checkpoint: ModelCheckpoint):

    if "CUDA out of memory" not in str(error):
        return
    path = model_checkpoint.best_model_path
    if path and os.path.exists(path):
        os.remove(path)
        print(f"Removed checkpoint due to CUDA OOM error: {path}")


def finalize_training_checkpoint(trainer, seqrec_module, model_checkpoint: ModelCheckpoint,
                                 experiment: ExperimentTracker) -> None:

    if getattr(trainer, "interrupted", False):
        print("Detected interruption of training. Removed checkpoint.")
        os.remove(model_checkpoint.best_model_path)
        return

    checkpoint = torch.load(model_checkpoint.best_model_path)
    seqrec_module.load_state_dict(checkpoint["state_dict"])
    best_epoch = checkpoint["epoch"]
    seqrec_module._best_epoch = best_epoch  # consumed by the compute-budget capacity stat
    print(f"Model best epoch: {best_epoch}")
    experiment.log_scalar("num_best_epochs", best_epoch)
    print(f"Loaded checkpoint from: {model_checkpoint.best_model_path}")


def predict(trainer, seqrec_module, data, config, experiment: ExperimentTracker,
            semantic_ids=None, prefix='test'):

    start_time = time.time()

    extra_kwargs = {'semantic_ids': semantic_ids} if semantic_ids is not None else {}
    extra_kwargs = filter_kwargs(config.dataset_params.predict_dataset._target_, **extra_kwargs)
    predict_dataset = instantiate(config.dataset_params.predict_dataset, df=data, validation_mode=False, **extra_kwargs)

    left_padding = getattr(config.model, "left_padding", False)
    predict_loader = DataLoader(
        predict_dataset, shuffle=False,
        collate_fn=PaddingCollateFn(left_padding=left_padding),
        batch_size=config.dataloader.test_batch_size,
        num_workers=config.dataloader.num_workers)

    seqrec_module.predict_top_k = max(config.evaluator.top_k)
    preds = trainer.predict(model=seqrec_module, dataloaders=predict_loader)

    recs = preds2recs(preds)
    print('recs shape', recs.shape)
    predict_time = time.time() - start_time
    print(f"{prefix} prediction time", predict_time)

    experiment.log_scalar(f"{prefix}_predict_time", predict_time)
    if config[f"save_{prefix}_predictions"]:
        experiment.log_artifact(f"{prefix}_pred.csv", recs)

    return recs


def evaluate(recs, test, train, experiment: ExperimentTracker, config, prefix='test',
             warm_items=None, target_exposure=None, item_props=None):
    """Compute, log, and return ranking metrics for one set of recommendations.

    Metrics are computed over the whole user population, prefixed with ``prefix``, and
    logged as scalars, a table, and an artifact. Passing ``warm_items`` additionally
    logs the target-stratified breakdowns described in ``log_stratified_metrics``.

    Args:
        recs: recommendations table (user_id, item_id, score) as returned by
            ``src.postprocess.preds2recs``.
        test: ground-truth interactions to score against, one target row per user.
        train: training interactions, needed by the corpus-level metrics
            (Coverage / Surprisal / Novelty) to define the item population.
        experiment: tracker receiving the scalars, table, and artifact.
        config: run config; ``config.evaluator`` supplies the metric list, top_k, and
            the polars-metrics switch.
        prefix: metric-name prefix, ``'val'`` or ``'test'``.
        warm_items: item ids reachable by any training window. When given, the
            cold/warm split is logged; when None, only the population metrics are.
        target_exposure: expected per-epoch occurrences per item, indexed by item id.
            Enables the exposure-decile breakdown.
        item_props: per-item collision properties, indexed by item id. Enables the
            collision-category breakdown.

    Returns:
        dict mapping the prefixed metric name (e.g. ``'val_NDCG@10'``) to its value.
    """
    start_time = time.time()
    evaluator = Evaluator(metrics=list(config.evaluator.metrics),
                            top_k=list(config.evaluator.top_k),
                            polars_metrics=config.evaluator.polars_metrics)

    metrics_dict = evaluator.compute_metrics(test, recs, train)
    metrics_dict = {prefix + '_' + key: value for key, value in metrics_dict.items()}
    print('metrics\n', metrics_dict)

    for key, value in metrics_dict.items():
        experiment.log_scalar(key, value)

    metrics_df = pd.Series(metrics_dict).to_frame().reset_index()
    metrics_df.columns = ['metric_name', 'metric_value']
    experiment.log_table(df=metrics_df, title=f'{prefix}_metrics', series='dataframe')
    experiment.log_artifact(f'{prefix}_metrics', metrics_df)

    if warm_items is not None:
        log_stratified_metrics(recs, test, train, experiment, config, prefix,
                               warm_items, target_exposure=target_exposure,
                               item_props=item_props)

    eval_time = time.time() - start_time
    print(f"{prefix} evaluation time", eval_time)
    experiment.log_scalar(f"{prefix}_eval_time", eval_time)

    return metrics_dict


def _per_user_metrics(recs, test, train, config):
    """Per-user ranking metrics as a DataFrame (index=user_id, columns='NDCG@10', ...).

    Uses the Evaluator's PerUser mode so the split is computed once; corpus metrics
    are dropped because they are undefined per user.

    Args:
        recs: recommendations table, as passed to ``evaluate``.
        test: ground-truth interactions.
        train: training interactions.
        config: run config; ``config.evaluator`` supplies the metric list and top_k.

    Returns:
        DataFrame indexed by ``user_id`` with one column per ranking metric. Empty when
        the evaluator produced no per-user values.
    """
    ranking = [m for m in config.evaluator.metrics if m not in CORPUS_METRICS]
    evaluator = Evaluator(metrics=ranking, top_k=list(config.evaluator.top_k),
                          modes=['PerUser'], polars_metrics=config.evaluator.polars_metrics)
    per_user_dicts = evaluator.compute_metrics(test, recs, train)
    per_user = pd.DataFrame({
        name.replace('-PerUser', ''): pd.Series(values)
        for name, values in per_user_dicts.items()
    })
    per_user.index.name = 'user_id'
    return per_user


def log_stratified_metrics(recs, test, train, experiment: ExperimentTracker, config,
                           prefix, warm_items, target_exposure=None, item_props=None):
    """Log ranking metrics split by properties of each user's target item.

    Three breakdowns, all keyed off the single target item per user:

    * cold vs warm, by whether the target is reachable by any training window
      (``warm_items``). The catalog-cold share, which instead asks whether the target
      appears in ``train`` at all, is logged next to it for reference.
    * training-exposure deciles of the target, when ``target_exposure`` is given.
    * collision category of the target, when ``item_props`` is given.

    The last two report a single primary metric: NDCG at the main cutoff when the
    evaluator produced it, otherwise the first available metric. Nothing is logged when
    no per-user metrics could be computed.

    Args:
        recs: recommendations table, as passed to ``evaluate``.
        test: ground-truth interactions, one target row per user.
        train: training interactions, used for the catalog-cold reference share.
        experiment: tracker receiving the scalars and tables.
        config: run config; ``config.evaluator`` supplies the metric list and top_k.
        prefix: metric-name prefix, ``'val'`` or ``'test'``.
        warm_items: item ids reachable by any training window.
        target_exposure: expected per-epoch occurrences per item, indexed by item id.
        item_props: per-item collision properties, indexed by item id.
    """

    per_user = _per_user_metrics(recs, test, train, config)
    if per_user.empty:
        return
    metric_cols = list(per_user.columns)
    top_k = list(config.evaluator.top_k)
    k0 = 10 if 10 in top_k else min(top_k)
    primary = f"NDCG@{k0}" if f"NDCG@{k0}" in metric_cols else metric_cols[0]

    targets = test.drop_duplicates('user_id').set_index('user_id')['item_id']
    targets = targets.reindex(per_user.index)

    warm_set = set(int(i) for i in warm_items)
    is_warm = targets.isin(warm_set).to_numpy()
    catalog_set = set(int(i) for i in pd.unique(train['item_id']))
    is_catalog_warm = targets.isin(catalog_set).to_numpy()

    scalars = {}
    for col in metric_cols:
        scalars[f"{prefix}_all_{col}"] = float(per_user[col].mean())
        if is_warm.any():
            scalars[f"{prefix}_warm_{col}"] = float(per_user.loc[is_warm, col].mean())
        if (~is_warm).any():
            scalars[f"{prefix}_cold_{col}"] = float(per_user.loc[~is_warm, col].mean())
    scalars[f"{prefix}_cold_target_share"] = float((~is_warm).mean())
    scalars[f"{prefix}_cold_target_count"] = int((~is_warm).sum())
    scalars[f"{prefix}_warm_target_count"] = int(is_warm.sum())
    scalars[f"{prefix}_catalog_cold_target_share"] = float((~is_catalog_warm).mean())
    experiment.log_scalars(scalars)

    if target_exposure is not None:
        _log_exposure_deciles(experiment, prefix, per_user[primary].to_numpy(),
                              target_exposure.reindex(targets.to_numpy()).fillna(0.0).to_numpy(),
                              primary)

    if item_props is not None:
        _log_collision_strata(experiment, prefix, per_user[primary].to_numpy(),
                              item_props.reindex(targets.to_numpy()), primary)


def _log_exposure_deciles(experiment, prefix, metric_values, exposure_values, primary):
    """Log the mean primary metric per training-exposure decile of the target item.

    Users are bucketed by how often their target item is expected to appear in a
    training epoch, so ranking quality can be read against exposure. Returns without
    logging when there are too few distinct exposure values to cut deciles.

    Args:
        experiment: tracker receiving the per-decile scalars and the summary table.
        prefix: metric-name prefix, ``'val'`` or ``'test'``.
        metric_values: per-user values of the primary metric.
        exposure_values: expected per-epoch occurrences of each user's target item,
            aligned element-wise with ``metric_values``.
        primary: name of the primary metric, used in the logged keys.
    """
    try:
        deciles = pd.qcut(exposure_values, 10, labels=False, duplicates='drop')
    except (ValueError, IndexError):
        return
    tmp = pd.DataFrame({'decile': deciles, 'metric': metric_values, 'exposure': exposure_values})
    agg = tmp.groupby('decile').agg(mean_metric=('metric', 'mean'),
                                    count=('metric', 'size'),
                                    mean_exposure=('exposure', 'mean'))
    for d, row in agg.iterrows():
        experiment.log_scalars({
            f"{prefix}_exposure_decile/{primary}/d{int(d)}": float(row['mean_metric']),
            f"{prefix}_exposure_decile/count/d{int(d)}": int(row['count']),
            f"{prefix}_exposure_decile/mean_exposure/d{int(d)}": float(row['mean_exposure']),
        })
    experiment.log_table(df=agg.reset_index(), title=f"{prefix}_exposure_deciles", series="stats")


def _log_collision_strata(experiment, prefix, metric_values, props, primary):
    """Log the mean primary metric by the collision category of the target item.

    Targets fall into three categories: ``noncolliding`` when the item's semantic prefix
    is unique, ``colliding_first`` when it shares a prefix and holds rank 0 in that
    group, and ``colliding_nonfirst`` otherwise. Mean metric, share, and count are
    logged per category. Targets absent from ``props`` count as non-colliding.

    Args:
        experiment: tracker receiving the per-category scalars and the summary table.
        prefix: metric-name prefix, ``'val'`` or ``'test'``.
        metric_values: per-user values of the primary metric.
        props: per-item ``is_colliding`` / ``is_first`` flags reindexed onto each user's
            target item, so rows align with ``metric_values``.
        primary: name of the primary metric, used in the logged keys.
    """
    is_coll = props['is_colliding'].fillna(False).to_numpy()
    is_first = props['is_first'].fillna(False).to_numpy()
    category = np.where(~is_coll, 'noncolliding',
                        np.where(is_first, 'colliding_first', 'colliding_nonfirst'))
    tmp = pd.DataFrame({'category': category, 'metric': metric_values})
    agg = tmp.groupby('category').agg(mean_metric=('metric', 'mean'), count=('metric', 'size'))
    total = len(tmp)
    for cat, row in agg.iterrows():
        experiment.log_scalars({
            f"{prefix}_collision/{primary}/{cat}": float(row['mean_metric']),
            f"{prefix}_collision/share/{cat}": row['count'] / total,
            f"{prefix}_collision/count/{cat}": int(row['count']),
        })
    experiment.log_scalar(f"{prefix}_colliding_target_share", float(is_coll.mean()))
    experiment.log_table(df=agg.reset_index(), title=f"{prefix}_collision_strata", series="stats")


def filter_kwargs(target_str: str, **kwargs):
    """Leave in kwargs only arguments from __init__ of target class.

    Lets a caller offer optional extras such as ``semantic_ids`` to any target without
    knowing whether that particular class accepts them.

    Args:
        target_str: fully-qualified class path, as used by Hydra's ``_target_``.
        **kwargs: candidate keyword arguments.

    Returns:
        dict holding only the keys the target's signature accepts.
    """
    target_class = get_class(target_str)
    sig = inspect.signature(target_class)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}

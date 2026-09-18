import logging
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from runs._resolvers import register_resolvers
from src.run_utils import crop_embeddings_to_num_items, pandas_reader
from src.semantic_ids._quantizer import Quantizer
from src.semantic_ids.evaluation.evaluator import SemanticIdsEvaluator
from src.semantic_ids.collision_solvers import get_collision_solver
from src.semantic_ids.utils import prepare_sids


register_resolvers()


def load_embeddings(config: DictConfig):

    embeddings = np.load(os.path.join(config.embs_path, "embs.npy"))
    logging.info(f"loaded embeddings of shape {embeddings.shape} from {config.embs_path}")

    if config.crop_embeddings_to_num_items:
        embeddings = crop_embeddings_to_num_items(embeddings, config)
        logging.info(f"cropped embeddings to shape {embeddings.shape}")

    return embeddings.astype(np.float32)

# Note: this loader is specific to the Amazon Reviews metadata layout.
def load_categories(
    config: DictConfig,
    dataset_cfg: DictConfig,
    num_embedding_rows: int,
) -> pd.DataFrame | None:
    """Load ground-truth item categories aligned row-by-row with the embeddings.

    Mirrors the dedup/sort/crop logic of ``runs/build_semantic_embeddings.py``
    so that the i-th row of the returned DataFrame corresponds to the i-th
    row of the loaded embeddings tensor. Returns ``None`` when no
    ``categories_column`` is configured on the dataset.

    The category column in the metadata table is expected to hold an
    array-like of strings ordered from most general (level 1) to most
    specific. ``min_category_level`` / ``max_category_level`` (both 1-based,
    inclusive, read from ``dataset_cfg``) restrict the range of levels kept;
    out-of-range or missing levels for a given item become NaN.
    """
    column = OmegaConf.select(dataset_cfg, "categories_column", default=None)
    meta_path = OmegaConf.select(config, "meta_path", default=None)
    if column is None or meta_path is None:
        return None

    mapping_path = os.path.join(config.base_path, "item_mapping.pkl")
    with open(mapping_path, "rb") as f:
        item_mapping = pickle.load(f)

    logging.info(f"loading item categories from {meta_path}")
    meta = pandas_reader(meta_path)
    meta["item_id"] = meta["item_id"].map(item_mapping)
    meta = meta.dropna(subset=["item_id"])
    meta = meta.drop_duplicates(subset=["item_id"], keep="first")
    meta = meta.sort_values(by="item_id").reset_index(drop=True)

    if len(meta) < num_embedding_rows:
        raise ValueError(
            f"meta has {len(meta)} rows after dedup but embeddings have "
            f"{num_embedding_rows}; categories cannot be aligned"
        )
    meta = meta.iloc[:num_embedding_rows]

    categories = pd.DataFrame(meta[column].tolist())
    min_level = OmegaConf.select(dataset_cfg, "min_category_level", default=1)
    max_level = OmegaConf.select(dataset_cfg, "max_category_level", default=None)
    end = max_level if max_level is not None else categories.shape[1]
    categories = categories.iloc[:, min_level - 1: end]
    categories.columns = [f"cat{min_level + i}" for i in range(categories.shape[1])]

    logging.info(
        f"loaded categories of shape {categories.shape} "
        f"with columns {list(categories.columns)}"
    )
    return categories


def fit_quantizer(X: np.ndarray, config: DictConfig, experiment):

    quantizer: Quantizer = instantiate(config.quantizer)
    logging.info(
        f"initialized {quantizer.__class__.__name__} with "
        f"num_codebooks = {quantizer.num_codebooks} and "
        f"codebook_size = {quantizer.codebook_size}"
    )

    start_time = time.time()
    logging.info("fitting quantizer...")
    quantizer.fit(X)
    training_time = time.time() - start_time
    logging.info(f"finished quantizer fit in {round(training_time, 2)} seconds")
    experiment.log_scalar("fit_time", training_time)

    return quantizer


def generate_sids(X: np.ndarray, quantizer: Quantizer, config: DictConfig, experiment):

    logging.info("generating codes...")
    clusters = quantizer.encode(X)
    semantics_mapping = {int(idx): cluster_ids.tolist() for idx, cluster_ids in enumerate(clusters)}

    start_time = time.time()
    logging.info("solving collisions...")
    collision_codebook_size = OmegaConf.select(
        config, "collision_codebook_size", default=quantizer.codebook_size
    )

    solver = OmegaConf.select(config, "collision_solver", default="sequential")
    solver_obj = get_collision_solver(
        solver,
        codebook_size=collision_codebook_size,
        extend_codebook=OmegaConf.select(config, "extend_codebook", default="global"),
    )

    semantics_mapping = solver_obj.fit_transform(semantics_mapping)
    max_collision_fact = solver_obj.max_collision_fact_
    # Sequential codes span 0..max_collision_fact-1, including overflow groups.
    collision_codebook_size = max(collision_codebook_size, max_collision_fact)
    logging.info(f"solved collisions in {round(time.time() - start_time, 2)} seconds")
    experiment.log_scalar("max_collision_fact", max_collision_fact)
    # Log the effective size of the appended collision column.
    experiment.log_scalar("collision_codebook_size_used", collision_codebook_size)

    # Prepare SIDs
    order = torch.as_tensor(list(int(key) for key in semantics_mapping.keys()))
    # Note: prepare_sids operates on torch tensors, not np.ndarray.
    raw_sids = torch.tensor(list(semantics_mapping.values()))[order, :]
    sids = prepare_sids(
        raw_sids,
        num_embeddings_per_codebook=quantizer.codebook_size,
        shift=1,
    )
    print(f"Prepared SIDs shape: {sids.shape}")

    return semantics_mapping, raw_sids, sids, collision_codebook_size


def evaluate_sids(raw_sids: torch.Tensor, X: np.ndarray, quantizer: Quantizer,
                  collision_codebook_size: int, config: DictConfig, experiment,
                  categories: pd.DataFrame | None = None):

    codebook_sizes = [quantizer.codebook_size] * quantizer.num_codebooks + [collision_codebook_size]

    logging.info("Evaluating semantic IDs quality...")
    evaluator = SemanticIdsEvaluator(
        sids=raw_sids,
        embeddings=X,
        codebook_sizes=codebook_sizes,
        quantizer=quantizer,
        experiment=experiment,
        categories=categories,
        **config.evaluator,
    )
    evaluator.run_and_log_all()

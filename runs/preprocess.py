"""
Preprocessing pipeline
"""

import json
import logging
import os

import hydra
import pandas as pd
import polars as pl
from omegaconf import DictConfig, OmegaConf

from src.preprocess.filters import (apply_last_n_days_filter,
                                    apply_min_relevance_filter,
                                    apply_n_core_filter, sample_users)
from src.preprocess.utils import dataset_stats, set_default_col_names
from src.run_utils import dataframe_reader, dataframe_writer


def preprocess(
    data: pd.DataFrame | pl.DataFrame,
    n_core_filtering: bool = True,
    rm_consecutive_dups: bool = False,
    item_min_count: int = 5,
    user_min_count: int = 5,
    min_relevance: float | int | None = None,
    num_users: int | None = None,
    last_n_days: float | int | None = None,
    user_col: str = "user_id",
    item_col: str = "item_id",
    time_col: str = "timestamp",
    relevance_col: str | None = "rating",
    meta_info: pd.DataFrame | pl.DataFrame | None = None,
) -> tuple[pd.DataFrame | pl.DataFrame, pd.DataFrame | pl.DataFrame | None]:
    """
    - optional crop to the last N days
    - columns renaming
    - N-core or N-filter for items and sequences along with iterative
    - removal of consecutive interactions with the same item
    - optional label encoding of users only; item_id is left as in the source data
    """

    columns = [key for key in [user_col, item_col, time_col, relevance_col] if key is not None]
    data = data[columns]

    data = set_default_col_names(data, user_col, item_col, time_col, relevance_col)

    logging.info("Raw data:\n%s", dataset_stats(data, extended=True))

    # Before every count-based filter, so item_min_count and user_min_count are counted
    # inside the retained window rather than over the whole history.
    if last_n_days is not None:
        logging.info(f"keeping the last {last_n_days} days of interactions")
        logging.info("before cropping:\n%s", dataset_stats(data))
        data = apply_last_n_days_filter(data, last_n_days)
        logging.info("after cropping:\n%s", dataset_stats(data))

    if min_relevance is not None:
        logging.info(f"filtering out events with {relevance_col!r} < {min_relevance}")
        logging.info("before filtering:\n%s", dataset_stats(data))
        data = apply_min_relevance_filter(data, min_relevance)
        logging.info("after filtering:\n%s", dataset_stats(data))

    if num_users is not None:
        logging.info(f"subsampling {num_users} users")
        logging.info("before subsampling:\n%s", dataset_stats(data))
        data = sample_users(data, num_users)
        logging.info("after subsampling:\n%s", dataset_stats(data))

    if n_core_filtering:
        logging.info("applying N-core filtering")
        logging.info("before N-core filtering:\n%s", dataset_stats(data))
        data = apply_n_core_filter(
            events=data,
            item_min_count=item_min_count,
            user_min_count=user_min_count,
            rm_consecutive_dups=rm_consecutive_dups,
        )
        logging.info("after N-core filtering:\n%s", dataset_stats(data, extended=True))

    elif not n_core_filtering:
        logging.info("Skipping filtering")
    else:
        raise NotImplementedError("N-core filtering is only one available")

    if meta_info is not None:
        meta_info = set_default_col_names(meta_info, user_col, item_col, time_col, relevance_col)

    return data, meta_info


@hydra.main(config_path="config", config_name="preprocess")
def main(config: DictConfig) -> None:

    print(OmegaConf.to_yaml(config, resolve=True))

    data_format = OmegaConf.select(config, "dataset.data_format", default="csv")
    data_path = os.environ["SEQ_REC_DATA_PATH"]
    raw_dir = os.path.join(data_path, "raw")

    dataset_name = config.dataset.name
    logging.info(f"started preprocessing for dataset {dataset_name}")

    data = dataframe_reader(os.path.join(raw_dir, dataset_name, f"{dataset_name}.{data_format}"), engine=config.engine)

    meta_info_path = os.path.join(raw_dir, dataset_name, f"{dataset_name}_meta.{data_format}")
    if os.path.exists(meta_info_path):
        meta_info = dataframe_reader(meta_info_path, engine=config.engine)
    else:
        meta_info = None

    data, meta_info = preprocess(
        data=data,
        **config.preprocessing_params,
        **config.dataset.column_name,
        meta_info=meta_info,
    )

    if getattr(config, "suffix", None) is not None:
        dataset_name = f"{dataset_name}_{config.suffix}"
    path_to_save = os.path.join(data_path, "preprocessed", dataset_name)
    os.makedirs(path_to_save, exist_ok=True)

    # save config
    OmegaConf.save(config, os.path.join(path_to_save, "config.yaml"), resolve=True)
    # save data
    dataframe_writer(data, os.path.join(path_to_save, f"{dataset_name}.parquet"), engine=config.engine)
    # save metadata
    if meta_info is not None:
        dataframe_writer(meta_info, os.path.join(path_to_save, f"{dataset_name}_meta.parquet"), engine=config.engine)
    # save statistics
    stats = dataset_stats(data, extended=True)
    with open(os.path.join(path_to_save, "statistics.json"), "w") as f:
        json.dump(stats, f)

if __name__ == "__main__":

    main()

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

from src.preprocess.filters import (
    apply_last_n_days_filter,
    apply_min_relevance_filter,
    apply_n_core_filter,
    remove_consecutive_duplicates,
    sample_users,
)
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
    audit: dict | None = None,
    timestamp_unit: str = "s",
) -> tuple[pd.DataFrame | pl.DataFrame, pd.DataFrame | pl.DataFrame | None]:
    """Clean source records before optional user sampling and N-core filtering.

    Exact duplicates keep the last source occurrence; consecutive same-item runs
    keep the first chronological event. Source order resolves timestamp ties.
    ``_source_row`` is the zero-based input row position, retained for event identity.
    If supplied, ``audit`` receives counts after each cleaning step.
    Cleaning uses the configured source columns; output column names are normalized.
    """

    columns = [user_col, item_col, time_col]
    if relevance_col is not None:
        columns.append(relevance_col)
    if "_source_row" in data.columns:
        raise ValueError("_source_row is reserved for the original input row position")
    source_columns = list(data.columns)
    if isinstance(data, pd.DataFrame):
        if data[[user_col, item_col, time_col]].isna().any().any():
            raise ValueError("User IDs, item IDs, and timestamps must not be missing")
        data = data.assign(_source_row=range(len(data)))
    else:
        if any(data[col].null_count() or (data[col].dtype.is_float() and data[col].is_nan().any())
               for col in (user_col, item_col, time_col)):
            raise ValueError("User IDs, item IDs, and timestamps must not be missing")
        data = data.with_row_index("_source_row").with_columns(pl.col("_source_row").cast(pl.Int64))

    steps = []
    if audit is not None:
        audit["steps"] = steps

    def record_step(name):
        stats = dataset_stats(data, user_id=user_col, item_id=item_col, timestamp=time_col)
        stats["removed_events"] = steps[-1]["n_interactions"] - len(data) if steps else 0
        steps.append({"step": name, **stats})
        logging.info("%s: %s", name, stats)

    record_step("raw")

    # Before every count-based filter, so item_min_count and user_min_count are counted
    # inside the retained window rather than over the whole history.
    if last_n_days is not None:
        data = apply_last_n_days_filter(data, last_n_days, time_col=time_col, timestamp_unit=timestamp_unit)
        record_step("last_n_days")

    if min_relevance is not None:
        if relevance_col is None:
            raise ValueError("min_relevance requires a relevance_col")
        data = apply_min_relevance_filter(data, min_relevance, relevance_col=relevance_col)
        record_step("min_relevance")

    # Compare all source fields before projecting model columns. Generated row
    # identity must not prevent identical source records from matching.
    if isinstance(data, pd.DataFrame):
        data = data.drop_duplicates(subset=source_columns, keep="last")
    else:
        data = data.unique(subset=source_columns, keep="last", maintain_order=True)
    record_step("exact_duplicates")
    data = data[columns + ["_source_row"]]

    if rm_consecutive_dups:
        data = remove_consecutive_duplicates(data, user_col=user_col, item_col=item_col, time_col=time_col)
        record_step("consecutive_repeats")
    elif isinstance(data, pd.DataFrame):
        data = data.sort_values([user_col, time_col], kind="stable")
    else:
        data = data.sort([user_col, time_col], maintain_order=True)

    if num_users is not None:
        data = sample_users(data, num_users, user_col=user_col)
        record_step("sample_users")

    if n_core_filtering:
        data = apply_n_core_filter(
            events=data,
            item_min_count=item_min_count,
            user_min_count=user_min_count,
            rm_consecutive_dups=rm_consecutive_dups,
            user_col=user_col,
            item_col=item_col,
            time_col=time_col,
        )
        record_step("n_core")

    data = set_default_col_names(data, user_col, item_col, time_col, relevance_col)
    if meta_info is not None:
        meta_info = set_default_col_names(meta_info, user_col, item_col, time_col, relevance_col)

    return data, meta_info


@hydra.main(config_path="config", config_name="preprocess")
def main(config: DictConfig) -> None:

    print(OmegaConf.to_yaml(config, resolve=True))

    data_format = OmegaConf.select(config, "dataset.data_format", default="csv")
    timestamp_unit = OmegaConf.select(config, "dataset.timestamp_unit", default="s")
    data_path = os.environ["SEQ_REC_DATA_PATH"]
    raw_dir = os.path.join(data_path, "raw")

    dataset_name = config.dataset.name
    output_name = f"{dataset_name}_{config.suffix}" if config.suffix is not None else dataset_name
    path_to_save = os.path.join(data_path, "preprocessed", output_name)
    if os.path.exists(path_to_save):
        raise FileExistsError(f"Output directory already exists: {path_to_save}. Choose another data root or suffix.")
    logging.info(f"started preprocessing for dataset {dataset_name}")

    data = dataframe_reader(os.path.join(raw_dir, dataset_name, f"{dataset_name}.{data_format}"), engine=config.engine)

    meta_info_path = os.path.join(raw_dir, dataset_name, f"{dataset_name}_meta.{data_format}")
    if os.path.exists(meta_info_path):
        meta_info = dataframe_reader(meta_info_path, engine=config.engine)
    else:
        meta_info = None

    audit = {}
    data, meta_info = preprocess(
        data=data,
        **config.preprocessing_params,
        **config.dataset.column_name,
        meta_info=meta_info,
        audit=audit,
        timestamp_unit=timestamp_unit,
    )

    dataset_name = output_name
    os.makedirs(path_to_save, exist_ok=False)

    # save config
    OmegaConf.save(config, os.path.join(path_to_save, "config.yaml"), resolve=True)
    # save data
    dataframe_writer(data, os.path.join(path_to_save, f"{dataset_name}.parquet"), engine=config.engine)
    # save metadata
    if meta_info is not None:
        dataframe_writer(meta_info, os.path.join(path_to_save, f"{dataset_name}_meta.parquet"), engine=config.engine)
    # save statistics
    stats = dataset_stats(data, extended=True, timestamp_unit=timestamp_unit)
    stats["steps"] = audit["steps"]
    with open(os.path.join(path_to_save, "statistics.json"), "w") as f:
        json.dump(stats, f, indent=2, allow_nan=False)

if __name__ == "__main__":

    main()

"""
Preprocessing utils.
"""
import pandas as pd
import polars as pl


def calculate_sequence_stats(lengths, prefix=''):
    """
    prefix: prefix for statistic names (e.g., 'input_' or 'gt_')
    """
    stats = {
        f'{prefix}mean': lengths.mean(),
        f'{prefix}std': lengths.std(),
        f'{prefix}min': lengths.min(),
        f'{prefix}max': lengths.max(),
        f'{prefix}median': lengths.median()
    }
    return stats


# === DATASET STATS ===
def dataset_stats(
    data: pd.DataFrame | pl.DataFrame,
    extended: bool = False,
    user_id: str = "user_id",
    item_id: str = "item_id",
    timestamp: str = "timestamp",
) -> dict:

    if isinstance(data, pd.DataFrame):
        return _pandas_dataset_stats(data, extended, user_id, item_id, timestamp)
    elif isinstance(data, pl.DataFrame):
        return _polars_dataset_stats(data, extended, user_id, item_id, timestamp)
    else:
        raise TypeError


def _pandas_dataset_stats(
    data, extended=False, user_id="user_id", item_id="item_id", timestamp="timestamp"
):
    """Compute dataset statistics for a pandas dataframe.

    :param data: pandas dataframe
    :param extended: if calc #items per user and vise versa, defaults to False
    :param user_id: user col name, defaults to 'user_id'
    :param item_id: item col name, defaults to 'item_id'
    :param timestamp: timestamp col name, defaults to 'timestamp'
    :return: statistics dict
    """
    n_users = data[user_id].nunique()
    n_items = data[item_id].nunique()
    n_interactions = len(data)
    seq_lengths = data.groupby(user_id).size()

    stats = {
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": n_interactions,
        "density": n_interactions / (n_users * n_items),
        "avg_seq_length": seq_lengths.mean(),
    }

    if extended:
        stats.update(calculate_sequence_stats(seq_lengths, prefix='seq_len_'))

        item_counts = data[item_id].value_counts()
        stats.update(calculate_sequence_stats(item_counts, prefix='item_occurrence_'))

        user_counts = data[user_id].value_counts()
        stats.update(calculate_sequence_stats(user_counts, prefix='seq_len_'))

        # Temporal statistics
        stats["max_timestamp"] = data[timestamp].max()
        stats["min_timestamp"] = data[timestamp].min()
        stats["timestamp_range_in_days"] = (stats["max_timestamp"] - stats["min_timestamp"]) / (60 * 60 * 24)
    return stats


def _polars_dataset_stats(
    data: pl.DataFrame,
    extended: bool = False,  # ignored
    user_id: str = "user_id",
    item_id: str = "item_id",
    timestamp: str = "timestamp",  # ignored
) -> dict:

    return data.select(
        pl.len().alias("Num. Events"),
        pl.col(user_id).n_unique().alias("Num. Users"),
        pl.col(item_id).n_unique().alias("Num. Items"),
        (pl.len() / pl.col(user_id).n_unique()).alias("Avg. Length"),
    ).to_dicts()[0]


# === SET DEFAULT COLUMN NAMES ===
def set_default_col_names(
    data: pd.DataFrame | pl.DataFrame,
    user_col: str | None = "user_id",
    item_col: str | None = "item_id",
    time_col: str | None = "timestamp",
    relevance_col: str | None = "rating",
) -> pd.DataFrame | pl.DataFrame:

    mapping = {
        source_col_name: target_col_name
        for source_col_name, target_col_name in (
            (user_col, "user_id"),
            (item_col, "item_id"),
            (time_col, "timestamp"),
            (relevance_col, "rating"),
        )
        if source_col_name is not None
    }

    if isinstance(data, pd.DataFrame):
        return data.rename(columns=mapping)
    elif isinstance(data, pl.DataFrame):
        return data.rename(mapping, strict=False)
    raise TypeError

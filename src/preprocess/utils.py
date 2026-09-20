"""
Preprocessing utils.
"""
import pandas as pd
import polars as pl


TIMESTAMP_UNITS_PER_DAY = {"s": 86400, "ms": 86400000}


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
    timestamp_unit: str = "s",
) -> dict:
    """Dataset counts and optional sequence/item-frequency statistics.

    If the timestamp column is present, ties count rows beyond the first per
    user/time. Extended distributions give exact event counts and their numbers
    of users or items.
    Day ranges use ``timestamp_unit`` (s or ms); timestamp bounds stay unchanged.
    Both engines return the same keys and JSON-compatible values.
    """
    if isinstance(data, pd.DataFrame):
        stats = _pandas_dataset_stats(data, extended, user_id, item_id, timestamp)
    elif isinstance(data, pl.DataFrame):
        stats = _polars_dataset_stats(data, extended, user_id, item_id, timestamp)
    else:
        raise TypeError

    for key, value in stats.items():
        if isinstance(value, list):
            continue
        if pd.isna(value):
            stats[key] = None
        elif hasattr(value, "item"):
            stats[key] = value.item()
    if extended:
        stats["timestamp_range_in_days"] = (
            (stats["max_timestamp"] - stats["min_timestamp"]) / TIMESTAMP_UNITS_PER_DAY[timestamp_unit]
            if stats["n_interactions"] else None
        )
    return stats


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
        "density": n_interactions / (n_users * n_items) if n_users and n_items else None,
        "avg_seq_length": seq_lengths.mean(),
    }
    if timestamp in data.columns and extended:
        stats.update({
            "max_timestamp": data[timestamp].max(),
            "min_timestamp": data[timestamp].min(),
            "n_timestamp_ties": data.duplicated([user_id, timestamp]).sum(),
        })

    if extended:
        item_counts = data[item_id].value_counts()
        for counts, prefix, population in (
            (seq_lengths, "seq_len_", "n_users"),
            (item_counts, "item_occurrence_", "n_items"),
        ):
            stats.update(calculate_sequence_stats(counts, prefix))
            stats[prefix + "distribution"] = (
                counts.value_counts().sort_index().rename_axis("event_count")
                .reset_index(name=population).to_dict("records")
            )
    return stats


def _polars_dataset_stats(
    data: pl.DataFrame,
    extended: bool = False,
    user_id: str = "user_id",
    item_id: str = "item_id",
    timestamp: str = "timestamp",
) -> dict:

    stats = data.select(
        pl.len().alias("n_interactions"),
        pl.col(user_id).n_unique().alias("n_users"),
        pl.col(item_id).n_unique().alias("n_items"),
        (pl.len() / (pl.col(user_id).n_unique().cast(pl.UInt64) * pl.col(item_id).n_unique())).alias("density"),
        (pl.len() / pl.col(user_id).n_unique()).alias("avg_seq_length"),
    ).to_dicts()[0]
    if timestamp in data.columns and extended:
        stats.update(data.select(
            pl.col(timestamp).max().alias("max_timestamp"),
            pl.col(timestamp).min().alias("min_timestamp"),
            (pl.len() - pl.struct(user_id, timestamp).n_unique()).alias("n_timestamp_ties"),
        ).to_dicts()[0])
    
    if extended:
        for col, prefix, population in (
            (user_id, "seq_len_", "n_users"),
            (item_id, "item_occurrence_", "n_items"),
        ):
            counts = data.group_by(col).len(name="event_count")
            stats.update(calculate_sequence_stats(counts["event_count"], prefix))
            stats[prefix + "distribution"] = (
                counts.group_by("event_count").len(name=population)
                .sort("event_count").to_dicts()
            )
    return stats


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

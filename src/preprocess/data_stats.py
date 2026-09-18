from typing import Dict, Optional, Union

import pandas as pd
import polars as pl


def get_deltas(data: pd.DataFrame, col="user_id", timestamp="timestamp") -> pd.DataFrame:
    """
    Computes the time difference (delta) between successive interactions for each enity (user or item).

    The delta is calculated as the time difference (in seconds) between
    each interaction and the previous interaction for the same entity.

    Args:
        data (pd.DataFrame): DataFrame with 'user_id', 'item_id' and 'timestamp' columns.
        col (str, optional): Name of the column used to group entities (e.g., user ID). Defaults to "user_id".
        timestamp (str, optional): Name of the timestamp column. Defaults to "timestamp".

    Returns:
        DataFrame: The original DataFrame with an added 'delta' column.
    """
    data = data.copy().reset_index(drop=True)

    # Calculate time difference between consecutive interactions per user
    data["delta"] = (
        data.sort_values([col, timestamp], kind="stable")
        .groupby(col)[timestamp]
        .diff()
    )
    return data


def get_ts_collisions(data: pd.DataFrame, user_id: str = "user_id", timestamp: str = "timestamp") -> Optional[pd.DataFrame]:
    """
    Adds a flag column indicating duplicate timestamp interactions.

    Args:
        data: DataFrame containing user interactions with columns: user_id, timestamp
        user_id (str, optional): Name of the user ID column.
        timestamp (str, optional): Name of the timestamp column.

    Returns:
        DataFrame with 'timestamp_collisions' column added
    """
    data = data.copy()
    data['timestamp_collisions'] = data.duplicated(subset=[user_id, timestamp], keep='first')

    return data


def calc_lifetime(
    data: pd.DataFrame, timestamp: str = "timestamp", col: str = "user_id"
) -> pd.DataFrame:
    """
    Calculate the lifetime (in days) of each entity (e.g., user or session) in a DataFrame,
    based on the minimum and maximum timestamps.

    Args:
        data (pd.DataFrame): Interaction DataFrame.
        timestamp (str, optional): Name of the timestamp column.
        col (str, optional): Name of the column used to group entities (e.g., user ID). Defaults to "user_id".

    Returns:
        pd.DataFrame: A DataFrame with the entity column, minimum timestamp, maximum timestamp,
                      and calculated lifetime in days.
    """
    duration = data.groupby(col)[timestamp].agg(min_ts="min", max_ts="max")
    duration["lifetime"] = duration["max_ts"] - duration["min_ts"]
    return duration.reset_index()


def get_mean_median(series: pd.Series, prefix: str = "") -> Dict[str, float]:
    """
    Compute mean and median of a series.

    Args:
        series (pd.Series): Series to compute mean and median of.
        prefix (str, optional): Prefix to prepend to each statistic name.

    Returns:
        Dictionary of mean and median.
    """
    return {
        f"mean_{prefix}": series.mean(),
        f"median_{prefix}": series.median(),
    }


def count_delta_stats(data: pd.DataFrame, col="user_id", timestamp="timestamp") -> Dict[str, float]:
    """
    Compute time between interactions statistics (mean and median) from timestamped data.

    The delta is calculated as the time difference (in seconds) between
    each interaction and the previous interaction of a user.

    Args:
        data (pd.DataFrame): DataFrame containing timestamped records.

    Returns:
        Dictionary with mean and median time between interactions.
    """
    deltas = get_deltas(data, col=col, timestamp=timestamp)
    return get_mean_median(deltas["delta"], prefix="time_between_interactions")


def temporal_stats(data: pd.DataFrame,
                   user_id: str = "user_id",
                   item_id: str = "item_id",
                   timestamp: str = "timestamp") -> Dict[str, Union[int, float]]:
    """
    Compute temporal statistics for a dataset.

    Args:
        data (pd.DataFrame): Interaction DataFrame.
        timestamp (str, optional): Name of the timestamp column.
        user_id (str, optional): Name of the user ID column.
        item_id (str, optional): Name of the item ID column.

    Returns:
        Dictionary of temporal statistics.
    """
    stats = {}
    # Temporal statistics
    stats.update(
        {
            "max_timestamp": data[timestamp].max(),
            "min_timestamp": data[timestamp].min(),
        }
    )
    stats["timeframe"] = data[timestamp].max() - data[timestamp].min()
    stats.update(count_delta_stats(data, col=user_id, timestamp=timestamp))
    # User lifetime stats (in days)
    user_lifetimes = calc_lifetime(data, timestamp, user_id)
    stats.update(get_mean_median(user_lifetimes["lifetime"], prefix="user_lifetime"))
    stats["mean_user_lifetime, %"] = (
        stats["mean_user_lifetime"] * 100 / stats["timeframe"]
    )
    # Item lifetime stats (in days)
    item_lifetimes = calc_lifetime(data, timestamp, item_id)
    stats.update(get_mean_median(item_lifetimes["lifetime"], prefix="item_lifetime"))

    stats["mean_item_lifetime, %"] = (
        stats["mean_item_lifetime"] * 100 / stats["timeframe"]
    )

    ts_collisions = get_ts_collisions(data, user_id, timestamp)['timestamp_collisions']
    stats["timestamp_collisions"] = ts_collisions.sum()
    stats["timestamp_collisions, %"] = ts_collisions.mean() * 100

    return stats


def base_stats(
    data: Union[pd.DataFrame, pl.DataFrame],
    extended: bool = False,
    user_id: str = "user_id",
    item_id: str = "item_id",
    timestamp: str = "timestamp",
    return_df: bool = True,
) -> Union[pl.DataFrame, pl.Series]:
    """
    Compute dataset-level statistics for user-item interaction data.

    Args:
        data: Interaction table as Polars DataFrame, or Pandas DataFrame (converted to Polars).
        extended (bool): Whether to compute advanced statistics (default: False).
        user_id (str, optional): Name of the user ID column.
        item_id (str, optional): Name of the item ID column.
        timestamp (str, optional): Name of the timestamp column.
        return_df: If True, return one-row Pandas DataFrame; otherwise a Series.

    Returns:
        Polars DataFrame or Series of dataset-level statistics.
    """
    if isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data)
    elif not isinstance(data, pl.DataFrame):
        raise TypeError(f"expected pandas or polars DataFrame, got {type(data)!r}")

    agg = data.select(
        pl.col(user_id).n_unique().alias("n_users"),
        pl.col(item_id).n_unique().alias("n_items"),
        pl.len().alias("n_interactions"),
        (pl.col(timestamp).max() - pl.col(timestamp).min()).alias("timeframe"),
    ).to_dicts()[0]
    n_users = agg["n_users"]
    n_items = agg["n_items"]
    n_interactions = agg["n_interactions"]
    duration_days = agg["timeframe"]

    avg_seq_length = (
        data.group_by(user_id).len().select(pl.col("len").mean()).item()
    )

    stats: Dict[str, Union[int, float]] = {
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": n_interactions,
        "avg_seq_length": avg_seq_length,
        "density": n_interactions / (n_users * n_items),
        "timeframe": duration_days,
    }
    if extended:
        stats.update(temporal_stats(data.to_pandas(), user_id, item_id, timestamp))
        item_len = data.group_by(item_id).len()["len"]
        stats["mean_item_occurrence"] = item_len.mean()
        stats["median_item_occurrence"] = item_len.median()

        user_len = data.group_by(user_id).len()["len"]
        stats["mean_user_activity"] = user_len.mean()
        stats["median_user_activity"] = user_len.median()

    if return_df:
        return pd.DataFrame([stats])
    return pd.Series(stats)

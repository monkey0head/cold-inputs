"""Filter interactions"""

import logging

import numpy as np
import pandas as pd
import polars as pl

from src.preprocess.utils import TIMESTAMP_UNITS_PER_DAY, dataset_stats


# === APPLY LAST N DAYS FILTER ===
def apply_last_n_days_filter(
    events: pd.DataFrame | pl.DataFrame,
    last_n_days: float | int,
    time_col: str = "timestamp",
    timestamp_unit: str = "s",
) -> pd.DataFrame | pl.DataFrame:
    """
    Keeps only the interactions from the last `last_n_days` days of the data.

    The cutoff is taken from the maximum timestamp present, so the window ends where the data
    ends and a later split on absolute thresholds is unaffected.

    Args:
        events (pd.DataFrame | pl.DataFrame): Input events dataframe (pandas or polars).
        last_n_days (float | int): Width of the retained window, in days.
        time_col (str, optional): Column name for timestamps. Defaults to "timestamp".
        timestamp_unit (str, optional): Timestamp unit, "s" or "ms". Defaults to "s".

    Returns:
        pd.DataFrame | pl.DataFrame: Events at or after the cutoff.
    """
    window = last_n_days * TIMESTAMP_UNITS_PER_DAY[timestamp_unit]
    if isinstance(events, pd.DataFrame):
        cutoff = events[time_col].max() - window
        return events[events[time_col] >= cutoff]
    elif isinstance(events, pl.DataFrame):
        cutoff = events[time_col].max() - window
        return events.filter(pl.col(time_col) >= cutoff)
    else:
        raise TypeError(f"expected either `pandas.DataFrame` or `polars.DataFrame`, got {type(events)!r}")


# === APPLY N CORE FILTER ===
def apply_n_core_filter(
    events: pd.DataFrame | pl.DataFrame,
    min_count: int | None = None,
    user_min_count: int | None = None,
    item_min_count: int | None = None,
    rm_consecutive_dups: bool = False,
    user_col: str = "user_id",
    item_col: str = "item_id",
    time_col: str = "timestamp",
) -> pd.DataFrame | pl.DataFrame:
    """
    Iteratively filters the events dataframe to enforce N-core constraints on users and items.

    Keeps only users and items that have appeared at least a minimum number of times. This operation
    is repeated until the dataframe is stable (i.e., filtering does not further change the number of rows).
    Optionally, removes consecutive duplicate user-item interactions.

    Args:
        events (pd.DataFrame | pl.DataFrame): Input events dataframe (pandas or polars).
        min_count (int | None, optional): Minimum number of interactions required for both users and items.
            If set, overrides `user_min_count` and `item_min_count`. Defaults to None.
        user_min_count (int | None, optional): Minimum number of interactions required per user.
            Required if `min_count` is not specified. Defaults to None.
        item_min_count (int | None, optional): Minimum number of interactions required per item.
            Required if `min_count` is not specified. Defaults to None.
        rm_consecutive_dups (bool, optional): Whether to remove consecutive duplicate user-item interactions.
            Defaults to False.
        user_col (str, optional): Name of the column identifying users. Defaults to "user_id".
        item_col (str, optional): Name of the column identifying items. Defaults to "item_id".
        time_col (str, optional): Name of the column specifying chronological order. Defaults to "timestamp".

    Returns:
        pd.DataFrame | pl.DataFrame: Filtered dataframe.
    """
    if min_count is None:
        if user_min_count is None or item_min_count is None:
            raise ValueError("if `min_count` is not specified, both `user_min_count` and `item_min_count` must be provided.")  # fmt: skip
    else:
        if user_min_count is not None or item_min_count is not None:
            logging.warning("`user_min_count` and `item_min_count` are overridden by `min_count`.")

        user_min_count = item_min_count = min_count

    if rm_consecutive_dups:
        events = remove_consecutive_duplicates(events, user_col, item_col, time_col)
        print("After consecutive repeats filtering")
        print(dataset_stats(events, user_id=user_col, item_id=item_col, timestamp=time_col))

    step = 1
    height = -1
    while len(events) > 0 and len(events) != height:
        height = len(events)

        events = apply_min_count_filter(events, user_min_count, user_col)
        events = apply_min_count_filter(events, item_min_count, item_col)

        if rm_consecutive_dups:
            events = remove_consecutive_duplicates(events, user_col, item_col, time_col)

        print(f"After n-core filtering on step {step}")
        print(dataset_stats(events, user_id=user_col, item_id=item_col, timestamp=time_col))
        step += 1

    return events


# === REMOVE CONSECUTIVE DUPLICATES ===
def remove_consecutive_duplicates(
    events: pd.DataFrame | pl.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    time_col: str = "timestamp",
) -> pd.DataFrame | pl.DataFrame:
    """
    Removes consecutive duplicate item interactions for each user. 
    For each user, if an item appears consecutively in the sorted order (by timestamp), 
    only the first occurrence is retained.

    Args:
        events (pd.DataFrame | pl.DataFrame): Interaction events data.
        user_col (str, optional): Column name for user IDs. Defaults to "user_id".
        item_col (str, optional): Column name for item IDs. Defaults to "item_id".
        time_col (str, optional): Column name specifying chronological order. Defaults to "timestamp".

    Returns:
        pd.DataFrame | pl.DataFrame: Events dataframe with consecutive duplicate user-item interactions removed.
    """
    if isinstance(events, pd.DataFrame):
        return _pandas_remove_consecutive_duplicates(events, user_col, item_col, time_col)
    elif isinstance(events, pl.DataFrame):
        return _polars_remove_consecutive_duplicates(events, user_col, item_col, time_col)
    else:
        raise TypeError(f"expected either `pandas.DataFrame` or `polars.DataFrame`, got {type(events)!r}")


def _pandas_remove_consecutive_duplicates(
    events: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    time_col: str = "timestamp",
) -> pd.DataFrame:

    events = events.sort_values([user_col, time_col], kind="stable")
    events["__shifted__"] = events.groupby(user_col)[item_col].shift(periods=1)
    return events[events[item_col] != events["__shifted__"]].drop("__shifted__", axis=1).reset_index(drop=True)


def _polars_remove_consecutive_duplicates(
    events: pl.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    time_col: str = "timestamp",
) -> pl.DataFrame:

    return (
        events.sort(user_col, time_col, maintain_order=True).with_columns(
            (pl.col(item_col) != pl.col(item_col).shift())
            .fill_null(True)
            .cum_sum()
            .over(user_col)
            .alias("__consecutive_series_id__")
        ).group_by(user_col, "__consecutive_series_id__", maintain_order=True)
        .agg(pl.all().first())
        .drop("__consecutive_series_id__")
    )  # fmt: skip


# === APPLY MIN COUNT FILTER ===
def apply_min_count_filter(
    events: pd.DataFrame | pl.DataFrame,
    min_count: int,
    col: str,
) -> pd.DataFrame | pl.DataFrame:
    """
    Filters the input DataFrame by keeping only the rows where the group defined by `col`
    has at least `min_count` occurrences.

    Args:
        events (pd.DataFrame | pl.DataFrame): Input DataFrame (either pandas or polars) containing event data.
        min_count (int): Minimum number of occurrences required to keep groups defined by `col`.
        col (str): Name of the column to group by for filtering.

    Returns:
        pd.DataFrame | pl.DataFrame: Filtered DataFrame containing only groups with at least `min_count` items.
    """
    if isinstance(events, pd.DataFrame):
        return _pandas_apply_min_count_filter(events, min_count, col)
    elif isinstance(events, pl.DataFrame):
        return _polars_apply_min_count_filter(events, min_count, col)
    else:
        raise TypeError(f"expected either `pandas.DataFrame` or `polars.DataFrame`, got {type(events)!r}")


def _pandas_apply_min_count_filter(events: pd.DataFrame, min_count: int, col: str) -> pd.DataFrame:
    return events[events.groupby(col).transform("size") >= min_count]


def _polars_apply_min_count_filter(events: pl.DataFrame, min_count: int, col: str) -> pl.DataFrame:
    return events.filter(pl.len().over(col) >= min_count)


# === APPLY MIN RATING FILTER ===
def apply_min_relevance_filter(
    events: pd.DataFrame | pl.DataFrame,
    min_relevance: float | int,
    relevance_col: str = "rating",
) -> pd.DataFrame | pl.DataFrame:
    """
    Filters the input DataFrame by keeping only the rows where the value in the `relevance_col`
    column is greater than or equal to `min_relevance`.

    Args:
        events (pd.DataFrame | pl.DataFrame): Input DataFrame (either pandas or polars) containing event data.
        min_relevance (float | int): Minimum relevance value required to keep rows.
        relevance_col (str, optional): Column name for relevance values. Defaults to "relevance".
    Returns:
        pd.DataFrame | pl.DataFrame: Filtered DataFrame.
    """
    if isinstance(events, pd.DataFrame):
        return _pandas_apply_min_relevance_filter(events, min_relevance, relevance_col)
    elif isinstance(events, pl.DataFrame):
        return _polars_apply_min_relevance_filter(events, min_relevance, relevance_col)
    else:
        raise TypeError(f"expected either `pandas.DataFrame` or `polars.DataFrame`, got {type(events)!r}")


def _pandas_apply_min_relevance_filter(
    events: pd.DataFrame,
    min_relevance: float | int,
    relevance_col: str = "rating",
) -> pd.DataFrame:

    return events[events[relevance_col] >= min_relevance]


def _polars_apply_min_relevance_filter(
    events: pl.DataFrame,
    min_relevance: float | int,
    relevance_col: str = "rating",
) -> pl.DataFrame:

    return events.filter(pl.col(relevance_col) >= min_relevance)


# === SAMPLE USERS ===
def sample_users(
    events: pl.DataFrame | pd.DataFrame,
    num_users: int,
    seed: int = 42,
    user_col: str = "user_id",
) -> pl.DataFrame | pd.DataFrame:
    """
    Subsamples the input DataFrame to keep only a specified number of users.
    Users are selected randomly without replacement.

    Args:
        events (pd.DataFrame | pl.DataFrame): Input DataFrame (either pandas or polars) containing event data.
        num_users (int): Number of users to keep.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        user_col (str, optional): Column name for user IDs. Defaults to "user_id".

    Returns:
        pd.DataFrame | pl.DataFrame: Subsampled DataFrame containing only the specified number of users.
    """
    if isinstance(events, pd.DataFrame):
        return _pandas_sample_users(events, num_users, seed, user_col)
    elif isinstance(events, pl.DataFrame):
        return _polars_sample_users(events, num_users, seed, user_col)
    else:
        raise TypeError


def _pandas_sample_users(
    events: pd.DataFrame,
    num_users: int,
    seed: int = 42,
    user_col: str = "user_id",
) -> pd.DataFrame:

    users = np.random.default_rng(seed).choice(
        np.unique(events[user_col].values),
        size=num_users,
        replace=False,
    )  # fmt: skip

    return events[events[user_col].isin(users)]


def _polars_sample_users(
    events: pl.DataFrame,
    num_users: int,
    seed: int = 42,
    user_col: str = "user_id",
) -> pl.DataFrame:

    users = np.random.default_rng(seed).choice(
        np.unique(events[user_col].to_numpy()),
        size=num_users,
        replace=False,
    )  # fmt: skip

    return events.filter(pl.col(user_col).is_in(users))

"""Data splits.

The splitters return separate inputs and targets for validation and test.
Global temporal splitting produces three input variants and exact training
frequency statistics. Rare-item thresholds are applied after splitting.

Target selection is always leave-last-out. Input/target separation, cold
filtering and user alignment happen inside the splitters.
"""

from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import pandas as pd


# === target selection / alignment / cold filtering helpers ===


def leave_last(
    holdout_data: pd.DataFrame,
    input_data: pd.DataFrame | None = None,
    user_col: str = "user_id",
    time_col: str = "timestamp",
    eligible_items=None,
    item_col: str = "item_id",
):
    """Leave-last-out split.

    The last eligible interaction per user becomes the target. All original
    preceding events become input. If ``input_data`` is provided, it is prepended to
    the input (combining an existing input sequence with earlier holdout events).
    """
    data_sorted = holdout_data.sort_values([user_col, time_col], kind="stable")
    positions = np.arange(len(data_sorted))
    eligible = np.ones(len(data_sorted), dtype=bool) if eligible_items is None else data_sorted[item_col].isin(eligible_items)
    candidate_positions = pd.Series(positions[eligible], index=data_sorted.loc[eligible, user_col])
    last_positions = candidate_positions.groupby(level=0).max()
    target_positions = data_sorted[user_col].map(last_positions)
    final_input = data_sorted[positions < target_positions]
    targets = data_sorted[positions == target_positions]

    if input_data is not None:
        # keep temporal order (sort by time, not item_id)
        final_input = pd.concat([input_data, final_input], ignore_index=True).sort_values(
            [user_col, time_col], kind="stable"
        )
    return final_input, targets


def align_input_target(
    df_input: pd.DataFrame,
    df_target: pd.DataFrame,
    user_col: str = "user_id",
    keep_targets_with_empty_inputs: bool = False,
):
    """Remove users that lost their input or target from both frames."""
    users_with_targets = df_target[user_col].unique()
    users_with_input = df_input[user_col].unique()
    common_users = np.intersect1d(users_with_targets, users_with_input)

    df_input = df_input[df_input[user_col].isin(common_users)]
    if not keep_targets_with_empty_inputs:
        df_target = df_target[df_target[user_col].isin(common_users)]
    return df_input, df_target


def filter_cold(
    filter_col_name: str,
    base: pd.DataFrame,
    df_input: pd.DataFrame,
    df_target: pd.DataFrame,
    user_col: str = "user_id",
):
    """Drop rows whose ``filter_col_name`` value is absent from ``base``.

    Applied to both input and target so a cold target is removed rather than
    leaving an earlier (in-train) interaction as the new last event.
    """
    warm_entities = base[filter_col_name].unique()
    df_target = df_target[df_target[filter_col_name].isin(warm_entities)]
    df_input = df_input[df_input[filter_col_name].isin(warm_entities)]
    return df_input, df_target


class SplittingStrategy(ABC):
    @abstractmethod
    def split(self, data: pd.DataFrame) -> dict: ...


class LeaveOneOutSplitter(SplittingStrategy):
    """Leave-one-out split."""

    def __init__(
        self,
        user_col: str = "user_id",
        item_col: str = "item_id",
        time_col: str = "timestamp",
        remove_cold_items: bool = False,
        remove_cold_users: bool = False,
    ):
        self.user_col = user_col
        self.item_col = item_col
        self.time_col = time_col
        self.remove_cold_items = remove_cold_items
        self.remove_cold_users = remove_cold_users

    def split(self, data) -> dict:
        data = data.sort_values([self.user_col, self.time_col], kind="stable")
        time_idx_reversed = data.groupby(self.user_col).cumcount(ascending=False)

        # train: all but last two; validation/test holdouts: all but last one / all
        train = data[time_idx_reversed >= 2]
        validation = data[time_idx_reversed >= 1]
        test = data

        val_input, val_target = leave_last(validation, user_col=self.user_col, time_col=self.time_col)
        test_input, test_target = leave_last(test, user_col=self.user_col, time_col=self.time_col)

        val_input, val_target, test_input, test_target = self._filter_cold(
            train, val_input, val_target, test_input, test_target
        )

        val_input, val_target = align_input_target(val_input, val_target, self.user_col)
        test_input, test_target = align_input_target(test_input, test_target, self.user_col)

        # keep only train users with at least two interactions (min-count guarantee)
        train = train[train.groupby(self.user_col)[self.user_col].transform("size") >= 2]

        return {
            "train": train,
            "validation_input": val_input,
            "validation_target": val_target,
            "test_input": test_input,
            "test_target": test_target,
        }

    def _filter_cold(self, train, val_input, val_target, test_input, test_target):
        if self.remove_cold_items:
            val_input, val_target = filter_cold(self.item_col, train, val_input, val_target, self.user_col)
            test_input, test_target = filter_cold(self.item_col, train, test_input, test_target, self.user_col)
        if self.remove_cold_users:
            val_input, val_target = filter_cold(self.user_col, train, val_input, val_target, self.user_col)
            test_input, test_target = filter_cold(self.user_col, train, test_input, test_target, self.user_col)
        return val_input, val_target, test_input, test_target


class GlobalTimeSplitter(SplittingStrategy):
    """Global temporal split.

    Train / test are split by a global time threshold. Training is further split
    into train / validation by a second global time threshold.
    Fixed training suffixes define item membership. Each holdout has shared
    targets and retain, remove-then-crop, and crop-then-remove input variants.
    Target records contain original-history histograms for later rarity analysis.
    """

    def __init__(
        self,
        time_threshold: int | datetime | str | float,
        val_time_threshold: int | datetime | str | float,
        train_max_events: int,
        time_format: str | None = None,
        user_col: str = "user_id",
        item_col: str = "item_id",
        time_col: str = "timestamp",
        remove_unseen_users: bool = False,
        remove_unseen_targets: bool = True,
    ) -> None:

        self.time_threshold = time_threshold
        self.time_format = time_format

        self.val_time_threshold = val_time_threshold

        self.user_col = user_col
        self.item_col = item_col
        self.time_col = time_col

        self.remove_unseen_users = remove_unseen_users
        self.remove_unseen_targets = remove_unseen_targets

        if isinstance(train_max_events, bool) or not isinstance(train_max_events, int) or train_max_events < 2:
            raise ValueError("train_max_events must be an integer of at least two")
        self.train_max_events = train_max_events

    def split(self, data: pd.DataFrame) -> dict:

        if "_source_row" not in data or data["_source_row"].isna().any() or not data["_source_row"].is_unique:
            raise ValueError("Temporal splitting requires unique non-null _source_row identities")
        data = data.sort_values([self.user_col, self.time_col, "_source_row"], kind="stable")

        # 1. Temporal windows retain the inherited training-user eligibility.
        self.resolved_time_threshold = self._resolve_time_threshold(self.time_threshold, data)
        train, test_input, test_holdout = self.split_by_time(data, self.time_threshold)
        self.resolved_val_time_threshold = self._resolve_time_threshold(self.val_time_threshold, train)
        train, val_input, val_holdout = self.split_by_time(train, self.val_time_threshold)

        # 2. Crop training only; evaluation histories remain available.
        period_counts = train.groupby(self.item_col).size()
        train = train.groupby(self.user_col, sort=False).tail(self.train_max_events).copy()

        # 3. Freeze exact training counts; rarity is computed after splitting.
        item_stats = train.groupby(self.item_col).agg(
            training_events=(self.user_col, "size"), training_users=(self.user_col, "nunique"))
        item_stats = item_stats.reindex(pd.Index(data[self.item_col].unique(), name=self.item_col), fill_value=0)
        item_stats["training_period_events"] = period_counts.reindex(item_stats.index, fill_value=0)
        item_counts = item_stats.training_events
        eligible_items = item_counts.index[item_counts > 0] if self.remove_unseen_targets else None
        result = {"train": train, "item_stats": item_stats.reset_index()}
        self.statistics = {
            "time_threshold": self.resolved_time_threshold,
            "val_time_threshold": self.resolved_val_time_threshold,
        }

        for name, earlier, holdout in (("validation", val_input, val_holdout), ("test", test_input, test_holdout)):
            # 4. Select a target and combine the original pre-target events.
            inputs, targets = leave_last(holdout, earlier, self.user_col, self.time_col, eligible_items, self.item_col)
            summary = {"holdout_users": int(holdout[self.user_col].nunique()),
                       "users_without_eligible_target": int(holdout[self.user_col].nunique() - len(targets))}
            before_user_filter = len(targets)
            if self.remove_unseen_users:
                inputs, targets = filter_cold(self.user_col, train, inputs, targets, self.user_col)
            summary["unseen_users_removed"] = before_user_filter - len(targets)

            # 5. Align original histories and save their threshold-independent statistics.
            before_alignment = len(targets)
            inputs, targets = align_input_target(inputs, targets, self.user_col)
            targets = targets.copy()
            summary["users_without_history"] = before_alignment - len(targets)
            targets["target_training_events"] = targets[self.item_col].map(item_counts)
            self._add_input_stats(targets, inputs, item_counts, "history")
            frequencies = inputs[[self.user_col]].assign(training_events=inputs[self.item_col].map(item_counts))
            histogram = frequencies.groupby([self.user_col, "training_events"]).size().reset_index(name="input_events")
            histograms = {user: frame[["training_events", "input_events"]].to_dict("records")
                          for user, frame in histogram.groupby(self.user_col)}
            targets["history_frequency_histogram"] = targets[self.user_col].map(histograms)

            # 6. Build all three variants from the same original history.
            retained = inputs.groupby(self.user_col, sort=False).tail(self.train_max_events - 1)
            seen_inputs = inputs[inputs[self.item_col].map(item_counts) > 0]
            variants = {
                "retain": retained,
                "remove_then_crop": seen_inputs.groupby(self.user_col, sort=False).tail(self.train_max_events - 1),
                "crop_then_remove": retained[retained[self.item_col].map(item_counts) > 0],
            }
            for variant, variant_input in variants.items():
                # 7. Align each variant without dropping shared target records.
                variant_input, _ = align_input_target(variant_input, targets, self.user_col)
                self._add_input_stats(targets, variant_input, item_counts, variant)
                result[f"{name}_input_{variant}"] = variant_input.copy()
            result[f"{name}_target"] = targets
            summary["target_users"] = len(targets)
            self.statistics[name] = summary
        return result

    def _add_input_stats(self, targets, inputs, item_counts, prefix):
        lengths = inputs.groupby(self.user_col).size()
        unseen = inputs[inputs[self.item_col].map(item_counts) == 0].groupby(self.user_col).size()
        events = targets[self.user_col].map(lengths).fillna(0).astype("int64")
        unseen_events = targets[self.user_col].map(unseen).fillna(0).astype("int64")
        targets[f"{prefix}_events"] = events
        targets[f"{prefix}_unseen_events"] = unseen_events
        targets[f"{prefix}_unseen_share"] = unseen_events / events.replace(0, np.nan)
        targets[f"{prefix}_available"] = events > 0
        targets[f"{prefix}_group"] = np.select(
            [events == 0, unseen_events == 0, unseen_events == events],
            [None, "seen_only", "unseen_only"], default="mixed")

    def split_by_time(self, data, time_threshold):

        data = data.sort_values([self.user_col, self.time_col], kind="stable")

        time_threshold = self._resolve_time_threshold(time_threshold, data)

        # we need at least two items in a train sequence for training:
        # keep users whose second interaction is on the train (earlier) side
        user_second_timestamp = (
            data.groupby(self.user_col)[self.time_col]
            .apply(lambda x: x.iloc[1] if len(x) > 1 else None)
            .dropna()
        )
        train_users = user_second_timestamp[user_second_timestamp <= time_threshold].index
        train = data[data[self.user_col].isin(train_users)]
        # train contains all interactions on or before the time threshold
        train = train[train[self.time_col] <= time_threshold]

        # holdout users: last interaction after the time threshold
        user_max_timestamp = data.groupby(self.user_col)[self.time_col].max()
        holdout_users = user_max_timestamp[user_max_timestamp > time_threshold].index
        holdout = data[data[self.user_col].isin(holdout_users)]

        holdout_input = holdout[holdout[self.time_col] <= time_threshold]
        holdout_target = holdout[holdout[self.time_col] > time_threshold]

        return train, holdout_input, holdout_target

    def _resolve_time_threshold(
        self,
        time_threshold: int | datetime | str | float,
        data: pd.DataFrame,
    ) -> int | datetime:

        if isinstance(time_threshold, str):
            if self.time_format is None:
                raise ValueError("string `time_threshold` requires a `time_format`.")
            time_threshold = datetime.strptime(time_threshold, self.time_format)
        elif isinstance(time_threshold, float):
            if not 0 <= time_threshold <= 1:
                raise ValueError("float `time_threshold` must be between 0.0 and 1.0.")

            time_threshold = data[self.time_col].quantile(time_threshold)

        return time_threshold


def last_item_split(df, user_col='user_id', timestamp_col='timestamp'):
    """Split user sequences to input data and ground truth with one last item."""

    user_counts = df[user_col].value_counts()
    df = df[df[user_col].isin(set(user_counts[user_counts > 1].index))]

    df = df.sort_values([user_col, timestamp_col], kind='stable')
    df['time_idx_reversed'] = df.groupby(user_col).cumcount(ascending=False)

    inputs = df[df['time_idx_reversed'] >= 1]
    last_item = df[df['time_idx_reversed'] == 0]

    inputs = inputs.drop(columns=['time_idx_reversed']).reset_index(drop=True)
    last_item = last_item.drop(columns=['time_idx_reversed']).reset_index(drop=True)

    return inputs, last_item

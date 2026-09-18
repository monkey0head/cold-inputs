"""Data splits.

The splitters produce train / validation / test subsets and return them as a
dict. Validation is returned as a single combined sequence (input + target) so
the training pipeline can keep deriving the target with ``last_item_split`` at
eval time, while test is returned pre-separated into ``test_input`` and
``test_target``.

Target selection is always leave-last-out. Input/target separation, cold
filtering and user alignment happen inside the splitters.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd


# === target selection / alignment / cold filtering helpers ===


def leave_last(
    holdout_data: pd.DataFrame,
    input_data: pd.DataFrame | None = None,
    user_col: str = "user_id",
    time_col: str = "timestamp",
):
    """Leave-last-out split.

    The last interaction per user in ``holdout_data`` becomes the target, all
    previous ones become input. If ``input_data`` is provided, it is prepended to
    the input (combining an existing input sequence with earlier holdout events).
    """
    data_sorted = holdout_data.sort_values([user_col, time_col], kind="stable")
    time_idx_reversed = data_sorted.groupby(user_col).cumcount(ascending=False)
    final_input = data_sorted[time_idx_reversed > 0]
    targets = data_sorted[time_idx_reversed == 0]

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

        validation = pd.concat([val_input, val_target], ignore_index=True).sort_values(
            [self.user_col, self.time_col], kind="stable"
        )

        return {
            "train": train,
            "validation": validation,
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
    into train / validation by ``val_type`` ('by_user', 'by_time' or
    'last_train_item'). Target selection is leave-last-out. Cold filtering and
    alignment are applied per input/target before validation is recombined.
    """

    def __init__(
        self,
        time_threshold: int | datetime | str | float,
        time_format: str | None = None,
        val_type: Literal["by_user", "by_time", "last_train_item"] = "by_user",
        val_num_users: int | float | None = None,
        val_time_threshold: int | datetime | str | float | None = None,
        user_col: str = "user_id",
        item_col: str = "item_id",
        time_col: str = "timestamp",
        remove_cold_items: bool = False,
        remove_cold_users: bool = False,
        random_state=42,
    ) -> None:

        self.time_threshold = time_threshold
        self.time_format = time_format

        self.val_type = val_type

        if val_type == "by_user":
            assert val_num_users is not None
        elif val_type == "by_time":
            assert val_time_threshold is not None

        self.val_time_threshold = val_time_threshold
        self.val_num_users = val_num_users

        self.user_col = user_col
        self.item_col = item_col
        self.time_col = time_col

        self.remove_cold_items = remove_cold_items
        self.remove_cold_users = remove_cold_users

        self.random_state = random_state

    def split(self, data: pd.DataFrame) -> dict:

        data = data.sort_values([self.user_col, self.time_col], kind="stable")

        # global split -> train and test (input / holdout)
        train, test_input, test_holdout = self.split_by_time(data, self.time_threshold)

        # validation holdout / target according to strategy
        if self.val_type == "by_time":
            train, val_input, val_holdout = self.split_by_time(train, self.val_time_threshold)
            val_holdout_is_target = False
        elif self.val_type == "by_user":
            train, val_input, val_holdout = self.split_validation_by_user(train)
            val_holdout_is_target = True
        elif self.val_type == "last_train_item":
            train, val_input, val_holdout = self.split_validation_last_train(train)
            val_holdout_is_target = True
        else:
            raise ValueError("Wrong val_type.")

        # cold filtering on input/holdout before target selection
        if self.remove_cold_items:
            val_input, val_holdout = filter_cold(self.item_col, train, val_input, val_holdout, self.user_col)
            test_input, test_holdout = filter_cold(self.item_col, train, test_input, test_holdout, self.user_col)
        if self.remove_cold_users:
            # by_user validation users are drawn from train, so never cold
            if self.val_type != "by_user":
                val_input, val_holdout = filter_cold(self.user_col, train, val_input, val_holdout, self.user_col)
            test_input, test_holdout = filter_cold(self.user_col, train, test_input, test_holdout, self.user_col)

        # target selection = leave-last-out
        if val_holdout_is_target:
            val_target = val_holdout
        else:
            val_input, val_target = leave_last(val_holdout, val_input, self.user_col, self.time_col)
        test_input, test_target = leave_last(test_holdout, test_input, self.user_col, self.time_col)

        val_input, val_target = align_input_target(val_input, val_target, self.user_col)
        test_input, test_target = align_input_target(test_input, test_target, self.user_col)

        # combine validation input + target to preserve training logic
        validation = pd.concat([val_input, val_target], ignore_index=True).sort_values(
            [self.user_col, self.time_col], kind="stable"
        )

        return {
            "train": train,
            "validation": validation,
            "test_input": test_input,
            "test_target": test_target,
        }

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

    def split_validation_by_user(self, train):
        """Random subset of train users held out for validation (target = last)."""

        val_num_users = self.val_num_users
        users = train[self.user_col].unique()

        if isinstance(val_num_users, float):
            assert 0 <= val_num_users <= 1
            val_num_users = int(len(users) * val_num_users)

        validation_users = np.random.default_rng(seed=self.random_state).choice(users, size=val_num_users, replace=False)  # fmt: skip

        validation = train[train[self.user_col].isin(validation_users)]
        train = train[~train[self.user_col].isin(validation_users)]

        val_input, val_target = leave_last(validation, user_col=self.user_col, time_col=self.time_col)

        return train, val_input, val_target

    def split_validation_last_train(self, train):
        """Last training interaction per user as validation target."""

        train = train.sort_values([self.user_col, self.time_col], kind="stable")
        train["time_idx_reversed"] = train.groupby(self.user_col).cumcount(ascending=False)

        # at least two interactions in validation
        validation = train[train.groupby(self.user_col)["time_idx_reversed"].transform("max") > 0].drop(
            columns=["time_idx_reversed"]
        )
        val_input, val_target = leave_last(validation, user_col=self.user_col, time_col=self.time_col)

        train = train[train.time_idx_reversed >= 1]
        # at least two interactions remaining in train
        train = train[train.groupby(self.user_col)["time_idx_reversed"].transform("max") > 1].drop(
            columns=["time_idx_reversed"]
        )

        return train, val_input, val_target


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

"""Semantic IDs based datasets."""

import numpy as np
import torch

from src.datasets.base import LMDataset


class SIDMapperMixin:

    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _index_mapping(self, input_ids: np.ndarray) -> np.ndarray:
        sids = self.semantic_ids[input_ids].reshape(-1)
        return self._to_numpy(sids)


class CausalLMDatasetSIDShift(SIDMapperMixin, LMDataset):

    def __init__(self, df, semantic_ids,
                 max_length=512, shift=2,
                 user_col='user_id', item_col='item_id',
                 time_col='timestamp',
                 random_slice=False,
                 random_state=None):

        super().__init__(df, max_length,
                         user_col=user_col, item_col=item_col, time_col=time_col,
                         random_state=random_state)

        self.semantic_ids = semantic_ids
        self.shift = shift
        self.random_slice = random_slice

        self.num_codebooks = self.semantic_ids.shape[1]

    def __getitem__(self, idx):

        item_sequence = self.data[self.user_ids[idx]]
        item_sequence = self._truncate(item_sequence, self.max_length // self.num_codebooks, self.random_slice)
        item_sequence_sid = self._index_mapping(item_sequence)

        if len(item_sequence_sid) > self.max_length + self.shift:
            item_sequence_sid = item_sequence_sid[-self.max_length - self.shift:]

        if self.shift == 0:
            # [:-0] is [:0] in Python (empty), so handle separately.
            # Use the full sequence; last label is -100 (ignored by cross-entropy).
            input_ids = np.array(item_sequence_sid)
            labels = np.concatenate([item_sequence_sid[1:], np.array([-100])])
        else:
            input_ids = np.array(item_sequence_sid[:-self.shift])
            if self.shift > 1:
                labels = np.array(item_sequence_sid[1:-self.shift + 1])
            else:
                labels = np.array(item_sequence_sid[1:])

        return {"input_ids": input_ids, "labels": labels}


class CausalLMPredictionDatasetSIDShift(SIDMapperMixin, LMDataset):

    def __init__(self, df, semantic_ids,
                 max_length=512, shift=2, validation_mode=True,
                 user_col='user_id', item_col='item_id',
                 time_col='timestamp'):

        super().__init__(df, max_length=max_length,
                         user_col=user_col, item_col=item_col, time_col=time_col)

        self.validation_mode = validation_mode
        self.semantic_ids = semantic_ids
        self.shift = shift

    def __getitem__(self, idx):

        user_id = self.user_ids[idx]
        item_sequence = self.data[user_id]

        if self.validation_mode:

            target = np.array(item_sequence[-1])
            input_ids = np.array(item_sequence[:-1])

            input_ids_sid = self._index_mapping(input_ids)
            if len(input_ids_sid) > self.max_length - self.shift:
                input_ids_sid = input_ids_sid[-self.max_length + self.shift:]

            return {'input_ids': input_ids_sid,
                    'user_id': user_id,
                    "target_item": int(target)}

        else:

            input_ids = np.array(item_sequence)
            input_ids_sid = self._index_mapping(input_ids)

            if len(input_ids_sid) > self.max_length - self.shift:
                input_ids_sid = input_ids_sid[-self.max_length + self.shift:]

            return {'input_ids': input_ids_sid, 'user_id': user_id}

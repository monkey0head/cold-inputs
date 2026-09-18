"""Item-ID datasets for full-softmax next-item training and prediction."""

import numpy as np

from src.datasets.base import LMDataset


class CausalLMDataset(LMDataset):

    def __init__(self, df, max_length=128, shift_labels=True,
                 user_col='user_id', item_col='item_id', time_col='timestamp',
                 random_slice=False, random_state=None):
        super().__init__(df, max_length, user_col=user_col, item_col=item_col,
                         time_col=time_col, random_state=random_state)
        self.shift_labels = shift_labels
        self.random_slice = random_slice

    def __getitem__(self, idx):
        item_sequence = self.data[self.user_ids[idx]]
        item_sequence = self._truncate(item_sequence, self.max_length + 1, self.random_slice)
        input_ids = np.array(item_sequence[:-1])
        labels = np.array(item_sequence[1:]) if self.shift_labels else input_ids
        return {'input_ids': input_ids, 'labels': labels}


class CausalLMPredictionDataset(LMDataset):

    def __init__(self, df, max_length=128, validation_mode=False,
                 user_col='user_id', item_col='item_id', time_col='timestamp'):
        super().__init__(df, max_length=max_length, user_col=user_col,
                         item_col=item_col, time_col=time_col)
        self.validation_mode = validation_mode

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        item_sequence = self.data[user_id]
        if self.validation_mode:
            target = item_sequence[-1]
            window = min(self.max_length, len(item_sequence) - 1)
            input_ids = item_sequence[-window-1:-1]
            labels = np.array(item_sequence[-window:])
            return {'input_ids': input_ids, 'user_id': user_id,
                    'full_history': item_sequence[:-1], 'labels': labels, 'target': target}
        return {'input_ids': item_sequence[-self.max_length:], 'user_id': user_id,
                'full_history': item_sequence}

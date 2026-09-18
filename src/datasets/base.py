"""
Torch base dataset and collate function.
"""

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class LMDataset(Dataset):

    def __init__(self, df, max_length=128,
                 user_col='user_id', item_col='item_id', time_col='timestamp',
                 random_state=None):

        self.max_length = max_length
        self.user_col = user_col
        self.item_col = item_col
        self.time_col = time_col

        self.data = df.sort_values(time_col, kind="stable").groupby(user_col)[item_col].agg(list).to_dict()
        self.user_ids = list(self.data.keys())
        self.rng = np.random.default_rng(random_state)

    def __len__(self):

        return len(self.data)

    def _truncate(self, item_sequence, max_length, random_slice=False):

        if len(item_sequence) <= max_length:
            return item_sequence

        if random_slice:
            max_start = len(item_sequence) - max_length
            start = self.rng.integers(0, max_start + 1)
            return item_sequence[start:start + max_length]
        else:
            return item_sequence[-max_length:]


class PaddingCollateFn:

    def __init__(self, padding_value=0, left_padding=False,
                 labels_padding_value=-100, labels_keys=['labels']):

        self.padding_value = padding_value
        self.left_padding = left_padding
        self.labels_padding_value = labels_padding_value
        self.labels_keys = labels_keys

    def __call__(self, batch):

        collated_batch = {}

        for key in batch[0].keys():

            if np.isscalar(batch[0][key]):
                collated_batch[key] = torch.tensor([example[key] for example in batch])
                continue

            if key in self.labels_keys:
                padding_value = self.labels_padding_value
            else:
                padding_value = self.padding_value
            # left padding is required for sequence generation with huggingface models
            if self.left_padding:
                values = [torch.tensor(example[key][::-1].copy()) for example in batch]
                collated_batch[key] = pad_sequence(values, batch_first=True,
                                                   padding_value=padding_value).flip(-1)
            else:
                values = [torch.tensor(example[key]) for example in batch]
                collated_batch[key] = pad_sequence(values, batch_first=True,
                                                   padding_value=padding_value)
        if 'input_ids' in collated_batch:
            attention_mask = collated_batch['input_ids'] != self.padding_value
            collated_batch['attention_mask'] = attention_mask.to(dtype=torch.float32)

        return collated_batch

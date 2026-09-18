""""
Utils.
"""

import os
import logging

import numpy as np
import torch


def load_sids(filepath: str) -> torch.Tensor:
    _, ext = os.path.splitext(filepath)

    if ext == ".npy":
        sids = torch.from_numpy(np.load(filepath))
    elif ext == ".pt":
        sids = torch.load(filepath)
    else:
        raise ValueError(f"unsupported file extension: {ext!r}")
    
    assert sids.ndim == 2, "sids must be 2D array"

    if sids.shape[0] < sids.shape[1]:
        logging.info("found sids first dim to be longer than the second one. Transposing them")
        sids = sids.T

    return sids.long()

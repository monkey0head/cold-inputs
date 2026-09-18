"""Callable scheduler factories for use with SeqRecBase.configure_optimizers.

Each class is instantiated from a Hydra config and called as
``factory(optimizer, total_steps)`` inside ``configure_optimizers``.
``total_steps`` is computed via ``trainer.estimated_stepping_batches``.
"""

import math

import torch

from torch.optim.lr_scheduler import LambdaLR


class CosineWithWarmup:
    """Linear warmup followed by cosine annealing decay."""

    def __init__(self, warmup_steps: int, min_lr_ratio: float = 0.0):

        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio

    def __call__(self, optimizer, total_steps: int):

        def lr_lambda(current_step: int) -> float:
            if current_step < self.warmup_steps:
                return float(current_step) / max(1, self.warmup_steps)
            progress = float(current_step - self.warmup_steps) / max(1, total_steps - self.warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

        return LambdaLR(optimizer, lr_lambda)


class ReduceOnPlateau:
    """``torch.optim.lr_scheduler.ReduceLROnPlateau`` driven by a validation metric.

    Unlike the step-wise schedules above this one needs no notion of the total run
    length: it cuts the learning rate by ``factor`` whenever ``monitor`` has failed to
    improve for ``patience`` epochs. ``interval`` and ``monitor`` are read by
    ``SeqRecBase.configure_optimizers`` to build the Lightning scheduler config.

    Keep ``patience`` below the EarlyStopping patience, otherwise the run is stopped
    before the schedule ever gets to reduce the rate.
    """

    interval = "epoch"

    def __init__(self, monitor: str = "val_ndcg", mode: str = "max", factor: float = 0.5,
                 patience: int = 5, min_lr_ratio: float = 0.0):

        self.monitor = monitor
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr_ratio = min_lr_ratio

    def __call__(self, optimizer, total_steps: int):

        # The floor is expressed relative to each group's starting rate, so it follows a
        # swept peak learning rate instead of being pinned to one absolute value.
        min_lrs = [group["lr"] * self.min_lr_ratio for group in optimizer.param_groups]

        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=self.mode, factor=self.factor,
            patience=self.patience, min_lr=min_lrs,
        )

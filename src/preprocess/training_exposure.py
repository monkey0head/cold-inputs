"""Training-exposure, capacity, and item-property accounting for the scaling study.

Pure functions (no GPU, deterministic across seeds) that quantify how much of the
catalog and of the SID-token space the GenRec model actually sees during training,
how that relates to model capacity, and which items/targets are cold or colliding.
Wired into ``runs/quantize_and_train.py`` and consumed by the stratified evaluation
in ``runs/train.py``.

Key regime knobs (read from the run config, not assumed):
 - ``window_items`` W = ``max_length // num_codebooks`` = items kept per user history.
 - ``random_slice``: if False the last-W items are used every epoch; if True a uniform
   random W-window is drawn per ``__getitem__`` (see ``LMDataset._truncate``).
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from src.preprocess.data_stats import base_stats


def _to_scalar(value):
    """Cast numpy/torch scalars to plain Python numbers for the tracker."""
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.item()
    return value


def log_subset_stats(data: dict, experiment, catalog_size: int | None = None) -> None:
    """Log core dataset statistics per subset (#1).

    Training and evaluation inputs get the full extended ``base_stats``.
    Targets have one row per user, so only their target-item counts are meaningful.
    """
    for subset in ("train", "validation_input", "test_input"):
        df = data.get(subset)
        if df is None or len(df) == 0:
            continue
        stats = base_stats(df, extended=True, return_df=False)
        experiment.log_scalars({f"data/{subset}/{k}": _to_scalar(v) for k, v in stats.items()})
        table = stats.to_frame(name="value").reset_index(names="stat")
        experiment.log_table(df=table, title=f"data_stats_{subset}", series="subset")

    for subset in ("validation_target", "test_target"):
        target = data.get(subset)
        if target is not None and len(target) > 0:
            experiment.log_scalars({
                f"data/{subset}/n_users": int(target["user_id"].nunique()),
                f"data/{subset}/n_target_items": int(target["item_id"].nunique()),
                f"data/{subset}/n_interactions": int(len(target)),
            })

    if catalog_size is not None:
        experiment.log_scalar("data/catalog_size", int(catalog_size))


@dataclass
class ExposureStats:
    """Container for training-exposure accounting reused by evaluation."""

    item_expected_occ: pd.Series          # item_id -> expected occurrences per epoch
    reachable_items: np.ndarray           # item_ids reachable by any training window
    scalars: dict = field(default_factory=dict)
    tokens_per_epoch: float = 0.0
    total_token_instances: int = 0


def _inclusion_weight(position: np.ndarray, length: np.ndarray, window: int,
                      random_slice: bool) -> np.ndarray:
    """Per-interaction expected inclusion in a training epoch.

    ``position`` is the 0-indexed rank within the user history (by time) and
    ``length`` the full history length. Short histories (``length <= window``) are
    always fully included. For longer ones: last-W deterministically when
    ``random_slice`` is False; the exact per-position window-overlap probability when
    True (per-user mean equals W/length by construction).
    """
    weight = np.zeros(len(position), dtype=np.float64)
    short = length <= window
    weight[short] = 1.0
    long = ~short
    if not long.any():
        return weight

    p = position[long].astype(np.float64)
    a = (length[long] - window).astype(np.float64)  # >= 1
    if not random_slice:
        weight[long] = (p >= a).astype(np.float64)
    else:
        n_windows = a + 1.0
        overlap = np.minimum(p, a) - np.maximum(0.0, p - window + 1.0) + 1.0
        weight[long] = overlap / n_windows
    return weight


def compute_training_exposure(train_df: pd.DataFrame, semantic_ids: torch.Tensor,
                              window_items: int, num_codebooks: int, random_slice: bool,
                              user_col: str = "user_id", item_col: str = "item_id",
                              time_col: str = "timestamp") -> ExposureStats:
    """Item- and token-level training-exposure accounting (#2).

    Returns an ``ExposureStats`` whose ``scalars`` are logged under ``train_exposure/``.
    All quantities respect the truncation window and ``random_slice`` regime.
    """
    window = int(window_items)
    df = train_df[[user_col, item_col, time_col]].sort_values(
        [user_col, time_col], kind="stable"
    )
    items = df[item_col].to_numpy()
    length = df.groupby(user_col)[item_col].transform("size").to_numpy()
    position = df.groupby(user_col).cumcount().to_numpy()

    weight = _inclusion_weight(position, length, window, random_slice)

    item_expected_occ = pd.Series(weight, index=items).groupby(level=0).sum()
    item_expected_occ = item_expected_occ[item_expected_occ > 0]
    reachable_items = item_expected_occ.index.to_numpy()

    total_interactions = int(len(df))
    catalog_size = int(semantic_ids.shape[0]) - 1  # row 0 is the padding SID
    n_train_items = int(df[item_col].nunique())

    item_instances_per_epoch = float(weight.sum())        # == sum_u min(L_u, W)
    tokens_per_epoch = num_codebooks * item_instances_per_epoch
    total_token_instances = num_codebooks * total_interactions

    # Independent self-check of the accounting (see analysis section 4/6):
    #   sum of per-position weights must equal sum over users of min(L_u, W).
    user_len = df.groupby(user_col).size().to_numpy()
    expected_instances = float(np.minimum(user_len, window).sum())
    if not np.isclose(item_instances_per_epoch, expected_instances, rtol=1e-6):
        logging.warning(
            "train_exposure self-check FAILED: sum(weights)=%.3f != sum min(L,W)=%.3f",
            item_instances_per_epoch, expected_instances,
        )
    n_truncated = int((user_len > window).sum())
    n_users = int(len(user_len))

    scalars = {
        # items
        "train_exposure/catalog_size": catalog_size,
        "train_exposure/n_train_items": n_train_items,
        "train_exposure/reachable_item_count": int(len(reachable_items)),
        "train_exposure/reachable_item_share": len(reachable_items) / catalog_size if catalog_size else 0.0,
        "train_exposure/item_instances_per_epoch": item_instances_per_epoch,
        "train_exposure/interactions_used_share": item_instances_per_epoch / total_interactions if total_interactions else 0.0,
        "train_exposure/mean_item_occ_per_epoch": float(item_expected_occ.mean()),
        "train_exposure/median_item_occ_per_epoch": float(item_expected_occ.median()),
        "train_exposure/truncated_user_share": n_truncated / n_users if n_users else 0.0,
        "train_exposure/window_items": window,
        "train_exposure/random_slice": bool(random_slice),
        # tokens (volume view)
        "train_exposure/tokens_per_epoch": tokens_per_epoch,
        "train_exposure/total_token_instances": total_token_instances,
    }

    # Tokens (vocab-id view, per codebook), weighted by expected exposure.
    scalars.update(
        _per_codebook_token_stats(semantic_ids, reachable_items, item_expected_occ.to_numpy(), num_codebooks)
    )

    return ExposureStats(
        item_expected_occ=item_expected_occ,
        reachable_items=reachable_items,
        scalars=scalars,
        tokens_per_epoch=tokens_per_epoch,
        total_token_instances=total_token_instances,
    )


def _entropy(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _per_codebook_token_stats(semantic_ids: torch.Tensor, reachable_items: np.ndarray,
                              reachable_weight: np.ndarray, num_codebooks: int) -> dict:
    """Distinct-used, reuse factor, and exposure-weighted entropy per codebook.

    Normalised entropy uses log of the number of codes the codebook assigns across the
    whole catalog (utilisation-aware). The last codebook (the collision/disambiguation
    token) is additionally mirrored under ``last_cb`` since it is the solver-dependent
    signal of interest.
    """
    sids = semantic_ids.cpu().numpy()
    reach_sids = sids[reachable_items]
    total_mass = float(reachable_weight.sum())

    stats = {}
    for k in range(num_codebooks):
        weighted = pd.Series(reachable_weight, index=reach_sids[:, k]).groupby(level=0).sum()
        distinct_used = int(len(weighted))
        catalog_distinct = int(np.unique(sids[1:, k]).size)  # skip padding row
        entropy = _entropy(weighted.to_numpy())
        entropy_norm = entropy / np.log(catalog_distinct) if catalog_distinct > 1 else 0.0
        reuse = total_mass / distinct_used if distinct_used else 0.0

        for name in (f"cb{k}",) + (("last_cb",) if k == num_codebooks - 1 else ()):
            stats[f"train_exposure/{name}_distinct_used"] = distinct_used
            stats[f"train_exposure/{name}_catalog_distinct"] = catalog_distinct
            stats[f"train_exposure/{name}_used_share"] = distinct_used / catalog_distinct if catalog_distinct else 0.0
            stats[f"train_exposure/{name}_reuse_factor"] = reuse
            stats[f"train_exposure/{name}_entropy"] = entropy
            stats[f"train_exposure/{name}_entropy_norm"] = entropy_norm
    return stats


def log_exposure_stats(experiment, exposure: ExposureStats) -> None:
    """Log #2 scalars and a summary table."""
    experiment.log_scalars({k: _to_scalar(v) for k, v in exposure.scalars.items()})
    table = pd.Series(exposure.scalars).to_frame(name="value").reset_index(names="stat")
    experiment.log_table(df=table, title="train_exposure", series="stats")


def log_capacity_stats(experiment, total_params: int, vocab_size: int, catalog_size: int,
                       exposure: ExposureStats, best_epoch: int | None = None) -> None:
    """Model capacity per item / per token (#3), plus the compute-budget ratio extra.

    Mirrors the columns of analysis section 6. ``distinct_train_tokens`` is the total
    number of token instances the model can be trained on (num_codebooks x train
    interactions), matching the analysis naming.
    """
    tokens_per_epoch = exposure.tokens_per_epoch
    distinct_train_tokens = exposure.total_token_instances

    scalars = {
        "capacity/total_params": int(total_params),
        "capacity/vocab_size": int(vocab_size),
        "capacity/params_per_item": total_params / catalog_size if catalog_size else 0.0,
        "capacity/items_per_vocab_token": catalog_size / vocab_size if vocab_size else 0.0,
        "capacity/params_per_vocab_token": total_params / vocab_size if vocab_size else 0.0,
        "capacity/tokens_per_epoch": tokens_per_epoch,
        "capacity/params_per_epoch_token": total_params / tokens_per_epoch if tokens_per_epoch else 0.0,
        "capacity/distinct_train_tokens": distinct_train_tokens,
        "capacity/params_per_distinct_token": total_params / distinct_train_tokens if distinct_train_tokens else 0.0,
    }

    if best_epoch is not None and best_epoch > 0:
        # best_epoch is 0-indexed; the model has trained through best_epoch + 1 epochs.
        total_tokens_seen = tokens_per_epoch * (best_epoch + 1)
        scalars["capacity/total_tokens_seen"] = total_tokens_seen
        scalars["capacity/params_per_total_token"] = (
            total_params / total_tokens_seen if total_tokens_seen else 0.0
        )

    experiment.log_scalars({k: _to_scalar(v) for k, v in scalars.items()})
    table = pd.Series(scalars).to_frame(name="value").reset_index(names="stat")
    experiment.log_table(df=table, title="capacity", series="stats")


def compute_item_collision_props(semantic_ids: torch.Tensor, num_codebooks: int) -> pd.DataFrame:
    """Per-item collision structure for target stratification (collision-rank extra).

    A collision group shares the first ``num_codebooks - 1`` tokens (the semantic
    prefix); the last token disambiguates. ``is_first`` marks the item with the
    smallest last-token value in its group (rank 0), independent of the codebook
    offset, so it works for every solver. Row 0 (padding) is excluded.
    """
    sids = semantic_ids.cpu().numpy()
    n = sids.shape[0]
    prefix = sids[:, : num_codebooks - 1]
    last = sids[:, num_codebooks - 1]

    df = pd.DataFrame({"item_id": np.arange(n), "last": last})
    df["prefix"] = list(map(tuple, prefix))
    grouped = df.groupby("prefix")
    df["group_size"] = grouped["last"].transform("size")
    df["rank"] = grouped["last"].rank(method="first").astype(int) - 1
    df["is_colliding"] = df["group_size"] > 1
    df["is_first"] = df["rank"] == 0

    df = df[df["item_id"] != 0]  # drop padding row
    return df.set_index("item_id")[["is_colliding", "is_first"]]

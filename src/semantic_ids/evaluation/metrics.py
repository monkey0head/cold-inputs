import torch
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm


def get_collision_rate(sids: torch.Tensor):
    """Calculate the collision rate (fraction of redundant items) up to the second-to-last level.

    Args:
        sids: Tensor of shape (N, L) containing semantic IDs.

    Returns:
        float: The collision rate calculated as 1 - (unique_prefixes / total_items).
    """
    if sids.shape[1] < 2:
        return 0.0

    # Use all levels except the last one
    prefixes = sids[:, :-1]
    num_unique_prefixes = torch.unique(prefixes, dim=0).shape[0]
    num_items = sids.shape[0]

    return 1.0 - (num_unique_prefixes / num_items)

def get_collision_load(sids: torch.Tensor):
    """Calculate the collision load (number of items per used collision code).
    """
    used_codes = sids[:, -1].unique()
    return sids.shape[0] / used_codes.shape[0]

def prefixes_density(sids: torch.Tensor,
    codebook_widths: int | list[int] | None = None,):
    """Calculate the prefixes density, how many prefixes are used per
    available prefix space. Collision-solver codebook is omitted.
    """
    if isinstance(codebook_widths, int):
        codebook_widths = [codebook_widths] * sids.shape[1]
    available_prefixes = np.prod(codebook_widths[:-1])
    unique_prefixes = torch.unique(sids[:, :-1], dim=0).shape[0]
    return unique_prefixes / available_prefixes


def count_unique_prefixes(sids: torch.Tensor):
    """Count unique prefixes at each depth of the semantic ID tree.

    At level ``i`` a "prefix" is the unique combination of the first ``i``
    code values.  The count grows monotonically: level 1 ≤ level 2 ≤ …

    Args:
        sids: Tensor of shape ``(N, L)`` — semantic IDs for N items across
            L codebook levels.  Padding rows (if any) should be excluded
            before calling this function.

    Returns:
        dict[str, int]: mapping ``"level_i"`` → number of distinct i-length
        prefixes present in ``sids``, for i in 1..L.
    """
    level_to_unique_prefixes: dict[str, int] = {}
    for level in range(1, sids.shape[1]):
        cnt_unique_prefixes = torch.unique(sids[:, :level], dim=0).shape[0]
        level_to_unique_prefixes[f"l{level}"] = cnt_unique_prefixes

    return level_to_unique_prefixes


def get_prefix_distribution(sids: torch.Tensor):
    """Count the number of items assigned to each unique prefix at each level."""
    level_to_prefix_counts = {}
    for level in range(1, sids.shape[1]):
        _, counts = torch.unique(sids[:, :level], dim=0, return_counts=True)
        level_to_prefix_counts[f"l{level}"] = counts.cpu().numpy()

    return level_to_prefix_counts


def get_codebook_utilization(
    sids: torch.Tensor,
    codebook_widths: int | list[int] | None = None,
):
    """Calculate codebook utilization at each level of the semantic ID.

    Args:
        sids: Tensor of shape (N, L) containing semantic IDs.
        codebook_widths: Size of the codebook. Can be a single int (if all
            levels have the same width) or a list of ints for each level.
            If None, returns raw counts of used codes instead of fractions.
      
    Returns:
        dict mapping level index (1-based) to utilization.
        Values are floats [0, 1] if codebook_widths is provided, 
        otherwise raw integer counts.
    """
    num_levels = sids.shape[1]
    if isinstance(codebook_widths, int):
        widths = [codebook_widths] * num_levels
    elif codebook_widths is not None:
        widths = list(codebook_widths)
        if len(widths) != num_levels:
            raise ValueError(f"Expected {num_levels} widths, got {len(widths)}")
    else:
        widths = None

    level_to_utilization = {}
    for level in range(1, num_levels + 1):
        used_codes_count = torch.unique(sids[:, level - 1]).shape[0]
        if widths is not None:
            level_to_utilization[f"l{level}"] = used_codes_count / widths[level - 1]
        else:
            level_to_utilization[f"l{level}"] = used_codes_count
  
    return level_to_utilization


def get_codes_distribution(
    sids: torch.Tensor,
    codebook_widths: int | list[int] = 256,
):
    """Compute the usage histogram of codebook entries at each level.

    **Important**: ``sids`` must contain raw (un-shifted) code indices in the
    range ``[0, codebook_widths[i])``.  Do **not** pass tensors produced by
    :func:`prepare_sids`, which shifts and offsets codes; those values would
    index out of the histogram array.

    Args:
        sids: CPU Tensor of shape ``(N, L)`` — raw semantic IDs for N items
            across L codebook levels, with codes in ``[0, codebook_widths[i])``.
        codebook_widths: histogram length(s) per level.  Pass a single ``int``
            to use the same width for every level, or a ``list[int]`` of
            length L to specify a different width per level (e.g. when the
            collision-solver column has a different codebook size).  Each
            value must exceed the maximum code present in the corresponding
            column.

    Returns:
        dict[str, np.ndarray]: mapping ``"level_i"`` → int array of length
        ``codebook_widths[i-1]`` where entry ``j`` is the number of items
        assigned to code ``j`` in the i-th codebook column, for i in 1..L.
    """
    num_levels = sids.shape[1]
    widths: list[int] = (
        [codebook_widths] * num_levels
        if isinstance(codebook_widths, int)
        else list(codebook_widths)
    )
    if len(widths) != num_levels:
        raise ValueError(
            f"codebook_widths has {len(widths)} entries but sids has {num_levels} levels"
        )

    level_to_code_counts: dict[str, np.ndarray] = {}
    for level in range(1, num_levels + 1):
        max_val = max(widths[level - 1], int(sids[:, level - 1].max().item()) + 1)
        counts = torch.bincount(sids[:, level - 1], minlength=max_val)
        level_to_code_counts[f"l{level}"] = counts.cpu().numpy().astype(int)

    return level_to_code_counts


def get_entropy_from_counts(level_to_counts: dict[str, np.ndarray], normalize: bool = False):
    """Calculate Shannon entropy of code or prefix distributions.

    A higher entropy indicates a more uniform distribution (better balance).
    A lower entropy indicates a skewed distribution where a few codes/prefixes dominate.

    Args:
        level_to_counts: mapping from level name to count arrays, 
            e.g., as returned by `get_codes_distribution()` or `get_prefix_distribution()`.
        normalize: if True, divides the entropy by ln(K) where K is the number 
            of bins (the length of the count array). The result will be in [0.0, 1.0].

    Returns:
        dict[str, float]: mapping from level name to Shannon entropy (in nats or normalized).
    """
    level_to_entropy = {}
    for level, counts in level_to_counts.items():
        K = len(counts)

        # Remove zeros to avoid log(0)
        nonzero_counts = counts[counts > 0]
        if len(nonzero_counts) == 0:
            level_to_entropy[level] = 0.0
            continue
 
        p = nonzero_counts / nonzero_counts.sum()
        entropy = -np.sum(p * np.log(p))
        if normalize:
            entropy = entropy / np.log(K)
        level_to_entropy[level] = float(entropy)

    return level_to_entropy


def eval_reconstruction_quality(
    src_embeddings: torch.Tensor,
    rec_embeddings: torch.Tensor,
):
    """Compute mean L2 reconstruction error and explained variance between original and decoded embeddings.

    Args:
        src_embeddings: Tensor of shape ``(N, D)`` — original input embeddings.
        rec_embeddings: Tensor of shape ``(N, D)`` — embeddings reconstructed
            by decoding the quantized codes back to the embedding space.

    Returns:
        tuple[float, float]: mean per-item L2 norm ``||src - rec||_2`` averaged over N items, and explained variance.
    """
    explained_variance = 1 - torch.var(src_embeddings - rec_embeddings, dim=1).mean().item() / torch.var(src_embeddings, dim=1).mean().item()
    reconstruction_error = torch.norm(src_embeddings - rec_embeddings, dim=1).mean().item()

    return reconstruction_error, explained_variance


def get_intra_prefix_distances(
    sids: torch.Tensor,
    embeddings: torch.Tensor,
    max_prefixes_per_level: int | None = 10_000,
    max_items_per_prefix: int | None = 10_000,
):
    """Calculate mean intra-prefix distances at each level of the semantic ID.

    Args:
        sids: semantic IDs
        embeddings: embeddings of the items
        max_prefixes_per_level: each level uses at most this many random prefixes
            (re-sampled per level). None means all prefixes.
        max_items_per_prefix: intra-prefix pairwise distances use at most this many
            random items per prefix. None means all items in the prefix.

    Returns:
        level_to_distances: mean intra-prefix distances at each level of the semantic ID
    """
    level_to_distances: dict[str, list[float]] = {}
    for level in tqdm(range(1, sids.shape[1])):
        _, inverse_indices, counts = torch.unique(
            sids[:, :level], dim=0, return_inverse=True, return_counts=True
        )
        sorted_indices = torch.argsort(inverse_indices)
        sorted_embeddings = embeddings[sorted_indices]
        prefixes_embeddings = torch.split(sorted_embeddings, counts.tolist())
        prefixes_embeddings = _subsample_prefixes(
            prefixes_embeddings, max_prefixes_per_level, embeddings.device
        )

        distances = []
        for prefix_vectors in prefixes_embeddings:
            distances.append(
                _mean_pairwise_l2_within_group(prefix_vectors, max_items_per_prefix)
            )
        level_to_distances[f"l{level}"] = distances

    return level_to_distances


def get_normalized_intra_prefix_distances(
    sids: torch.Tensor,
    embeddings: torch.Tensor,
    max_inter_prefix_samples: int = 10000,
    max_prefixes_per_level: int | None = 10_000,
    max_items_per_prefix: int | None = 10_000,
):
    """
    Calculate normalized intra-prefix distances for each level of the semantic ID.

    Args:
        sids: semantic IDs
        embeddings: embeddings of the items
        max_inter_prefix_samples: maximum number of prefix centroids used to estimate
            inter-prefix distance for normalization (to avoid OOM).
        max_prefixes_per_level: at most this many random prefixes per level; None means all.
        max_items_per_prefix: at most this many random items per prefix for pairwise intra-prefix L2; None means all.

    Returns:
        level_to_normalized_intra_prefix_distances: normalized intra-prefix distances at each level
    """
    level_to_normalized_intra_prefix_distances = {}
    for level in tqdm(range(1, sids.shape[1])):
        _, inverse_indices, counts = torch.unique(
            sids[:, :level], dim=0, return_inverse=True, return_counts=True
        )
        sorted_indices = torch.argsort(inverse_indices)
        sorted_embeddings = embeddings[sorted_indices]
        prefixes_embeddings = torch.split(sorted_embeddings, counts.tolist())
        prefixes_embeddings = _subsample_prefixes(
            prefixes_embeddings, max_prefixes_per_level, embeddings.device
        )

        intra_prefix_distances = []
        prefix_mean_embeddings = []

        for prefix_vectors in prefixes_embeddings:
            prefix_mean_embedding = prefix_vectors.mean(dim=0)
            prefix_mean_embeddings.append(prefix_mean_embedding)

            intra_prefix_distances.append(
                _mean_pairwise_l2_within_group(prefix_vectors, max_items_per_prefix)
            )

        prefix_mean_embeddings = torch.stack(prefix_mean_embeddings, dim=0)

        # Calculate inter-prefix distances with sampling to prevent OOM
        num_prefixes = prefix_mean_embeddings.shape[0]
        if num_prefixes > max_inter_prefix_samples:
            # Randomly sample prefix centroids for inter-prefix normalization.
            indices = torch.randperm(num_prefixes)[:max_inter_prefix_samples]
            sampled_mean_embeddings = prefix_mean_embeddings[indices]
        else:
            sampled_mean_embeddings = prefix_mean_embeddings

        inter_prefix_distances = torch.cdist(sampled_mean_embeddings, sampled_mean_embeddings)
        mask = ~torch.eye(len(inter_prefix_distances), dtype=torch.bool, device=sampled_mean_embeddings.device)
        inter_prefix_distance = inter_prefix_distances[mask].mean().item()

        normalized_intra_prefix_distances = np.array(intra_prefix_distances) / inter_prefix_distance
        level_to_normalized_intra_prefix_distances[f"l{level}"] = normalized_intra_prefix_distances

    return level_to_normalized_intra_prefix_distances


def get_prefix_category_purity(
    sids: torch.Tensor,
    categories: pd.DataFrame,
    min_prefix_size: int = 2,
):
    """Measure how homogeneous ground-truth categories are within each SID prefix.

    For every level 1..L-1 and every category column, items are grouped by their
    SID prefix of that length and the following metrics are computed over the
    distribution of categories inside each prefix:

      - purity:    sum(max class count) / sum(total) — equivalent to
                            the fraction of items whose category equals their
                            prefix's mode (averaged over items).
      - purity_unweighted:  mean over prefixes of (max class count) / (total) —
                            each prefix contributes equally regardless of size.
      - entropy:   Shannon entropy (nats) inside a prefix, averaged
                            over items.
      - gini:      1 - sum(p^2) inside a prefix, averaged over items.
      - coverage:           fraction of items in ``sids`` with a non-missing
                            value in this category column.
      - num_prefixes:       number of prefixes contributing to the aggregates
                            (those with >= min_prefix_size labeled items).

    Items with missing categories are dropped column-by-column. Prefixes whose
    number of *labeled* items is below ``min_prefix_size`` are excluded from
    the aggregates (single-item prefixes are trivially "pure" and would inflate
    the metric otherwise).

    The last SID level is intentionally skipped — it is the collision-solver
    column and adding it gives no meaningful signal.

    Args:
        sids: Tensor of shape (N, L) with semantic IDs.
        categories: DataFrame of shape (N, C) aligned row-by-row with ``sids``.
            Cells may be NaN/None for missing categories. Text and integer
            values are both supported (factorized internally).
        min_prefix_size: minimum number of labeled items in a prefix for it to
            contribute to the aggregates.

    Returns:
        dict mapping ``"l{level}"`` to a dict mapping category column name to
        a dict of metric name -> float.
    """
    num_levels = sids.shape[1]

    factorized = {}
    for col in categories.columns:
        codes, _ = pd.factorize(categories[col], use_na_sentinel=True)
        factorized[col] = codes

    result = {}
    for level in range(1, num_levels):
        _, inverse_indices = torch.unique(
            sids[:, :level], dim=0, return_inverse=True
        )
        prefix_ids = inverse_indices.cpu().numpy()

        level_result = {}
        for col, codes in factorized.items():
            valid_cat_mask = codes != -1
            coverage = float(valid_cat_mask.mean()) if len(codes) else 0.0
            if not valid_cat_mask.any():
                continue

            cross = pd.crosstab(prefix_ids[valid_cat_mask], codes[valid_cat_mask])
            counts = cross.to_numpy()
            totals = counts.sum(axis=1)
            keep = totals >= min_prefix_size
            if not keep.any():
                level_result[col] = {
                    "purity": float("nan"),
                    "purity_unweighted": float("nan"),
                    "entropy": float("nan"),
                    "gini": float("nan"),
                    "coverage": coverage,
                    "num_prefixes": 0,
                }
                continue

            counts = counts[keep]
            totals = totals[keep]
            probs = counts / totals[:, None]

            max_counts = counts.max(axis=1)
            purity_per_prefix = max_counts / totals
            with np.errstate(divide="ignore", invalid="ignore"):
                log_probs = np.where(probs > 0, np.log(probs), 0.0)
            entropy_per_prefix = -(probs * log_probs).sum(axis=1)
            gini_per_prefix = 1.0 - (probs ** 2).sum(axis=1)

            total_items = int(totals.sum())
            level_result[col] = {
                "purity": float(max_counts.sum() / total_items),
                "purity_unweighted": float(purity_per_prefix.mean()),
                "entropy": float((entropy_per_prefix * totals).sum() / total_items),
                "gini": float((gini_per_prefix * totals).sum() / total_items),
                "coverage": coverage,
                "num_prefixes": int(keep.sum()),
            }

        result[f"l{level}"] = level_result

    return result


def get_prefix_clustering_metrics(
    sids: torch.Tensor,
    categories: pd.DataFrame,
):
    """Compare SID prefix partitions against ground-truth category labelings.

    For every level 1..L-1 and every category column, treats the prefix at that
    level as a cluster assignment and the category as ground-truth labels, and
    computes:

      - homogeneity:      each prefix contains only members of one category.
      - completeness:     all members of a category are in the same prefix.
      - v_measure:        harmonic mean of homogeneity and completeness.
      - norm_mutual_info: normalized mutual information (arithmetic mean variant).
      - adj_mutual_info:  adjusted mutual information (chance-corrected NMI).
      - adj_rand_index:   adjusted rand index.
      - fowlkes_mallows:  geometric mean of pairwise precision and recall.

    Items with a missing value in the category column are dropped before the
    comparison. Single-item prefixes are NOT filtered — these metrics evaluate
    the partition globally and benefit from seeing all items.

    The last SID level is skipped for the same reason as in
    :func:`get_prefix_category_purity`.

    Args:
        sids: Tensor of shape (N, L) with semantic IDs.
        categories: DataFrame of shape (N, C) aligned row-by-row with ``sids``.

    Returns:
        dict mapping ``"l{level}"`` to a dict mapping category column name to
        a dict of metric name -> float.
    """
    from sklearn.metrics import (adjusted_mutual_info_score,
                                 adjusted_rand_score, completeness_score,
                                 fowlkes_mallows_score, homogeneity_score,
                                 normalized_mutual_info_score, v_measure_score)

    num_levels = sids.shape[1]

    factorized = {}
    for col in categories.columns:
        codes, _ = pd.factorize(categories[col], use_na_sentinel=True)
        factorized[col] = codes

    result = {}
    for level in range(1, num_levels):
        _, inverse_indices = torch.unique(
            sids[:, :level], dim=0, return_inverse=True
        )
        prefix_ids = inverse_indices.cpu().numpy()

        level_result = {}
        for col, codes in factorized.items():
            valid_mask = codes != -1
            if not valid_mask.any():
                continue
            labels_true = codes[valid_mask]
            labels_pred = prefix_ids[valid_mask]
            level_result[col] = {
                "homogeneity": float(homogeneity_score(labels_true, labels_pred)),
                "completeness": float(completeness_score(labels_true, labels_pred)),
                "v_measure": float(v_measure_score(labels_true, labels_pred)),
                "norm_mutual_info": float(normalized_mutual_info_score(labels_true, labels_pred)),
                "adj_mutual_info": float(adjusted_mutual_info_score(labels_true, labels_pred)),
                "adj_rand_index": float(adjusted_rand_score(labels_true, labels_pred)),
                "fowlkes_mallows": float(fowlkes_mallows_score(labels_true, labels_pred)),
            }
        result[f"l{level}"] = level_result

    return result


def _subsample_prefixes(
    prefixes_embeddings: tuple[torch.Tensor, ...],
    max_prefixes_per_level: int | None,
    device: torch.device,
):
    blocks = list(prefixes_embeddings)
    if max_prefixes_per_level is None or len(blocks) <= max_prefixes_per_level:
        return blocks
    indices = torch.randperm(len(blocks), device=device)[:max_prefixes_per_level]
    return [blocks[i] for i in indices.tolist()]


def _mean_pairwise_l2_within_group(
    prefix_vectors: torch.Tensor,
    max_items_per_prefix: int | None,
):
    num_items_in_prefix = prefix_vectors.shape[0]
    if num_items_in_prefix <= 1:
        return 0.0

    vectors = prefix_vectors
    if max_items_per_prefix is not None and num_items_in_prefix > max_items_per_prefix:
        idx = torch.randperm(num_items_in_prefix, device=prefix_vectors.device)[:max_items_per_prefix]
        vectors = prefix_vectors[idx]
    mask = ~torch.eye(vectors.shape[0], dtype=torch.bool, device=vectors.device)

    return torch.cdist(vectors, vectors, p=2)[mask].mean().item()

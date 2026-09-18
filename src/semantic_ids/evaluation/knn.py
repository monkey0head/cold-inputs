"""KNN-based metrics for evaluating semantic ID quality.

Measures how well the hierarchical prefix structure of semantic IDs preserves
the neighborhood structure of the original item embedding space.
"""

import torch
import torch.nn.functional as F


def evaluate_prefix_knn(
    semantic_ids: torch.Tensor,
    embeddings: torch.Tensor,
    k: int | list[int] = 10,
    num_query_samples=10000,
    batch_size: int = 1024,
    seed: int = 0,
    device=None,
    metric: str = "cosine",
):
    """Compute prefix-preservation Recall@k and Precision@k per SID level.

    For each prefix level l (1..num_levels), items sharing the same prefix are grouped
    into buckets. Recall@k measures what fraction of the true KNN neighbors fall into
    the same bucket as the query. Precision@k measures what fraction of bucket
    members appear among the KNN neighbors.

    Args:
        semantic_ids: (N, m) integer tensor of semantic IDs.
        embeddings: (N, D) float tensor of item embeddings.
        k: number of nearest neighbors for ground truth. Can be a list of ints.
        num_query_samples: number of query items to sample. None means all N items.
        batch_size: batch size for cosine similarity computation.
        seed: random seed for query sampling reproducibility.
        device: device to run on. Defaults to embeddings.device.
        metric: distance metric for ground-truth KNN search. "cosine" or "l2".

    Returns:
        Dict with keys:
            "recall@k":       dict mapping "l1".."l{m-1}" to float recall values.
            "precision@k":    dict mapping "l1".."l{m-1}" to float precision values
                              (None for a level where no query has bucket_size > 1).
            "f1_score@k":     dict mapping "l1".."l{m-1}" to F1 score (harmonic mean
                              of recall and precision; None when precision is None).
            "jaccard@k":      dict mapping "l1".."l{m-1}" to Jaccard similarity between
                              top-k neighbors and bucket members
                              (hits / (k + bucket_size - 1 - hits)).
            "overlap@k":      dict mapping "l1".."l{m-1}" to overlap coefficient
                              (hits / min(k, bucket_size - 1); None when no query has
                              bucket_size > 1).
            "avg_bucket_size": dict mapping "l1".."l{m-1}" to average bucket size
                              for query items at that prefix level.
            "avg_lcp@k":      float — average longest common prefix length between
                              each query and its top-k true neighbors
                              (over all m levels of the SID).
    """
    assert semantic_ids.shape[0] == embeddings.shape[0], (
        f"shape mismatch: {semantic_ids.shape[0]} != {embeddings.shape[0]}"
    )
    if device is None:
        device = embeddings.device
    else:
        embeddings = embeddings.to(device)
    semantic_ids = semantic_ids.to(device)
    num_items, num_levels = semantic_ids.shape

    ks = [k] if isinstance(k, int) else sorted(list(k))
    max_k = max(ks)

    # Uniform random sampling of query items
    if num_query_samples is None or num_query_samples >= num_items:
        query_ids = torch.arange(num_items, device=device)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
        query_ids = torch.randperm(num_items, generator=generator, device=device)[:num_query_samples]

    # top-k neighbors for each query across the full corpus
    neighbors = find_topk_neighbors(embeddings, query_ids, max_k, batch_size, metric=metric)

    metrics_per_k = {k_val: {"recalls": {}, "precisions": {}, "f1_scores": {}, "jaccards": {}, "overlaps": {}} for k_val in ks}
    avg_bucket_sizes = {}

    for level in range(1, num_levels):
        item_to_bucket, bucket_sizes = build_prefix_buckets(semantic_ids, level)

        query_buckets = item_to_bucket[query_ids]
        neighbor_buckets_max = item_to_bucket[neighbors]

        query_bucket_size = bucket_sizes[query_buckets]
        avg_bucket_sizes[f"l{level}"] = query_bucket_size.float().mean().item()

        num_candidates = query_bucket_size - 1
        has_candidates = query_bucket_size > 1

        for k_val in ks:
            neighbor_buckets = neighbor_buckets_max[:, :k_val]
            hits = (query_buckets.unsqueeze(1) == neighbor_buckets).sum(dim=1)
            recall = (hits.float() / k_val).mean().item()
            metrics_per_k[k_val]["recalls"][f"l{level}"] = recall

            jaccard_union = (k_val + num_candidates - hits).clamp(min=1)
            metrics_per_k[k_val]["jaccards"][f"l{level}"] = (hits.float() / jaccard_union.float()).mean().item()

            if has_candidates.any():
                precision = (
                    hits[has_candidates].float() / num_candidates[has_candidates].float()
                ).mean().item()
                metrics_per_k[k_val]["precisions"][f"l{level}"] = precision
                denom = precision + recall
                metrics_per_k[k_val]["f1_scores"][f"l{level}"] = 2 * precision * recall / denom if denom > 0 else 0.0

                # Overlap coefficient: hits / min(k, bucket_size - 1)
                min_size = num_candidates[has_candidates].clamp(max=k_val)
                metrics_per_k[k_val]["overlaps"][f"l{level}"] = (
                    hits[has_candidates].float() / min_size.float()
                ).mean().item()
            else:
                metrics_per_k[k_val]["precisions"][f"l{level}"] = None
                metrics_per_k[k_val]["f1_scores"][f"l{level}"] = None
                metrics_per_k[k_val]["overlaps"][f"l{level}"] = None

    results = {"avg_bucket_size": avg_bucket_sizes}

    for k_val in ks:
        # Average longest common prefix between each query and its top-k true neighbors.
        query_sids = semantic_ids[query_ids].unsqueeze(1)         # (Q, 1, m)
        neighbor_sids = semantic_ids[neighbors[:, :k_val]]        # (Q, k, m)
        lcp_per_pair = (query_sids == neighbor_sids).long().cumprod(dim=-1).sum(dim=-1)  # (Q, k)
        avg_lcp = lcp_per_pair.float().mean().item()

        results[f"recall@{k_val}"] = metrics_per_k[k_val]["recalls"]
        results[f"precision@{k_val}"] = metrics_per_k[k_val]["precisions"]
        results[f"f1_score@{k_val}"] = metrics_per_k[k_val]["f1_scores"]
        results[f"jaccard@{k_val}"] = metrics_per_k[k_val]["jaccards"]
        results[f"overlap@{k_val}"] = metrics_per_k[k_val]["overlaps"]
        results[f"avg_lcp@{k_val}"] = avg_lcp

    return results


def find_topk_neighbors(
    embeddings: torch.Tensor,
    query_ids: torch.Tensor,
    k: int,
    batch_size: int = 1024,
    device=None,
    metric: str = "cosine",
):
    """Find top-k nearest neighbors for query items in the corpus.

    Args:
        embeddings: (N, D) float tensor of corpus embeddings.
        query_ids: (Q,) tensor of query item indices within the corpus.
        k: number of nearest neighbors to retrieve.
        batch_size: number of query items processed per batch to control memory usage.
        device: device to run on. Defaults to embeddings.device.
        metric: distance metric. "cosine" (max similarity) or "l2" (min Euclidean distance).

    Returns:
        (Q, k) long tensor of neighbor indices in the corpus. Self is excluded.
    """
    if metric not in ("cosine", "l2"):
        raise ValueError(f"Unknown metric: {metric!r}. Expected 'cosine' or 'l2'.")

    if device is None:
        device = embeddings.device
    else:
        embeddings = embeddings.to(device)

    if metric == "cosine":
        embeddings = F.normalize(embeddings, dim=1)
    query_emb = embeddings[query_ids]  # (Q, D)
    num_query_items = query_ids.shape[0]

    neighbors = torch.empty((num_query_items, k), dtype=torch.long, device=device)

    for start in range(0, num_query_items, batch_size):
        end = min(start + batch_size, num_query_items)
        if metric == "cosine":
            # higher = closer
            scores = query_emb[start:end] @ embeddings.T          # (B, N)
        else:
            # negative L2 distance: higher = closer
            scores = -torch.cdist(query_emb[start:end], embeddings, p=2)
        # Mask self
        scores.scatter_(
            dim=1,
            index=query_ids[start:end].unsqueeze(1),
            value=float("-inf"),
        )
        _, top_idx = scores.topk(k, dim=1)
        neighbors[start:end] = top_idx

    return neighbors


def build_prefix_buckets(semantic_ids: torch.Tensor, level: int):
    """Group items by their SID prefix and return a bucket ID for each item.

    Args:
        semantic_ids: (N, m) integer tensor of semantic IDs.
        level: prefix length (1..m).

    Returns:
        item_to_bucket: (N,) tensor of bucket IDs, one per item.
        bucket_sizes: (B,) tensor of bucket sizes, where B is the number of unique prefixes.
    """

    prefixes = semantic_ids[:, :level].contiguous()
    _, item_to_bucket = torch.unique(prefixes, dim=0, return_inverse=True)
    bucket_sizes = torch.bincount(item_to_bucket)

    return item_to_bucket, bucket_sizes

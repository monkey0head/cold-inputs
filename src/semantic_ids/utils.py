"""Helpers for preparing semantic-ID tokens for GPT-SID.

Collision solving lives in ``src.semantic_ids.collision_solvers``
(``get_collision_solver(name, ...)``).
"""
import torch


def prepare_sids(sids_tensor, num_embeddings_per_codebook=None, shift=1):
    """Prepare raw semantic IDs for GPT-SID training.

    Steps:
    1. Shift every code by ``shift`` to reserve 0 as the padding token id.
    2. Offset each codebook so token id ranges do not overlap between codebooks,
       making every token id globally unique.
    3. Prepend a padding item (row 0) whose token ids are all 0.

    Args:
        sids_tensor: tensor of shape ``(num_items, num_codebooks)`` holding the
            raw semantic IDs for each item across the codebooks.
        num_embeddings_per_codebook: fixed number of embeddings per codebook. When
            ``None``, it is auto-computed per codebook to fit the data.
        shift: amount to shift the codes by so that 0 stays free as the padding token.

    Returns:
        tensor of shape ``(num_items + 1, num_codebooks)`` (the extra leading row is
        the padding item).
    """
    num_codebooks = sids_tensor.shape[1]
    codebook_index = torch.arange(num_codebooks)
    if num_embeddings_per_codebook is None:
        base, _ = sids_tensor.max(axis=0)
        base = (base + 1).roll(1).reshape(1, -1)
        base[:, 0] = 0
        num_embeddings_per_codebook = base.max()
        base = torch.cumsum(base, axis=-1)
    else:
        base = (codebook_index * num_embeddings_per_codebook).reshape(1, -1)
    final_sids = sids_tensor + shift + base.repeat(sids_tensor.shape[0], 1)
    assert final_sids.min() > 0
    # exclude collisions codebook from the check
    assert final_sids[:, :-1].max() < num_embeddings_per_codebook * (num_codebooks -1) + shift
    final_sids = torch.vstack([torch.zeros(1, num_codebooks, dtype=final_sids.dtype), final_sids])
    return final_sids

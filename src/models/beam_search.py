"""Beam-search mixin for autoregressive recommender models.

Provides a reusable ``generate_beam`` implementation.
Models mix in ``BeamSearchMixin`` and implement the abstract methods
to wire up their specific logic.
"""

import torch

from src.models.token_trie import TokenTrie


class BeamSearchMixin:
    """Beam search mixin.

    Mix into any ``nn.Module`` that implements the following methods:

    * :meth:`_forward_with_kv_cache` – token ids → (hidden states, kv-cache)
    * :meth:`_compute_logits`  – hidden states → logits over vocabulary

    Optionally override:

    * :meth:`_expand_kv_cache`  – replicate cache for beam expansion
    * :meth:`_reorder_kv_cache` – reindex cache after beam pruning
    """

    # ------------------------------------------------------------------
    # Abstract methods – must be implemented by every concrete model
    # ------------------------------------------------------------------

    def _forward_with_kv_cache(self, input_ids, attention_mask,
                               past_key_values=None, position_ids=None,
                               step=None, **kwargs):
        """Run forward pass and return hidden states with KV-cache.

        Args:
            input_ids:      ``(batch_size, seq_length)`` input ids.
            attention_mask: ``(batch_size, seq_length)`` attention mask.
            past_key_values: KV-cache object (``None`` on prefill).
            position_ids:   ``(batch_size, seq_length)`` position ids, or ``None``.
            step:           ``None`` on the initial prefill pass;
                            ``int`` (starting from 1) on each generation step.
            **kwargs:       extra arguments forwarded from ``generate_beam`` (e.g. ``codebook_idx``).

        Returns:
            ``(last_hidden_state, past_key_values)`` tuple, where:
            - last_hidden_state: ``(batch_size, seq_length, hidden_size)`` last hidden state.
            - past_key_values: KV-cache object.
        """

        raise NotImplementedError

    def _compute_logits(self, hidden):
        """Project hidden states to vocabulary logits.

        Args:
            hidden: ``(batch_size, hidden_size)`` last-position hidden states.

        Returns:
            ``(batch_size, vocab_size)`` logits over the vocabulary.
        """

        raise NotImplementedError

    # ------------------------------------------------------------------
    # Hooks – override in subclasses as needed
    # ------------------------------------------------------------------

    def _expand_kv_cache(self, past_key_values, beam_size):
        """Repeat KV-cache tensors along the batch dim for beam expansion."""

        return tuple(
            tuple(tensor.repeat_interleave(beam_size, dim=0) for tensor in layer)
            for layer in past_key_values
        )

    def _reorder_kv_cache(self, past_key_values, beam_idx):
        """Re-index KV-cache tensors to match reordered beams."""

        return tuple(
            tuple(tensor[beam_idx] for tensor in layer)
            for layer in past_key_values
        )

    # ------------------------------------------------------------------
    # Beam search method
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_beam(self, input_ids, attention_mask, num_steps=4,
                      beam_size=10, return_topk=True, padding_side="right",
                      token_trie: TokenTrie | None = None,
                      **kwargs):
        """Autoregressively generate next tokens using beam search.

        Uses KV-cache for efficient step-by-step decoding.

        Args:
            input_ids:      ``(batch_size, seq_length)`` context token ids.
            attention_mask: ``(batch_size, seq_length)`` mask (1 = attend, 0 = ignore).
            num_steps:      how many tokens to generate.
            beam_size:      beam width ``K``.
            return_topk:    if ``True`` return all ``K`` candidates per sample,
                            otherwise return only the best one.
            padding_side:   ``"right"`` or ``"left"``.
            token_trie:     trie with valid token continuations for constrained decoding.
            **kwargs:       forwarded to :meth:`_forward_with_kv_cache`.

        Returns:
            generated: ``(batch_size, beam_size, num_steps)`` if ``return_topk`` else
                       ``(batch_size, num_steps)`` tensor of generated tokens.
            scores:    ``(batch_size, beam_size)`` if ``return_topk`` else ``(batch_size,)``
                        tensor of cumulative log-probabilities.
        """

        if padding_side not in ("left", "right"):
            raise ValueError(f"padding_side must be 'left' or 'right', got '{padding_side}'")

        batch_size, seq_length = input_ids.shape
        device = input_ids.device

        # ---- prefill: run full context through the model ----

        last_hidden_state, past_key_values = self._forward_with_kv_cache(
            input_ids, attention_mask, step=None, **kwargs)

        # extract hidden state at the last real token (depends on padding side)
        if padding_side == "left":
            last_hidden = last_hidden_state[:, -1, :]
        else:
            last_pos = attention_mask.sum(dim=1, dtype=torch.long) - 1
            last_hidden = last_hidden_state[
                torch.arange(batch_size, device=device), last_pos]

        # initialize beams with top-K tokens from the first prediction
        logits = self._compute_logits(last_hidden)
        log_probs = torch.log_softmax(logits, dim=-1)

        if token_trie is not None:
            _apply_token_trie_mask(log_probs, token_trie)
        
        beam_scores, first_tokens = torch.topk(log_probs, beam_size, dim=-1)
        generated = first_tokens.unsqueeze(-1)  # (batch_size, beam_size, 1)

        if num_steps == 1:
            if return_topk:
                return generated, beam_scores
            return generated[:, 0, :], beam_scores[:, 0]

        # replicate KV-cache and mask for each beam
        past_key_values = self._expand_kv_cache(past_key_values, beam_size)
        attention_mask = attention_mask.repeat_interleave(beam_size, dim=0)
        if padding_side == "right":
            next_pos = (last_pos + 1).repeat_interleave(beam_size, dim=0)

        # ---- autoregressive steps ----

        for step in range(1, num_steps):
            last_token = generated[:, :, -1].reshape(batch_size * beam_size, 1)

            # extend mask for the newly generated token
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(batch_size * beam_size, 1, device=device, dtype=attention_mask.dtype),
            ], dim=1)

            # right-padded sequences need explicit position ids
            position_ids = None
            if padding_side == "right":
                position_ids = (next_pos + step - 1).unsqueeze(1)

            last_hidden_state, past_key_values = self._forward_with_kv_cache(
                last_token, attention_mask, past_key_values=past_key_values,
                position_ids=position_ids, step=step, **kwargs)

            logits = self._compute_logits(last_hidden_state[:, -1, :])
            log_probs = torch.log_softmax(logits, dim=-1).view(batch_size, beam_size, -1)

            if token_trie is not None:
                _apply_token_trie_mask(log_probs, token_trie, generated)
            
            vocab_size = log_probs.shape[-1]

            scores = beam_scores.unsqueeze(-1) + log_probs
            scores_flat = scores.view(batch_size, -1)

            topk_scores, topk_flat_idx = torch.topk(scores_flat, beam_size, dim=-1)
            beam_idx = topk_flat_idx // vocab_size
            token_idx = topk_flat_idx % vocab_size
            beam_scores = topk_scores

            # reorder generated sequences to match selected beams, append new tokens
            generated = torch.gather(
                generated, 1,
                beam_idx.unsqueeze(-1).expand(-1, -1, generated.shape[-1])
            )
            generated = torch.cat([generated, token_idx.unsqueeze(-1)], dim=-1)

            # reorder KV-cache and mask to match selected beams
            flat_idx = (torch.arange(batch_size, device=device).unsqueeze(1) * beam_size + beam_idx).view(-1)
            past_key_values = self._reorder_kv_cache(past_key_values, flat_idx)
            attention_mask = attention_mask[flat_idx]

        if return_topk:
            return generated, beam_scores

        return generated[:, 0, :], beam_scores[:, 0]


def _mask_except_allowed_tokens(scores: torch.Tensor, allowed_tokens: list[int]) -> None:
    """Keep only allowed token scores in-place and mask the rest with ``-inf``."""

    if not allowed_tokens:
        scores.fill_(-torch.inf)
        return

    allowed_idx = torch.as_tensor(allowed_tokens, device=scores.device, dtype=torch.long)
    allowed_scores = scores[allowed_idx].clone()

    scores.fill_(-torch.inf)
    scores[allowed_idx] = allowed_scores


def _apply_token_trie_mask(log_probs: torch.Tensor, token_trie: TokenTrie,
                           prefixes: torch.Tensor | None = None) -> None:
    """Mask log-probabilities in-place so only trie-consistent continuations remain."""

    if prefixes is None:
        allowed_root_tokens = token_trie.get_allowed_next_tokens(())
        flat_log_probs = log_probs.reshape(-1, log_probs.shape[-1])

        for row in flat_log_probs:
            _mask_except_allowed_tokens(row, allowed_root_tokens)

        return

    flat_log_probs = log_probs.reshape(-1, log_probs.shape[-1])
    flat_prefixes = prefixes.reshape(-1, prefixes.shape[-1])

    for prefix, row in zip(flat_prefixes, flat_log_probs):
        allowed_token_ids = token_trie.get_allowed_next_tokens(prefix.tolist())
        _mask_except_allowed_tokens(row, allowed_token_ids)

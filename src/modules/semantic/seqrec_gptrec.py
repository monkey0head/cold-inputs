"""Semantic IDs based lightning module for GPTRec."""

import torch
import torch.nn.functional as F

from src.modules.base import SeqRecBase
from src.models.token_trie import TokenTrie


class SeqRecSIDGPTRec(SeqRecBase):

    def __init__(self, model, semantic_ids, lr=1e-3,
                 padding_idx=0, num_codebooks=4, beam_size=10,
                 constrained_inference: bool = False,
                 extended_val_metrics: bool = True,
                 codebook_sizes: list[int] | None = None,
                 optimizer=None, lr_scheduler=None):

        if padding_idx != 0:
            raise ValueError("padding_idx must be 0 in current implementation, SIDs must start from 1")

        super().__init__(model, lr, padding_idx, optimizer=optimizer, lr_scheduler=lr_scheduler)

        self.num_codebooks = num_codebooks
        self.beam_size = beam_size
        self.semantic_ids = semantic_ids
        self.sid_tuple_to_item = {
            tuple(row.tolist()): i
            for i, row in enumerate(self.semantic_ids.detach().cpu())
        } # {tuple(SIDs): item_id}

        self.constrained_inference = constrained_inference
        self.extended_val_metrics = extended_val_metrics
        self.token_trie = self._build_token_trie(self.semantic_ids) if constrained_inference else None

        self.codebook_sizes = codebook_sizes
        if codebook_sizes is not None:
            # suggests that padding index is 0
            boundaries = torch.tensor([0] + list(codebook_sizes), dtype=torch.long).cumsum(0)
            self.register_buffer("_cb_boundaries", boundaries, persistent=False)

    def training_step(self, batch, batch_idx):

        codebook_idx = batch.get('codebook_idx', None)
        if codebook_idx is not None:
            logits = self.model(batch['input_ids'], batch['attention_mask'], codebook_idx)
        else:
            logits = self.model(batch['input_ids'], batch['attention_mask'])

        targets = batch['labels']
        loss = self.compute_loss(logits, targets)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def compute_loss(self, logits, targets):

        total_vocab_size = logits.shape[-1]
        loss = F.cross_entropy(
            logits.reshape(-1, total_vocab_size),
            targets.reshape(-1),
            ignore_index=-100,
        )

        return loss

    def _compute_val_losses(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        logits = self.model(input_ids, attention_mask)

        # Shift input_ids left by 1 to form causal LM labels; mask padding and last position.
        val_labels = input_ids.clone()
        val_labels[:, :-1] = input_ids[:, 1:]
        val_labels[:, -1] = -100
        val_labels[attention_mask == 0] = -100

        cb_assignments = self._label_codebook_assignments(val_labels)
        return self._per_codebook_val_loss(logits, val_labels, cb_assignments)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        generate_kwargs = dict(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            beam_size=self.beam_size,
            num_steps=self.num_codebooks,
            token_trie=self.token_trie,
        )
        codebook_idx = batch.get('codebook_idx', None)
        if codebook_idx is not None:
            generate_kwargs['codebook_idx'] = codebook_idx
        beams, beam_scores = self.model.generate_beam(**generate_kwargs) # [B, beam_size, H]

        preds = self._beam_to_items(beams)
        targets = batch["target_item"].detach().cpu()
        item_metrics = self.compute_val_metrics(targets, preds)

        for name, value in item_metrics.items():
            self.log(f"val_{name}", value, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_valid_items_share", (preds != -1).float().sum() / preds.numel(), on_step=False, on_epoch=True, prog_bar=True)

        # Hierarchical names: val_sid/{metric}/{series} — one ClearML chart per metric
        if self.extended_val_metrics:
            sid_metrics = self._compute_sid_metrics(beams, targets)
            for key, value in sid_metrics.items():
                self.log(f"val_sid/{key}", value, on_step=False, on_epoch=True, prog_bar=False)

            if self.codebook_sizes is not None:
                per_cb_losses = self._compute_val_losses(batch)
                for key, value in per_cb_losses.items():
                    self.log(f"val_loss/{key}", value, on_step=False, on_epoch=True, prog_bar=False)

        return item_metrics

    def make_prediction(self, batch):

        generate_kwargs = dict(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            beam_size=self.beam_size,
            num_steps=self.num_codebooks,
            token_trie=self.token_trie,
        )
        codebook_idx = batch.get('codebook_idx', None)
        if codebook_idx is not None:
            generate_kwargs['codebook_idx'] = codebook_idx
        beams, beam_scores = self.model.generate_beam(**generate_kwargs) # [B, beam_size, H]

        preds = self._beam_to_items(beams) # [B, beam_size]
        # scores_row: [1, 1/2, 1/3, ...] - placeholder rank-based scores (not model probabilities)
        scores_row = 1 / (torch.arange(self.beam_size, dtype=torch.float32, device=beams.device) + 1.)
        scores = scores_row.repeat(beams.shape[0], 1)

        return preds, scores

    def _label_codebook_assignments(self, labels: torch.Tensor) -> torch.Tensor:
        """Return 0-indexed codebook assignment for each label token. Shape [B, L].

        Uses ``codebook_sizes`` boundaries (label token id → codebook). Positions where
        ``labels == -100`` are don't-care (values still produced for shape consistency).
        """
        if self.codebook_sizes is not None:
            # bucketize maps token v in (boundaries[i-1], boundaries[i]] → bucket i.
            # Subtract 1 for 0-indexed codebook; clamp handles padding (label=-100→0).
            return (torch.bucketize(labels.clamp(min=0), self._cb_boundaries, right=False) - 1).clamp(min=0)
        else:
            raise ValueError("codebook_sizes must be provided for per-codebook loss computation")


    def _per_codebook_val_loss(
        self,
        logits: torch.Tensor,           # [batch, seq, vocab]
        labels: torch.Tensor,           # [batch, seq], -100 for ignored positions
        cb_assignments: torch.Tensor,   # [batch, seq], 0-indexed
    ) -> dict[str, float]:
        """Mean CE per codebook: one CE per position, then mean within each codebook.

        ``cross_entropy(..., reduction='none')`` runs once over all positions; ignored
        labels contribute 0 via ``ignore_index``. Each codebook mean uses only valid
        positions assigned to that codebook.

        Returns keys ``"loss/cb{k}"`` for each codebook and ``"loss/cb_last"`` for the last.
        Codebooks with no valid positions are omitted.
        """
        vocab_size = logits.shape[-1]
        flat_logits = logits.reshape(-1, vocab_size)
        flat_labels = labels.reshape(-1)
        flat_codebook = cb_assignments.reshape(-1)
        valid_mask = flat_labels != -100

        per_position_loss = F.cross_entropy(
            flat_logits,
            flat_labels,
            ignore_index=-100,
            reduction="none",
        )

        result: dict[str, float] = {}
        result["total"] = per_position_loss[valid_mask].mean().item()
        for codebook_index in range(self.num_codebooks):
            codebook_mask = valid_mask & (flat_codebook == codebook_index)
            if codebook_mask.any():
                mean_loss = per_position_loss[codebook_mask].mean()
                result[f"cb{codebook_index}"] = mean_loss.item()
                if codebook_index == self.num_codebooks - 1:
                    result["cb_last"] = mean_loss.item()
        return result

    def _compute_sid_metrics(self, beams: torch.Tensor, target_items: torch.Tensor) -> dict[str, float]:
        """Per-codebook and prefix hit/MRR/NDCG.

        Returns keys like ``hr/cb0``, ``mrr/prefix2`` — ready for ``val_sid/`` logging
        so each metric type becomes one ClearML chart with series per codebook/prefix.
        """
        beams_cpu = beams.detach().cpu()
        _, _, num_codebooks = beams_cpu.shape
        target_sids = self.semantic_ids[target_items.long()].detach().cpu()
        pos_match = beams_cpu == target_sids.unsqueeze(1).expand_as(beams_cpu)

        metrics: dict[str, float] = {}

        for codebook_idx in range(num_codebooks):
            for name, value in self._beam_match_metrics(pos_match[:, :, codebook_idx]).items():
                metrics[f"{name}/cb{codebook_idx}"] = value
                if codebook_idx == num_codebooks - 1:
                    metrics[f"{name}/cb_last"] = value

        for prefix_len in range(2, num_codebooks + 1):
            for name, value in self._beam_match_metrics(pos_match[:, :, :prefix_len].all(dim=2)).items():
                metrics[f"{name}/prefix{prefix_len}"] = value
                if prefix_len == num_codebooks:
                    metrics[f"{name}/prefix_last"] = value

        return metrics

    @staticmethod
    def _first_true_rank(match: torch.BoolTensor) -> torch.LongTensor:
        """Return 1-based rank of first True per row, or 0 if no match."""
        K = match.shape[1]
        ranks = torch.arange(1, K + 1, device=match.device).unsqueeze(0).expand_as(match)
        masked = ranks.where(match, torch.full_like(ranks, K + 1))
        min_rank = masked.min(dim=1).values
        min_rank[min_rank == K + 1] = 0
        return min_rank

    @classmethod
    def _beam_match_metrics(cls, match: torch.BoolTensor) -> dict[str, float]:
        """Compute HR, MRR, NDCG from a [B, K] boolean match matrix."""
        any_hit = match.any(dim=1)
        rank = cls._first_true_rank(match).float().clamp(min=1)
        zero = torch.zeros(1)
        return {
            "hr": any_hit.float().mean().item(),
            "mrr": (1.0 / rank).where(any_hit, zero).mean().item(),
            "ndcg": (1.0 / torch.log2(rank + 1)).where(any_hit, zero).mean().item(),
        }

    def _beam_to_items(self, beams):

        if self.sid_tuple_to_item is not None:
            B, K, _ = beams.shape
            out = torch.full((B, K), -1, dtype=torch.long)
            x = beams.detach().cpu()
            for b in range(B):
                for j in range(K):
                    out[b, j] = self.sid_tuple_to_item.get(tuple(x[b, j].tolist()), -1)
            return out
        return beams

    def _build_token_trie(self, semantic_ids: torch.LongTensor) -> TokenTrie:
        """
        Args:
            semantic_ids (torch.Tensor): tensor of shape (*, C), where C - codebook size.
        """
        token_trie = TokenTrie()

        for sid in semantic_ids[1:]:  # NOTE: ignore padding SID
            token_trie.add(sid.tolist())

        return token_trie
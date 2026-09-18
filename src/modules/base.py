"""
Pytorch Lightning Base module.
"""

import pytorch_lightning as pl
import torch


class SeqRecBase(pl.LightningModule):

    def __init__(self,
                 model,
                 lr=1e-3,
                 padding_idx=0,
                 predict_top_k=10,
                 optimizer=None,
                 lr_scheduler=None):

        super().__init__()

        self.model = model
        self.lr = lr
        self.padding_idx = padding_idx
        self.predict_top_k = predict_top_k
        self._optimizer_factory = optimizer        # callable: (params, lr) -> Optimizer
        self._lr_scheduler_factory = lr_scheduler  # callable: (optimizer, total_steps) -> Scheduler

    def configure_optimizers(self):

        if self._optimizer_factory is not None:
            optimizer = self._optimizer_factory(self.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        if self._lr_scheduler_factory is None:
            return optimizer

        total_steps = self.trainer.estimated_stepping_batches
        scheduler = self._lr_scheduler_factory(optimizer, total_steps)

        # Step-wise schedules need nothing beyond the scheduler; metric-driven ones
        # (ReduceLROnPlateau) must step once per epoch and be told which metric to watch.
        # A factory declares those needs via `interval` / `monitor` attributes.
        scheduler_config = {
            "scheduler": scheduler,
            "interval": getattr(self._lr_scheduler_factory, "interval", "step"),
        }
        monitor = getattr(self._lr_scheduler_factory, "monitor", None)
        if monitor is not None:
            scheduler_config["monitor"] = monitor

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    def predict_step(self, batch, batch_idx):

        preds, scores = self.make_prediction(batch)

        scores = scores.detach().cpu().float().numpy()
        preds = preds.detach().cpu().numpy()
        user_ids = batch['user_id'].detach().cpu().numpy()

        return {'preds': preds, 'scores': scores, 'user_ids': user_ids}

    
    def compute_val_metrics_at_k(self, targets: torch.LongTensor, preds: torch.LongTensor) -> dict[str, float]:
        """
        Args:
            targets (torch.LongTensor): tensor of shape (B,).
            preds (torch.LongTensor): tensor of shape (B, K).

        Returns:
            dict[str, float]: computed metrics.
    
        """
        hits = preds.eq(targets.unsqueeze(-1))

        # compute Hit Rate
        hit_rate = hits.any(dim=1).float().mean().item()

        hits = hits.float()
        ranks = torch.arange(1, hits.shape[1] + 1, device=hits.device, dtype=hits.dtype)

        # compute Mean Reciprocal Rank (MRR)
        # NOTE: top-k predictions contain unique items, so each row has at most one hit
        mrr = (hits / ranks).sum(dim=1).mean().item()

        # compute Normalized Discounted Cumulative Gain (NDCG)
        # NOTE: formally, this code computes DCG, which is equivalent to NDCG since the ground truth
        # contains exactly one item
        ndcg = (hits / torch.log2(ranks + 1)).sum(dim=1).mean().item()
        return {'hit_rate': hit_rate, 'mrr': mrr, 'ndcg': ndcg}


    def compute_val_metrics(self, targets: torch.LongTensor, preds: torch.LongTensor) -> dict[str, float]:
        """
        Args:
            targets (torch.LongTensor): tensor of shape (B,).
            preds (torch.LongTensor): tensor of shape (B, K).

        Returns:
            dict[str, float]: computed metrics.
        """
        metrics = {}
        for k in [1, 5, self.predict_top_k]:
            current_metrics = self.compute_val_metrics_at_k(targets, preds[:, :k])
            metrics.update({f'hit_rate_at_k/{k}': current_metrics['hit_rate'], f'mrr_at_k/{k}': current_metrics['mrr'], f'ndcg_at_k/{k}': current_metrics['ndcg']})
            if k == self.predict_top_k:
                metrics.update({f'hit_rate': current_metrics['hit_rate'], f'mrr': current_metrics['mrr'], f'ndcg': current_metrics['ndcg']})
        return metrics

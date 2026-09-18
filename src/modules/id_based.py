"""Full-softmax item-ID training and ranking."""

import torch
from torch import nn

from src.modules.base import SeqRecBase


class SeqRec(SeqRecBase):

    def __init__(self, model, lr=1e-3, padding_idx=0, predict_top_k=10,
                 filter_seen=True, optimizer=None, lr_scheduler=None):
        super().__init__(model, lr=lr, padding_idx=padding_idx, predict_top_k=predict_top_k,
                         optimizer=optimizer, lr_scheduler=lr_scheduler)
        self.filter_seen = filter_seen

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch['input_ids'], batch['attention_mask'])
        loss = self.compute_loss(outputs, batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def compute_loss(self, outputs, batch):
        loss_fct = nn.CrossEntropyLoss()
        return loss_fct(outputs.view(-1, outputs.size(-1)), batch['labels'].view(-1))

    def prediction_output(self, batch):
        return self.model(batch['input_ids'], batch['attention_mask'])

    def validation_step(self, batch, batch_idx):
        outputs = self.prediction_output(batch)
        if 'labels' in batch:
            loss = self.compute_loss(outputs, batch)
            self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True,
                     batch_size=batch['input_ids'].shape[0])
        preds, scores = self.convert_outputs_to_preds(outputs, batch)
        metrics = self.compute_val_metrics(batch['target'], preds)
        self.log("val_ndcg", metrics['ndcg'], prog_bar=True, batch_size=batch['input_ids'].shape[0])
        self.log("val_hit_rate", metrics['hit_rate'], prog_bar=True, batch_size=batch['input_ids'].shape[0])
        self.log("val_mrr", metrics['mrr'], prog_bar=True, batch_size=batch['input_ids'].shape[0])

    def convert_outputs_to_preds(self, outputs, batch):
        input_ids = batch['input_ids']
        rows_ids = torch.arange(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        last_item_idx = (input_ids != self.padding_idx).sum(axis=1) - 1
        preds = outputs[rows_ids, last_item_idx, :]
        if self.filter_seen:
            seen_items = batch['full_history']
            seen_mask = torch.zeros_like(preds, dtype=torch.bool, device=input_ids.device)
            seen_mask.scatter_(1, seen_items.clamp(min=0, max=preds.shape[1] - 1), True)
            preds = preds.masked_fill(seen_mask, float("-inf"))
        scores, preds = torch.topk(preds, self.predict_top_k, dim=1)
        return preds, scores

    def make_prediction(self, batch):
        return self.convert_outputs_to_preds(self.prediction_output(batch), batch)

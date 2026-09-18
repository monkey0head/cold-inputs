"""
GPTRec model.
"""

import torch
from torch import nn
from transformers import GPT2Config, GPT2Model
from transformers.cache_utils import DynamicCache

from src.models.beam_search import BeamSearchMixin


class GPTRec(BeamSearchMixin, nn.Module):

    def __init__(self, gpt_config, vocab_size=None, embeddings_matrix=None,
                 add_head=True, tie_weights=True, padding_idx=0, init_std=0.02):

        super().__init__()

        self.gpt_config = gpt_config
        self.vocab_size = vocab_size or len(embeddings_matrix)
        self.embeddings_matrix = embeddings_matrix
        self.add_head = add_head
        self.tie_weights = tie_weights
        self.padding_idx = padding_idx
        self.init_std = init_std

        if embeddings_matrix is not None:
            self.embed_layer = nn.Embedding.from_pretrained(
                torch.FloatTensor(embeddings_matrix), freeze=False, padding_idx=padding_idx)
        elif vocab_size is not None:
            self.embed_layer = nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=gpt_config['n_embd'],
                                            padding_idx=padding_idx)
        else:
            raise ValueError('Either vocab_size or embeddings_matrix should be not None.')

        self.transformer_model = GPT2Model(GPT2Config(**gpt_config))

        if self.add_head:
            self.head = nn.Linear(gpt_config['n_embd'], self.vocab_size, bias=False)
            if self.tie_weights:
                self.head.weight = self.embed_layer.weight

        self.init_weights()

    def init_weights(self):

        # initialization in huggingface transformers
        # https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/gpt2/modeling_gpt2.py#L462
        # initialization in pytorch Embeddings
        # https://github.com/pytorch/pytorch/blob/1.7/torch/nn/modules/sparse.py#L117

        if self.embeddings_matrix is None:
            self.embed_layer.weight.data.normal_(mean=0.0, std=self.init_std)
            if self.padding_idx is not None:
                self.embed_layer.weight.data[self.padding_idx].zero_()
        else:
            self.embed_layer.weight.data = \
                self.embed_layer.weight.data / self.embed_layer.weight.std() * self.init_std

    def forward(self, input_ids, attention_mask, return_logits: bool = True) -> torch.Tensor:

        embeds = self.embed_layer(input_ids)
        transformer_outputs = self.transformer_model(
            inputs_embeds=embeds, attention_mask=attention_mask)
        outputs = transformer_outputs.last_hidden_state

        if return_logits and self.add_head:
            outputs = self.head(outputs)

        return outputs

    def _forward_with_kv_cache(self, input_ids, attention_mask,
                               past_key_values=None, position_ids=None,
                               step=None, **kwargs):

        embeds = self.embed_layer(input_ids)
        outputs = self.transformer_model(
            inputs_embeds=embeds, attention_mask=attention_mask,
            past_key_values=past_key_values, position_ids=position_ids,
            use_cache=True)

        return outputs.last_hidden_state, outputs.past_key_values

    def _expand_kv_cache(self, past_key_values, beam_size):

        if isinstance(past_key_values, DynamicCache):
            past_key_values.batch_repeat_interleave(beam_size)
            return past_key_values

        return super()._expand_kv_cache(past_key_values, beam_size)

    def _reorder_kv_cache(self, past_key_values, beam_idx):

        if isinstance(past_key_values, DynamicCache):
            past_key_values.reorder_cache(beam_idx)
            return past_key_values

        return super()._reorder_kv_cache(past_key_values, beam_idx)

    def _compute_logits(self, hidden):

        return self.head(hidden)

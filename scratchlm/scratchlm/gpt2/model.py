"""The full from-scratch GPT-2 model: token + position embeddings, a stack
of pre-norm Transformer blocks, a final LayerNorm, and a weight-tied
language-modeling head."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import Block
from .config import GPT2Config


class GPT2(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying

        self.apply(self._init_weights)
        # GPT-2 paper: scale residual projection weights by 1/sqrt(2*n_layer)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, use_cache: bool = False, past_kvs=None):
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"sequence length {T} exceeds block_size {self.config.block_size}"
        )

        past_len = past_kvs[0][0].size(2) if past_kvs is not None else 0
        pos = torch.arange(past_len, past_len + T, device=idx.device)

        x = self.wte(idx) + self.wpe(pos)[None, :, :]
        x = self.drop(x)

        presents = [] if use_cache else None
        for i, block in enumerate(self.h):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, present = block(x, use_cache=use_cache, past_kv=past_kv)
            if use_cache:
                presents.append(present)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss, presents

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wpe.weight.numel()
        return n

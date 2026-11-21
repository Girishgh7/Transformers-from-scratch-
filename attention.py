"""Causal self-attention with a fused QKV projection (matches GPT-2's
checkpoint layout) and optional KV-caching for fast autoregressive
generation."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPT2Config


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # fused q, k, v projection -- matches GPT-2's checkpoint layout exactly
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("bias_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x, use_cache: bool = False, past_kv=None):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        present = (k, v) if use_cache else None

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        Tq, Tk = att.size(-2), att.size(-1)
        mask = self.bias_mask[:, :, Tk - Tq:Tk, :Tk]
        att = att.masked_fill(mask == 0, torch.finfo(att.dtype).min)
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.c_proj(out))
        return out, present

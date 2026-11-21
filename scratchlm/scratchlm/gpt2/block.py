"""Pre-LayerNorm Transformer block, GPT-2 style:

    x = x + attn(ln_1(x))
    x = x + mlp(ln_2(x))

The residual stream itself is never normalized, only the branch inputs are.
"""

import torch.nn as nn

from .attention import CausalSelfAttention
from .config import GPT2Config
from .mlp import MLP


class Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, use_cache: bool = False, past_kv=None):
        attn_out, present = self.attn(self.ln_1(x), use_cache=use_cache, past_kv=past_kv)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present

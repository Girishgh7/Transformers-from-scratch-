"""Configuration for the from-scratch GPT-2 model."""

from dataclasses import dataclass


@dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0  # 0.0 for inference / fine-tuning with pretrained weights
    bias: bool = True  # GPT-2 uses biases on all Linears and LayerNorms


# n_layer, n_head, n_embd for each released OpenAI GPT-2 checkpoint size
GPT2_SIZES = {
    "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
    "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
    "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
}

"""Attention mask helpers for the encoder-decoder Transformer."""

import torch


def make_src_mask(src: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """Padding mask for the encoder: (B, 1, 1, S), True where real tokens are."""
    return (src != pad_id).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """Combined padding + causal mask for the decoder: (B, 1, T, T)."""
    B, T = tgt.size()
    pad_mask = (tgt != pad_id).unsqueeze(1).unsqueeze(2)
    causal_mask = torch.tril(torch.ones((T, T), device=tgt.device)).bool().unsqueeze(0).unsqueeze(0)
    return pad_mask & causal_mask

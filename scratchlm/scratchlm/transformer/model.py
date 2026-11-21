"""Post-norm encoder-decoder Transformer, e.g. for sequence-to-sequence
tasks like translation."""

import torch
import torch.nn as nn

from .layers import FeedForward, MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = self.norm1(x + self.attn(x, x, x, mask))
        x = self.norm2(x + self.ffn(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads)
        self.cross_attn = MultiHeadAttention(d_model, heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc, src_mask, tgt_mask):
        x = self.norm1(x + self.self_attn(x, x, x, tgt_mask))
        x = self.norm2(x + self.cross_attn(x, enc, enc, src_mask))
        x = self.norm3(x + self.ffn(x))
        return x


class Transformer(nn.Module):
    """Full encoder-decoder Transformer with shared token + position embeddings."""

    def __init__(
        self,
        vocab: int,
        d_model: int = 384,
        layers: int = 4,
        heads: int = 6,
        d_ff: int = 1536,
        max_len: int = 512,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.enc = nn.ModuleList([EncoderLayer(d_model, heads, d_ff) for _ in range(layers)])
        self.dec = nn.ModuleList([DecoderLayer(d_model, heads, d_ff) for _ in range(layers)])
        self.fc = nn.Linear(d_model, vocab)

    def forward(self, src, tgt, src_mask, tgt_mask):
        B, S = src.size()
        T = tgt.size(1)
        src = self.embed(src) + self.pos(torch.arange(S, device=src.device))
        tgt = self.embed(tgt) + self.pos(torch.arange(T, device=tgt.device))
        for layer in self.enc:
            src = layer(src, src_mask)
        for layer in self.dec:
            tgt = layer(tgt, src, src_mask, tgt_mask)
        return self.fc(tgt)

    @torch.no_grad()
    def encode(self, src, src_mask):
        B, S = src.size()
        x = self.embed(src) + self.pos(torch.arange(S, device=src.device))
        for layer in self.enc:
            x = layer(x, src_mask)
        return x

    @torch.no_grad()
    def decode_step(self, tgt, enc_out, src_mask, tgt_mask):
        B, T = tgt.size()
        x = self.embed(tgt) + self.pos(torch.arange(T, device=tgt.device))
        for layer in self.dec:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.fc(x)

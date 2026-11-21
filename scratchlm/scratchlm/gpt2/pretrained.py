"""Load official pretrained GPT-2 weights (via Hugging Face) into the
from-scratch `GPT2` model defined in `model.py`.

Hugging Face keeps GPT-2's original TensorFlow-style `Conv1D` layers for
`c_attn`/`c_proj` (attention) and `c_fc`/`c_proj` (MLP). `Conv1D.weight`
has shape (in_features, out_features) -- the transpose of
`nn.Linear.weight`'s (out_features, in_features) -- so those four weight
matrices need `.t()` before copying into our `nn.Linear` layers.
"""

from typing import Tuple

import torch

from .config import GPT2Config, GPT2_SIZES
from .model import GPT2

_TRANSPOSED_SUFFIXES = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)


@torch.no_grad()
def gpt2_from_pretrained(model_type: str = "gpt2") -> Tuple[GPT2, GPT2Config]:
    """Build a from-scratch `GPT2` and load OpenAI's pretrained weights into it.

    Requires the `transformers` package (`pip install transformers`).
    `model_type` is one of: "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl".
    """
    if model_type not in GPT2_SIZES:
        raise ValueError(f"unknown model_type {model_type!r}, expected one of {list(GPT2_SIZES)}")

    from transformers import GPT2LMHeadModel  # local import: optional dependency

    config = GPT2Config(vocab_size=50257, block_size=1024, **GPT2_SIZES[model_type])
    model = GPT2(config)
    sd = model.state_dict()
    sd_keys = [k for k in sd.keys() if not k.endswith(".bias_mask")]

    hf_model = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = hf_model.state_dict()
    sd_keys_hf = [
        k
        for k in sd_hf.keys()
        if not k.endswith(".attn.bias") and not k.endswith(".attn.masked_bias")
    ]
    # lm_head is tied to wte in both models, so we only need the `transformer.*` keys
    sd_keys_hf = [k for k in sd_keys_hf if k.startswith("transformer.")]

    if len(sd_keys_hf) != len(sd_keys):
        raise RuntimeError(
            f"mismatched key counts: {len(sd_keys_hf)} (hf) vs {len(sd_keys)} (ours)"
        )

    for k_ours in sd_keys:
        k_hf = "transformer." + k_ours
        src = sd_hf[k_hf]
        dst = sd[k_ours]
        if any(k_ours.endswith(suffix) for suffix in _TRANSPOSED_SUFFIXES):
            if src.shape[::-1] != dst.shape:
                raise RuntimeError(f"shape mismatch at {k_ours}: {src.shape} vs {dst.shape}")
            dst.copy_(src.t())
        else:
            if src.shape != dst.shape:
                raise RuntimeError(f"shape mismatch at {k_ours}: {src.shape} vs {dst.shape}")
            dst.copy_(src)

    return model, config

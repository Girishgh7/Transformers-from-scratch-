"""Autoregressive sampling loop for the from-scratch GPT-2, using the
model's built-in KV-cache so each new token only costs one forward step."""

import torch
import torch.nn.functional as F

from .model import GPT2


@torch.no_grad()
def generate(
    model: GPT2,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 40,
    temperature: float = 0.8,
    top_k: int = 50,
    device: str = "cpu",
) -> str:
    """Generate text continuing `prompt`.

    `tokenizer` is any Hugging Face GPT-2 tokenizer
    (e.g. `GPT2Tokenizer.from_pretrained("gpt2")`).
    """
    model.eval()
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    past_kvs = None
    cur_ids = ids
    for _ in range(max_new_tokens):
        step_input = cur_ids if past_kvs is None else cur_ids[:, -1:]
        logits, _, past_kvs = model(step_input, use_cache=True, past_kvs=past_kvs)

        next_logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(next_logits, top_k)
            next_logits[next_logits < v[:, [-1]]] = -float("inf")

        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        cur_ids = torch.cat([cur_ids, next_id], dim=1)
        if next_id.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(cur_ids[0].tolist())

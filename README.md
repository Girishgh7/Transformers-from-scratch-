# scratchlm

Language models built from scratch, with no model internals borrowed from any
existing library:

- **`scratchlm.ngram`** — character-level Unigram, Bigram, Trigram, and general
  N-gram models for generating names.
- **`scratchlm.transformer`** — a post-norm encoder-decoder Transformer
  (the original "Attention Is All You Need" layout), for sequence-to-sequence
  tasks such as translation.
- **`scratchlm.gpt2`** — a pre-norm, decoder-only, GPT-2-style Transformer,
  built entirely from `nn.Linear`/`nn.LayerNorm`/`nn.Embedding` primitives,
  with a loader that copies official pretrained GPT-2 weights (via Hugging
  Face) into it tensor-by-tensor.

## Install

```bash
git clone <this-repo-url>
cd scratchlm
pip install -e .

# only needed for scratchlm.gpt2's pretrained-weight loader / tokenizer:
pip install -e ".[pretrained]"
```

## `scratchlm.ngram` — name generation

```python
from scratchlm.ngram import load_names, UnigramModel, BigramModel, TrigramModel, NGramModel

names = load_names()  # ~32k US first names bundled with the package

uni = UnigramModel().fit(names)
bi  = BigramModel().fit(names)
tri = TrigramModel().fit(names)
four = NGramModel(n=4).fit(names)   # NGramModel(n=1)==unigram, n=2==bigram, n=3==trigram

print("Unigram :", uni.generate(5, seed=0))
print("Bigram  :", bi.generate(5, seed=0))
print("Trigram :", tri.generate(5, seed=0))
print("4-gram  :", four.generate(5, seed=0))

print("perplexity of the 4-gram model:", four.perplexity(names[:500]))
```

As context length grows (unigram -> bigram -> trigram -> n-gram), the
generated strings go from unpronounceable noise to names that increasingly
look plausible but are more prone to memorizing training examples verbatim
when the context gets long relative to the dataset size.

## `scratchlm.transformer` — encoder-decoder Transformer

```python
import torch
from scratchlm.transformer import Transformer, make_src_mask, make_tgt_mask

model = Transformer(vocab=16000, d_model=384, layers=4, heads=6, d_ff=1536)

src = torch.randint(1, 16000, (2, 20))
tgt = torch.randint(1, 16000, (2, 18))
logits = model(src, tgt, make_src_mask(src), make_tgt_mask(tgt))
print(logits.shape)  # (2, 18, 16000)
```

## `scratchlm.gpt2` — GPT-2 from scratch, initialized from pretrained weights

```python
from transformers import GPT2Tokenizer
from scratchlm.gpt2 import gpt2_from_pretrained, generate

model, config = gpt2_from_pretrained("gpt2")  # 124M params; also: gpt2-medium/large/xl
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

print(generate(model, tokenizer, "The quick brown fox", max_new_tokens=40))
```

Architecturally, `scratchlm.gpt2` differs from `scratchlm.transformer` in the
ways the real GPT-2 differs from the original Transformer: decoder-only (no
cross-attention), pre-LayerNorm residual blocks, tanh-approximate GELU,
learned absolute position embeddings, a fused QKV projection, and weight
tying between the token embedding and the output head.

## Repository layout

```
scratchlm/
  ngram/          unigram.py, bigram.py, trigram.py, ngram.py, data_utils.py, data/names.txt
  transformer/    layers.py, model.py, masks.py
  gpt2/           config.py, attention.py, mlp.py, block.py, model.py, pretrained.py, generate.py
```

## Credits

1. Belongs to Girish GH.

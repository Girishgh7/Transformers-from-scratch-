# sigBert v3 (distilled from google/siglip-base-patch16-224)

Files: `sigbert_model.py` (torch only), `config.json`, `sigbert.pt`, `report.json`, `figures/`.

```python
from transformers import AutoTokenizer
from sigbert_model import SigBert
tok = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")   # tokenizer only; SigLIP weights are NOT needed
model = SigBert.from_pretrained("path/to/this/folder").eval()
emb = model.encode_texts(["a red mug on a wooden table"], tok)            # [B, 768] L2-normalised, SigLIP joint space
# per-token features for grounding heads:
# ids, lens = tokenize_to_padded(tok, texts, 64, model.cfg.pad_token_id); mask = lengths_to_mask(lens, ids.shape[1])
# tokens, _ = model.encode_tokens(ids, mask)                              # [B, L, 256]
```
The embeddings are compatible with SigLIP image embeddings (`SiglipModel.vision_model(...).pooler_output`) for retrieval / grounding.

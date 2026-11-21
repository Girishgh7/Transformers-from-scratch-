"""General character-level n-gram name model, of configurable order.

`NGramModel(n=1)` behaves like `UnigramModel`, `n=2` like `BigramModel`,
`n=3` like `TrigramModel`; `n >= 4` extends the same counting idea to
longer contexts (at the cost of increasingly sparse statistics -- with
only ~32k training names, n=5 or higher will mostly memorize rather than
generalize).
"""

import math
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .data_utils import build_vocab


class NGramModel:
    """P(next_char | prev_(n-1)_chars) via smoothed, sparsely-stored counts."""

    def __init__(self, n: int = 4, smoothing: float = 1.0):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self.smoothing = smoothing
        self.stoi: dict = {}
        self.itos: dict = {}
        self.counts: Dict[Tuple[int, ...], List[float]] = {}
        self._V = 0

    @property
    def context_len(self) -> int:
        return self.n - 1

    def fit(self, names: List[str]) -> "NGramModel":
        self.stoi, self.itos = build_vocab(names)
        V = len(self.stoi)
        self._V = V
        ctx_len = self.context_len
        counts: Dict[Tuple[int, ...], List[float]] = defaultdict(lambda: [self.smoothing] * V)

        for name in names:
            seq = "." * ctx_len + name + "."
            ids = [self.stoi[c] for c in seq]
            for i in range(ctx_len, len(ids)):
                context = tuple(ids[i - ctx_len:i])
                counts[context][ids[i]] += 1

        self.counts = dict(counts)
        return self

    def _probs_for(self, context: Tuple[int, ...]) -> List[float]:
        row = self.counts.get(context)
        if row is None:
            row = [1.0] * self._V  # uniform back-off for an unseen context
        total = sum(row)
        return [c / total for c in row]

    def generate(self, n_samples: int = 10, max_len: int = 20, seed: Optional[int] = None) -> List[str]:
        if not self.counts:
            raise RuntimeError("call .fit(names) before .generate(...)")
        rng = random.Random(seed)
        start = self.stoi["."]
        ctx_len = self.context_len
        results = []
        for _ in range(n_samples):
            out: List[str] = []
            context = [start] * ctx_len
            for _ in range(max_len):
                probs = self._probs_for(tuple(context))
                idx = rng.choices(range(len(probs)), weights=probs, k=1)[0]
                ch = self.itos[idx]
                if ch == ".":
                    break
                out.append(ch)
                if ctx_len > 0:
                    context = context[1:] + [idx]
            results.append("".join(out))
        return results

    def log_likelihood(self, name: str) -> float:
        """Log-likelihood of a name under the fitted model (base e).

        Returns -inf if the name contains a character outside the
        training vocabulary.
        """
        if not self.counts:
            raise RuntimeError("call .fit(names) before .log_likelihood(...)")
        ctx_len = self.context_len
        seq = "." * ctx_len + name.lower() + "."
        ids = []
        for c in seq:
            if c not in self.stoi:
                return float("-inf")
            ids.append(self.stoi[c])

        ll = 0.0
        for i in range(ctx_len, len(ids)):
            context = tuple(ids[i - ctx_len:i])
            probs = self._probs_for(context)
            ll += math.log(probs[ids[i]])
        return ll

    def perplexity(self, names: List[str]) -> float:
        """Average per-character perplexity over a list of names."""
        total_ll = 0.0
        total_chars = 0
        for name in names:
            total_ll += self.log_likelihood(name)
            total_chars += len(name) + 1  # +1 for the end sentinel
        return math.exp(-total_ll / total_chars)

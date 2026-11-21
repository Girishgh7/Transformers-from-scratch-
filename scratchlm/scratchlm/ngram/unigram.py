"""Character-level unigram name model.

Each character is sampled independently from the overall character
frequency distribution of the training names -- no context at all. This
is the weakest of the four models and serves as a baseline: expect
unpronounceable, unstructured output.
"""

import random
from collections import Counter
from typing import List, Optional

from .data_utils import build_vocab


class UnigramModel:
    """P(char) estimated from raw character frequency, context-free."""

    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing
        self.stoi: dict = {}
        self.itos: dict = {}
        self.probs: Optional[List[float]] = None

    def fit(self, names: List[str]) -> "UnigramModel":
        self.stoi, self.itos = build_vocab(names)
        counts: Counter = Counter()
        for name in names:
            counts.update("." + name + ".")

        vocab_size = len(self.stoi)
        freqs = [counts.get(self.itos[i], 0) + self.smoothing for i in range(vocab_size)]
        total = sum(freqs)
        self.probs = [f / total for f in freqs]
        return self

    def generate(self, n: int = 10, max_len: int = 20, seed: Optional[int] = None) -> List[str]:
        if self.probs is None:
            raise RuntimeError("call .fit(names) before .generate(...)")
        rng = random.Random(seed)
        results = []
        for _ in range(n):
            out = []
            for _ in range(max_len):
                idx = rng.choices(range(len(self.probs)), weights=self.probs, k=1)[0]
                ch = self.itos[idx]
                if ch == ".":
                    break
                out.append(ch)
            results.append("".join(out))
        return results

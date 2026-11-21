"""Character-level bigram name model.

P(next_char | current_char), estimated with Laplace (additive) smoothing
over a V x V count table, where V is the vocabulary size (26 letters +
the start/end sentinel). Generation is a simple Markov chain: sample the
next character conditioned only on the current one, starting and
stopping at the sentinel token.
"""

import random
from typing import List, Optional

from .data_utils import build_vocab


class BigramModel:
    """P(next_char | current_char) via a smoothed V x V transition table."""

    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing
        self.stoi: dict = {}
        self.itos: dict = {}
        self.probs: Optional[List[List[float]]] = None  # probs[i][j] = P(j | i)

    def fit(self, names: List[str]) -> "BigramModel":
        self.stoi, self.itos = build_vocab(names)
        V = len(self.stoi)
        counts = [[self.smoothing] * V for _ in range(V)]

        for name in names:
            seq = "." + name + "."
            for c1, c2 in zip(seq, seq[1:]):
                counts[self.stoi[c1]][self.stoi[c2]] += 1

        self.probs = []
        for row in counts:
            total = sum(row)
            self.probs.append([c / total for c in row])
        return self

    def generate(self, n: int = 10, max_len: int = 20, seed: Optional[int] = None) -> List[str]:
        if self.probs is None:
            raise RuntimeError("call .fit(names) before .generate(...)")
        rng = random.Random(seed)
        start = self.stoi["."]
        results = []
        for _ in range(n):
            out = []
            idx = start
            for _ in range(max_len):
                row = self.probs[idx]
                idx = rng.choices(range(len(row)), weights=row, k=1)[0]
                ch = self.itos[idx]
                if ch == ".":
                    break
                out.append(ch)
            results.append("".join(out))
        return results

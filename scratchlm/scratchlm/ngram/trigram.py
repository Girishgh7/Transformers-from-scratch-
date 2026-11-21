"""Character-level trigram name model.

P(next_char | prev_2_chars). Contexts are stored sparsely in a dict
(rather than a dense V x V x V tensor) since most 2-character contexts
never occur. Unseen contexts back off to a uniform distribution.
"""

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .data_utils import build_vocab


class TrigramModel:
    """P(next_char | prev_2_chars) via smoothed, sparsely-stored counts."""

    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing
        self.stoi: dict = {}
        self.itos: dict = {}
        self.counts: Dict[Tuple[int, int], List[float]] = {}
        self._V = 0

    def fit(self, names: List[str]) -> "TrigramModel":
        self.stoi, self.itos = build_vocab(names)
        V = len(self.stoi)
        self._V = V
        counts: Dict[Tuple[int, int], List[float]] = defaultdict(lambda: [self.smoothing] * V)

        for name in names:
            seq = ".." + name + "."
            ids = [self.stoi[c] for c in seq]
            for c1, c2, c3 in zip(ids, ids[1:], ids[2:]):
                counts[(c1, c2)][c3] += 1

        self.counts = dict(counts)
        return self

    def _probs_for(self, context: Tuple[int, int]) -> List[float]:
        row = self.counts.get(context)
        if row is None:
            row = [1.0] * self._V  # uniform back-off for an unseen context
        total = sum(row)
        return [c / total for c in row]

    def generate(self, n: int = 10, max_len: int = 20, seed: Optional[int] = None) -> List[str]:
        if not self.counts:
            raise RuntimeError("call .fit(names) before .generate(...)")
        rng = random.Random(seed)
        start = self.stoi["."]
        results = []
        for _ in range(n):
            out = []
            c1, c2 = start, start
            for _ in range(max_len):
                probs = self._probs_for((c1, c2))
                idx = rng.choices(range(len(probs)), weights=probs, k=1)[0]
                ch = self.itos[idx]
                if ch == ".":
                    break
                out.append(ch)
                c1, c2 = c2, idx
            results.append("".join(out))
        return results

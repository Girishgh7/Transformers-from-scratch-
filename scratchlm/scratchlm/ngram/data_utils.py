"""Shared helpers for the character-level n-gram name models."""

import os
from typing import Dict, List, Optional, Tuple

DEFAULT_NAMES_PATH = os.path.join(os.path.dirname(__file__), "data", "names.txt")


def load_names(path: Optional[str] = None) -> List[str]:
    """Load one name per line from a text file.

    Defaults to the bundled dataset (~32k US first names).
    """
    path = path or DEFAULT_NAMES_PATH
    with open(path, "r", encoding="utf-8") as f:
        names = [line.strip().lower() for line in f if line.strip()]
    return names


def build_vocab(
    names: List[str], start_end_token: str = "."
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build a character vocabulary from a list of names.

    Index 0 is always the sentinel `start_end_token`, used to mark the
    start and end of a name (same convention as Karpathy's makemore).
    """
    chars = sorted(set("".join(names)))
    vocab = [start_end_token] + chars
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

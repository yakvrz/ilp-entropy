"""ilp_entropy.io
~~~~~~~~~~~~~~~~~~
Corpus loading and indexing helpers.

The canonical English unigram list lives at **``data/unigrams_en.csv``**
with two columns:

* ``unigram``       - raw word (lower-case ASCII)
* ``unigram_freq``  - frequency (float)

This module
1. **Cleans** the CSV (filters non-alphabetic rows, enforces 4-11-letter range).
2. **Encodes** every word as ``uint8`` letter codes ``0-25``.
3. Exposes a single public accessor: :pyfunc:`get_corpus_index` which returns
   a *cached* dict keyed by word length ⇒ (codes, freqs) ndarray pair.

All heavy objects are memoised with ``functools.lru_cache`` so the CSV is
parsed once per session.

Author: Koby Raz
"""

from __future__ import annotations

import functools
import string
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

__all__ = [
    "get_corpus_index",
    "get_top_words",
]

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #
_ALPHABET: str = string.ascii_lowercase  # 'abcdefghijklmnopqrstuvwxyz'
_ALPHA_TO_INT = {ch: i for i, ch in enumerate(_ALPHABET)}

# Project-relative path to the CSV - resolves to ``PROJECT_ROOT/data/…``
_DATA_PATH: Path = (Path(__file__).resolve().parent.parent / "data" / "unigrams_en.csv").resolve()

# Default word-length range we care about
_DEFAULT_RANGE = range(4, 12)  # 4-through-11 inclusive


# --------------------------------------------------------------------------- #
# Helper functions                                                             #
# --------------------------------------------------------------------------- #

def _encode_word(word: str) -> np.ndarray:
    """Return ``uint8[L]`` array with values 0-25 for each character."""
    # We assume caller has already validated [a-z]+
    return np.fromiter((_ALPHA_TO_INT[c] for c in word), dtype=np.uint8, count=len(word))

# ---------------------------------------------------------------------------
# Internal loader (hashable signature)                                       #
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _get_corpus_index_cached(
    csv_path: str | Path | None,
    word_lengths: tuple[int, ...],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Core loader; used only through the public wrapper.  Every parameter is
    hashable, so @lru_cache works regardless of how the caller supplied them.
    """
    csv_path = Path(csv_path or 'data/unigrams_en.csv')
    df = pd.read_csv(csv_path, usecols=["unigram", "unigram_freq"])

    # Filter to only alphabetic words (a-z) before processing
    df = df[df.unigram.str.match(r'^[a-z]+$', na=False)]

    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for L in word_lengths:
        sub = df[df.unigram.str.len() == L]
        if sub.empty:
            continue
        # encode letters a-z → 0-25
        codes = (
            sub.unigram.str.lower()
            .apply(lambda s: [ord(ch) - 97 for ch in s])
            .explode()
            .astype("uint8")
            .to_numpy()
            .reshape(-1, L)
        )
        freqs = sub.unigram_freq.to_numpy(dtype="float32")
        out[L] = (codes, freqs)
    return out


# ---------------------------------------------------------------------------
# Public API                                                                 #
# ---------------------------------------------------------------------------
def get_corpus_index(
    csv_path: str | Path | None = None,
    *,
    word_lengths: Iterable[int] = _DEFAULT_RANGE,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Load (and cache) the requested slice(s) of the unigram corpus.

    ``word_lengths`` can be any iterable - list, set, range, etc.
    """
    return _get_corpus_index_cached(csv_path, tuple(word_lengths))

def get_top_words(
    n: int,
    csv_path: str | Path | None = None,
    *,
    word_lengths: Iterable[int] = _DEFAULT_RANGE,
    min_freq: float = 0.0,
) -> list[str]:
    """
    Get the top N most frequent words from the corpus.

    Parameters
    ----------
    n : int
        Number of top words to return
    csv_path : str or Path, optional
        Path to the corpus CSV file
    word_lengths : Iterable[int], optional
        Word lengths to consider (default: 4-11)
    min_freq : float, optional
        Minimum frequency threshold

    Returns
    -------
    list[str]
        Top N words sorted by frequency (descending)
    """
    csv_path = Path(csv_path or 'data/unigrams_en.csv')
    df = pd.read_csv(csv_path, usecols=["unigram", "unigram_freq"])

    # Filter to only alphabetic words and specified lengths
    df = df[df.unigram.str.match(r'^[a-z]+$', na=False)]
    df = df[df.unigram.str.len().isin(word_lengths)]
    df = df[df.unigram_freq >= min_freq]

    # Sort by frequency (descending) and take top N
    df = df.sort_values('unigram_freq', ascending=False).head(n)

    return df['unigram'].tolist()

"""
This module handles loading and processing of corpus data.

It reads a unigram frequency list from a CSV file, cleans the data,
and encodes the words into a numerical format suitable for computation.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "get_corpus_index",
    "get_corpus_words",
]


def get_corpus_words(
    csv_path: str | Path, min_freq: float = 1e-7
) -> list[str]:
    """
    Loads a corpus file and returns a list of words that meet the minimum
    frequency requirement.
    """
    df = pd.read_csv(str(csv_path), usecols=["word", "freq"])

    # Filter by minimum frequency and for valid alphabetic words
    df = df[df["freq"] >= min_freq]
    df = df[df["word"].str.match(r"^[a-z]+$", na=False)]

    return df["word"].tolist()


def get_corpus_index(
    word_lengths: Iterable[int],
    csv_path: str | Path,
    min_freq: float = 1e-7,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Loads and processes a corpus file into a structured dictionary.

    The function reads a CSV file, filters for valid alphabetic words of
    specified lengths, and returns a dictionary where keys are word lengths
    and values are tuples containing NumPy arrays for word codes and their
    corresponding frequencies.

    Args:
        word_lengths: An iterable of integer word lengths to include.
        csv_path: Optional path to the corpus CSV file. If None, a default
                  path is used.

    Returns:
        A dictionary mapping each word length to a tuple of (codes, freqs)
        NumPy arrays. `codes` is a 2D array of letter-to-integer mappings,
        and `freqs` is a 1D array of word frequencies.
    """
    df = pd.read_csv(str(csv_path), usecols=["word", "freq"])

    # Filter by minimum frequency and for valid alphabetic words
    df = df[df["freq"] >= min_freq]
    df = df[df["word"].str.match(r"^[a-z]+$", na=False)]

    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for L in word_lengths:
        sub = df[df["word"].str.len() == L]
        if sub.empty:
            continue

        codes = (
            sub["word"]
            .str.lower()
            .apply(lambda s: [ord(ch) - 97 for ch in s])
            .explode()
            .astype("uint8")
            .to_numpy()
            .reshape(-1, L)
        )
        freqs = sub["freq"].to_numpy(dtype="float32")
        out[L] = (codes, freqs)
    return out

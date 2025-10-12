"""
This module handles the loading and processing of corpus data for ILP-based
entropy calculations.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "load_corpus",
    "load_words",
]


def load_corpus(
    corpus_file: str | Path, min_freq: float = 1e-7
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Loads a corpus from a CSV file and performs initial cleaning and filtering.

    The function reads a CSV file expecting 'word' and 'freq' columns, filters out
    words below a minimum frequency, and ensures all words consist of lowercase
    alphabetic characters.

    Args:
        corpus_file (str | Path): The path to the corpus CSV file.
        min_freq (float): The minimum frequency threshold for including a word.

    Returns:
        A tuple containing:
        - A pandas DataFrame with the filtered and cleaned corpus data.
        - A dictionary mapping each character to its integer representation.
    """
    df = pd.read_csv(str(corpus_file), usecols=["word", "freq"])
    df = df[df["freq"] >= min_freq].copy()
    df["word"] = df["word"].str.lower()
    df = df[df["word"].str.match(r"^[a-z]+$", na=False)]

    # Create character-to-integer mapping from the corpus
    chars = sorted(list(set("".join(df["word"]))))
    char_map = {char: i for i, char in enumerate(chars)}

    return df, char_map


def load_words(word_file: str | Path) -> list[str]:
    """
    Loads a list of words from a text file.

    Each line in the file is treated as a single word. Leading/trailing whitespace
    is removed.

    Args:
        word_file (str | Path): The path to the text file containing words.

    Returns:
        A list of words.
    """
    with open(word_file, "r") as f:
        return [line.strip() for line in f if line.strip()]

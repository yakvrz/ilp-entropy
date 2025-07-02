"""ilp_entropy.config
~~~~~~~~~~~~~~~~~~~~~~

Configuration constants, default parameters, and validation functions for the
ILP Entropy package.

This module centralizes all configuration to ensure consistency across the
package and provides validation for user inputs.

Author: Koby Raz
"""

from __future__ import annotations

import string
from pathlib import Path
from typing import Any, Final, Iterable

__all__ = [
    "DEFAULT_DROP_LEFT",
    "DEFAULT_DROP_RIGHT",
    "DEFAULT_WORD_LENGTHS",
    "DEFAULT_MIN_FREQ",
    "DEFAULT_CORPUS_PATH",
    "DEFAULT_CACHE_DIR",
    "MAX_WORD_LENGTH",
    "MIN_WORD_LENGTH",
    "ALPHABET",
    "validate_drop_rate",
    "validate_word_length",
    "validate_word_lengths",
    "get_data_path",
    "get_cache_path",
]

# --------------------------------------------------------------------------- #
# Core constants                                                              #
# --------------------------------------------------------------------------- #

ALPHABET: Final[str] = string.ascii_lowercase
MIN_WORD_LENGTH: Final[int] = 1
MAX_WORD_LENGTH: Final[int] = 11  # Maximum supported by bit operations

# --------------------------------------------------------------------------- #
# Default parameters                                                          #
# --------------------------------------------------------------------------- #

DEFAULT_DROP_LEFT: Final[float] = 0.10
DEFAULT_DROP_RIGHT: Final[float] = 0.10
DEFAULT_WORD_LENGTHS: Final[range] = range(4, 12)  # 4-11 inclusive
DEFAULT_MIN_FREQ: Final[float] = 1e-7

# --------------------------------------------------------------------------- #
# Path configuration                                                          #
# --------------------------------------------------------------------------- #

def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent.parent

def get_data_path(filename: str | None = None) -> Path:
    """Return path to data directory or specific data file."""
    data_dir = get_project_root() / "data"
    if filename is None:
        return data_dir
    return data_dir / filename

def get_cache_path(subdir: str | None = None) -> Path:
    """Return path to cache directory or specific cache subdirectory."""
    cache_dir = Path.home() / ".cache" / "ilp_entropy"
    if subdir is None:
        return cache_dir
    return cache_dir / subdir

DEFAULT_CORPUS_PATH: Final[Path] = get_data_path("unigrams_en.csv")
DEFAULT_CACHE_DIR: Final[Path] = get_cache_path()

# --------------------------------------------------------------------------- #
# Validation functions                                                        #
# --------------------------------------------------------------------------- #

def validate_drop_rate(value: float, name: str = "drop_rate") -> float:
    """Validate that a drop rate is in the valid range [0, 1]."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")

    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in range [0, 1], got {value}")

    return float(value)

def validate_word_length(length: int, name: str = "word_length") -> int:
    """Validate that a word length is in the supported range."""
    if not isinstance(length, int):
        raise TypeError(f"{name} must be an integer, got {type(length).__name__}")

    if not (MIN_WORD_LENGTH <= length <= MAX_WORD_LENGTH):
        raise ValueError(
            f"{name} must be in range [{MIN_WORD_LENGTH}, {MAX_WORD_LENGTH}], got {length}"
        )

    return length

def validate_word_lengths(lengths: Iterable[int]) -> list[int]:
    """Validate a collection of word lengths."""
    validated = []
    for i, length in enumerate(lengths):
        try:
            validated.append(validate_word_length(length, f"word_lengths[{i}]"))
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid word length at index {i}: {e}") from e

    if not validated:
        raise ValueError("word_lengths cannot be empty")

    return sorted(set(validated))  # Remove duplicates and sort

def validate_word(word: str) -> str:
    """Validate that a word contains only lowercase letters."""
    if not isinstance(word, str):
        raise TypeError(f"word must be a string, got {type(word).__name__}")

    if not word:
        raise ValueError("word cannot be empty")

    if not word.islower():
        raise ValueError("word must be lowercase")

    if not all(c in ALPHABET for c in word):
        raise ValueError("word must contain only lowercase letters a-z")

    validate_word_length(len(word), "word length")

    return word

def validate_corpus_path(path: str | Path) -> Path:
    """Validate and resolve a corpus file path."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Corpus path is not a file: {path}")

    if path.suffix.lower() != '.csv':
        raise ValueError(f"Corpus file must be a CSV file, got: {path.suffix}")

    return path.resolve()

def validate_parameters(
    drop_left: float | None = None,
    drop_right: float | None = None,
    word_lengths: Iterable[int] | None = None,
    min_freq: float | None = None,
) -> dict[str, Any]:
    """Validate a set of parameters and return validated values."""
    validated: dict[str, Any] = {}

    if drop_left is not None:
        validated["drop_left"] = validate_drop_rate(drop_left, "drop_left")

    if drop_right is not None:
        validated["drop_right"] = validate_drop_rate(drop_right, "drop_right")

    if word_lengths is not None:
        validated["word_lengths"] = validate_word_lengths(word_lengths)

    if min_freq is not None:
        if not isinstance(min_freq, (int, float)):
            raise TypeError(f"min_freq must be a number, got {type(min_freq).__name__}")
        if min_freq < 0:
            raise ValueError(f"min_freq must be non-negative, got {min_freq}")
        validated["min_freq"] = float(min_freq)

    return validated

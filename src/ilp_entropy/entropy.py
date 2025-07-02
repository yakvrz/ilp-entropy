"""entropy.py
Compute Initial Landing-Position (ILP) entropy using the *vectorised* acuity
helpers from ``acuity.py``.

Key design choices
------------------
* **Delegation** All letter-level probabilities come from
  ``acuity_weights`` / ``weight_matrix`` - we never re-implement the drop-off.
* **Vectorised** ``ilp_entropy`` does a single NumPy call, so scanning all
  fixations is O(L²) in memory but *fast* in practice (<1µs for L≤15).
* **Lean API** Only three public functions - exactly the ones downstream code
  expects: ``position_entropy``, ``ilp_entropy``, ``optimal_fixation``.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np

# Relative import keeps intra-package references explicit.
from .acuity import acuity_weights, weight_matrix

__all__ = [
    "position_entropy",
    "ilp_entropy",
    "optimal_fixation",
]

# ---------------------------------------------------------------------------
# Single-fixation entropy
# ---------------------------------------------------------------------------


def position_entropy(
    word: str,
    fixation_pos: int,
    drop_left: float,
    drop_right: float,
    *,
    alphabet_size: int = 26,
    floor: float = 0.0,
) -> float:
    """Residual Shannon entropy (bits) for **one** landing position."""
    probs = acuity_weights(
        len(word),
        fixation_pos,
        drop_left=drop_left,
        drop_right=drop_right,
        floor=floor,
        dtype="float32",
    )  # shape (L,)
    unknown_bits = math.log2(alphabet_size)
    return float(np.sum((1.0 - probs) * unknown_bits))


# ---------------------------------------------------------------------------
# Full ILP curve (vectorised)
# ---------------------------------------------------------------------------


def ilp_entropy(
    word: str,
    drop_left: float,
    drop_right: float,
    *,
    alphabet_size: int = 26,
    floor: float = 0.0,
) -> list[float]:
    """Return the ILP-entropy curve *for every possible fixation* in ``word``."""
    word_len = len(word)
    # Pre-compute a (L×L) matrix of recognition probabilities
    weight_mat = weight_matrix(
        word_len,
        drop_left=drop_left,
        drop_right=drop_right,
        floor=floor,
        dtype="float32",
    )  # W[fix, pos]  ← p(correct)
    unknown_bits = math.log2(alphabet_size)
    entropies = np.sum((1.0 - weight_mat) * unknown_bits, axis=1)  # shape (L,)
    return cast(list[float], entropies.tolist())


# ---------------------------------------------------------------------------
# Optimal landing position
# ---------------------------------------------------------------------------


def optimal_fixation(
    word: str,
    drop_left: float,
    drop_right: float,
    *,
    alphabet_size: int = 26,
    floor: float = 0.0,
) -> int:
    """Index (0-based) of the fixation position that *minimises* ILP entropy."""
    entropies = ilp_entropy(
        word,
        drop_left,
        drop_right,
        alphabet_size=alphabet_size,
        floor=floor,
    )
    return int(np.argmin(entropies))

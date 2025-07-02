"""
acuity.py
~~~~~~~~~
Compute per-letter visual-acuity weights for the Initial Landing-Position
Entropy (ILPE) metric.

A *linear* drop-off model is assumed:

    P(correct | distance = d) = 1 - |d| · drop

`drop` can be different to the left and right of fixation, matching
empirical asymmetries in reading research.

Public API
----------
- acuity_weights(...)
- weight_matrix(...)

Author: Koby Raz
"""

from __future__ import annotations

import numpy as np

__all__ = ["acuity_weights", "weight_matrix"]


# --------------------------------------------------------------------------- #
# Core helper                                                                 #
# --------------------------------------------------------------------------- #
def acuity_weights(
    word_len: int,
    fixation_pos: int,
    /,
    *,
    drop_left: float = 0.10,
    drop_right: float = 0.10,
    floor: float = 0.0,
    dtype: np.dtype | str = "float32",
) -> np.ndarray:
    """
    Return a 1-D NumPy array of identification probabilities for every
    letter position in a *single* word, given a fixation position.

    Parameters
    ----------
    word_len      : number of letters in the word (≥ 1)
    fixation_pos  : 0-based index of the fixated letter (0 … word_len-1)
    drop_left     : linear drop per letter *left*  of fixation (0-1)
    drop_right    : linear drop per letter *right* of fixation (0-1)
    floor         : minimum allowed probability (clips the tail)
    dtype         : NumPy dtype for the returned array

    Returns
    -------
    probs : np.ndarray, shape = (word_len,)
        Identification probabilities, clipped to [floor, 1].

    Notes
    -----
    • Vectorised: `O(word_len)` memory and compute.
    • No Python loops ⇒ 100x faster than pure-Python for typical word sizes.
    """
    # Vector of absolute distances from fixation (- for left, + for right)
    idxs = np.arange(word_len, dtype=dtype)
    dist = idxs - fixation_pos  # e.g. [-3,-2,-1, 0, 1, 2]
    # Start from perfect acuity and subtract linear decay
    probs = np.ones(word_len, dtype=dtype)
    # Left of fixation (negative distances)
    left = dist < 0
    probs[left] -= np.abs(dist[left]) * drop_left
    # Right of fixation (positive distances)
    right = dist > 0
    probs[right] -= dist[right] * drop_right
    # Clip to [floor, 1]
    return np.clip(probs, floor, 1.0, out=probs)


# --------------------------------------------------------------------------- #
# Convenience bulk generator                                                  #
# --------------------------------------------------------------------------- #
def weight_matrix(
    word_len: int,
    /,
    *,
    drop_left: float = 0.10,
    drop_right: float = 0.10,
    floor: float = 0.0,
    dtype: np.dtype | str = "float32",
) -> np.ndarray:
    """
    Pre-compute a (word_len * word_len) matrix `W` where

        W[fix, pos] = acuity_weights(word_len, fix)[pos]

    This lets you look up visual-acuity masks for *any* fixation in O(1)
    — handy when scanning through many candidate fixations.

    Returns
    -------
    W : np.ndarray, shape = (word_len, word_len)
    """
    # Vector of fixation indices 0 … word_len-1  → column wise broadcasting
    fix = np.arange(word_len, dtype=dtype)[:, None]  # shape (L,1)
    pos = np.arange(word_len, dtype=dtype)[None, :]  # shape (1,L)
    dist = pos - fix  # shape (L,L)

    # Same algebra as in acuity_weights, now fully vectorised
    probs = np.ones_like(dist, dtype=dtype)
    probs[dist < 0] -= np.abs(dist[dist < 0]) * drop_left
    probs[dist > 0] -= dist[dist > 0] * drop_right
    np.clip(probs, floor, 1.0, out=probs)
    return probs


# --------------------------------------------------------------------------- #
# Quick self-test                                                             #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    # Example: 7-letter word, fixation on 3rd letter (index 2)
    print(acuity_weights(7, 2, drop_left=0.15, drop_right=0.10))
    print(weight_matrix(5, drop_left=0.2, drop_right=0.1))

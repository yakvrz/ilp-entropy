"""
This module provides functions to compute per-letter visual acuity weights,
which are central to the Initial Landing Position (ILP) entropy model.

The model assumes a linear drop-off in recognition probability with distance
from a fixation point, allowing for different rates to the left and right.
"""
from __future__ import annotations

import numpy as np

__all__ = ["acuity_weights", "weight_matrix"]


def acuity_weights(
    word_len: int,
    fixation_pos: int,
    *,
    drop_left: float = 0.10,
    drop_right: float = 0.10,
    floor: float = 0.0,
    dtype: np.dtype | str = "float32",
) -> np.ndarray:
    """
    Computes letter identification probabilities for a single fixation.

    Args:
        word_len: The number of letters in the word.
        fixation_pos: The 0-indexed fixation position.
        drop_left: The linear drop-off rate per letter to the left of fixation.
        drop_right: The linear drop-off rate per letter to the right of fixation.
        floor: The minimum allowed probability.
        dtype: The NumPy dtype for the returned array.

    Returns:
        A 1D NumPy array of shape (word_len,) containing the identification
        probabilities for each letter, clipped to [floor, 1.0].
    """
    positions = np.arange(word_len, dtype=dtype)
    distances = positions - fixation_pos

    probs = np.ones(word_len, dtype=dtype)
    
    left_mask = distances < 0
    probs[left_mask] -= np.abs(distances[left_mask]) * drop_left
    
    right_mask = distances > 0
    probs[right_mask] -= distances[right_mask] * drop_right
    
    return np.clip(probs, floor, 1.0, out=probs)


def weight_matrix(
    word_len: int,
    *,
    drop_left: float = 0.10,
    drop_right: float = 0.10,
    floor: float = 0.0,
    dtype: np.dtype | str = "float32",
) -> np.ndarray:
    """
    Computes a matrix of acuity weights for all possible fixation positions.

    The element W[i, j] in the returned matrix is the probability of
    recognizing the letter at position j when the fixation is at position i.

    Args:
        word_len: The number of letters in the word.
        drop_left: The linear drop-off rate per letter to the left of fixation.
        drop_right: The linear drop-off rate per letter to the right of fixation.
        floor: The minimum allowed probability.
        dtype: The NumPy dtype for the returned array.

    Returns:
        A 2D NumPy array of shape (word_len, word_len) with acuity weights.
    """
    fixation_indices = np.arange(word_len, dtype=dtype)[:, None]
    position_indices = np.arange(word_len, dtype=dtype)[None, :]
    distances = position_indices - fixation_indices

    probs = np.ones_like(distances, dtype=dtype)

    left_mask = distances < 0
    probs[left_mask] -= np.abs(distances[left_mask]) * drop_left

    right_mask = distances > 0
    probs[right_mask] -= distances[right_mask] * drop_right
    
    return np.clip(probs, floor, 1.0, out=probs)

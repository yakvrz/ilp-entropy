"""
This module computes the probability of observing each possible visibility
mask given a fixation position and visual acuity parameters.

The calculation assumes that the recognition of each letter is an
independent event, conditional on its distance from the fixation point.
"""
from __future__ import annotations

import numpy as np

from .acuity import weight_matrix

__all__ = [
    "get_mask_prob_matrix",
]


def get_mask_prob_matrix(
    word_len: int,
    masks_bits: np.ndarray,
    drop_left: float,
    drop_right: float,
    *,
    floor: float = 0.0,
    dtype: np.dtype | str = "float32",
) -> np.ndarray:
    """
    Computes a matrix of mask probabilities for all fixation positions.

    The element P[i, j] in the returned matrix is the probability of
    observing the visibility mask j when the fixation is at position i.

    Args:
        word_len: The number of letters in the word.
        masks_bits: A 2D boolean array of unpacked visibility masks.
        drop_left: The linear drop-off rate per letter to the left of fixation.
        drop_right: The linear drop-off rate per letter to the right of fixation.
        floor: The minimum allowed letter recognition probability.
        dtype: The NumPy dtype for the returned array.

    Returns:
        A 2D array of shape (word_len, 2**word_len) containing the
        probability of each mask for each fixation position.
    """
    W = weight_matrix(
        word_len,
        drop_left=drop_left,
        drop_right=drop_right,
        floor=floor,
        dtype=dtype,
    )

    # Broadcast W to compute probabilities for all masks at once
    # W_exp has shape (word_len, 1, word_len)
    # bits_exp has shape (1, 2**word_len, word_len)
    W_exp = W[:, None, :]
    bits_exp = masks_bits[None, :, :]

    # Calculate probability of each letter's visibility state for each mask
    prob_tensor = np.where(bits_exp, W_exp, 1.0 - W_exp)

    # Multiply probabilities across letters to get probability of each mask
    probs = np.prod(prob_tensor, axis=2, dtype=dtype)

    return probs.astype(dtype, copy=False)

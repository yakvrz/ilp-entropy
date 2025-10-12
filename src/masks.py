"""
This module provides utilities for creating and handling visibility masks,
which represent patterns of visible and occluded letters in a word.

Masks are represented as unsigned integers, where each bit corresponds to a
letter's visibility (1 for visible, 0 for occluded). This allows for
efficient, vectorized operations.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "bit_dtype",
    "enumerate_masks",
    "unpack_bits",
]


def bit_dtype(length: int) -> np.dtype[np.unsignedinteger]:
    """
    Selects the smallest unsigned integer dtype to hold 2**length values.
    
    Args:
        length: The number of bits required.

    Returns:
        The appropriate NumPy unsigned integer dtype (e.g., uint8, uint16).
    """
    if length <= 8:
        return np.dtype(np.uint8)
    if length <= 16:
        return np.dtype(np.uint16)
    if length <= 32:
        return np.dtype(np.uint32)
        return np.dtype(np.uint64)


def enumerate_masks(length: int) -> NDArray[np.unsignedinteger]:
    """
    Generates all 2**length possible visibility masks for a given length.

    Args:
        length: The word length, which determines the number of bits in each mask.

    Returns:
        A 1D NumPy array containing all possible integer masks, from 0 to 2**length - 1.
    """
    dtype = bit_dtype(length)
    return np.arange(2**length, dtype=dtype)


def unpack_bits(
    masks: NDArray[np.unsignedinteger],
    length: int,
    dtype: np.dtype[np.bool_] | str = "bool",
) -> NDArray[np.bool_]:
    """
    Unpacks integer masks into a 2D boolean array of their bit representations.
    This is a vectorized implementation that avoids explicit Python loops.

    Args:
        masks: A 1D array of integer masks.
        length: The number of bits (word length) to include in each unpacked row.
        dtype: The output NumPy dtype for the boolean array.

    Returns:
        A 2D boolean array where each row is the bit pattern of a mask.
    """
    # Use broadcasting to efficiently extract all bits at once.
    # The masks are shaped into a column vector (N, 1) and broadcast against
    # a row vector of bit shifts (1, L), resulting in an (N, L) matrix.
    powers_of_2 = np.arange(length, dtype=np.uint8)
    unpacked = ((masks[:, None] >> powers_of_2) & 1).astype(np.bool_)
    return unpacked.astype(dtype)

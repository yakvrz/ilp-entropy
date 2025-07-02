"""ilp_entropy.masks
~~~~~~~~~~~~~~~~~~~~~~

Mask enumeration & bit-trick utilities for the ILP-entropy package.

Every letter-visibility *mask* is represented as an **unsigned integer**
where bit *i* (least-significant = first letter) encodes whether the
letter at position *i* is visible (1) or occluded (0).

For a word of length *L* there are exactly ``2**L`` possible masks,
labelled ``0 … (1<<L)-1``.  Enumerating them once and operating with
bit-operations lets us replace Python loops / conditionals with pure
NumPy broadcast arithmetic.

Public API
----------
- enumerate_masks(...)
- unpack_bits(...)
- bit_dtype(...)
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


# --------------------------------------------------------------------------- #
# Helper: choose the minimal unsigned integer dtype that can hold 2**L masks  #
# --------------------------------------------------------------------------- #
def bit_dtype(length: int) -> np.dtype[np.unsignedinteger]:
    """Return the smallest unsigned integer dtype that can represent 2^length - 1."""
    if length <= 8:
        return np.dtype(np.uint8)
    elif length <= 16:
        return np.dtype(np.uint16)
    elif length <= 32:
        return np.dtype(np.uint32)
    else:
        return np.dtype(np.uint64)


# --------------------------------------------------------------------------- #
# Mask enumeration                                                            #
# --------------------------------------------------------------------------- #
def enumerate_masks(length: int) -> NDArray[np.unsignedinteger]:
    """Generate all possible visibility masks for a word of given length.

    Parameters
    ----------
    length : int
        Word length (number of letter positions)

    Returns
    -------
    NDArray[np.unsignedinteger]
        Array of all possible masks (2^length total)
        Each mask is a bit pattern where 1 = visible, 0 = invisible
    """
    dtype = bit_dtype(length)
    return np.arange(2**length, dtype=dtype)


# --------------------------------------------------------------------------- #
# Bit-unpacking                                                               #
# --------------------------------------------------------------------------- #
def unpack_bits(
    masks: NDArray[np.unsignedinteger],
    length: int,
    dtype: np.dtype[np.bool_] | str = "bool"
) -> NDArray[np.bool_]:
    """Unpack integer masks into boolean arrays.

    Parameters
    ----------
    masks : NDArray[np.unsignedinteger]
        Integer masks to unpack
    length : int
        Number of bits to unpack (word length)
    dtype : np.dtype[np.bool_] | str, default np.bool_
        Output array dtype

    Returns
    -------
    NDArray[np.bool_]
        Boolean array of shape (len(masks), length)
        Each row represents the bit pattern of one mask
    """
    # Use numpy's unpackbits for efficient bit extraction
    # We need to handle different integer sizes
    if masks.dtype == np.uint8:
        # For uint8, unpackbits works directly
        unpacked = np.unpackbits(masks.view(np.uint8), bitorder='little')
        return unpacked.reshape(-1, 8)[:, :length].astype(dtype)
    else:
        # For larger integers, we need to convert bit by bit
        result = np.zeros((len(masks), length), dtype=dtype)
        for i in range(length):
            result[:, i] = (masks >> i) & 1
        return result

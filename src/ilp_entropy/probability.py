"""ilp_entropy.probability
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Probability tables ``P(mask | fixation, dropL, dropR)`` built on top of
vectorised visual-acuity *weight_matrix* and integer bit-mask utilities.

These tables are the workhorse for later ILP-entropy computation: once a
table is cached to disk, looking up all mask probabilities for a given
fixation is an O(1) memory fetch.

Public API
----------
- mask_prob_matrix(...)
- get_mask_prob_matrix(...)

Author: Koby Raz
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final, cast

import numpy as np

from .acuity import weight_matrix
from .masks import enumerate_masks, unpack_bits

__all__ = [
    "mask_prob_matrix",
    "get_mask_prob_matrix",
]

DEFAULT_CACHE_DIR: Final[Path] = Path.home() / ".cache" / "ilp_entropy" / "prob_tables"
DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Core computation                                                            #
# --------------------------------------------------------------------------- #
def mask_prob_matrix(
    word_len: int,
    drop_left: float,
    drop_right: float,
    *,
    floor: float = 0.0,
    dtype: np.dtype | str = "float32",
) -> np.ndarray:
    """Return array ``P`` with shape ``(word_len, 2**word_len)``.

    ``P[fix, mask]`` = probability of observing **exact** visibility mask
    ``mask`` when the eyes land at fixation ``fix`` (0-based).

    The calculation assumes conditional independence of letter recognitions
    given distance from fixation – the same assumption used throughout the
    ILP literature and by our *acuity* helpers.
    """
    # Enumerate masks once – shape (M,)
    masks = enumerate_masks(word_len)
    # Boolean unpack: shape (M, word_len)
    bits = unpack_bits(masks, length=word_len, dtype=bool)

    # Pre-compute per-fixation per-letter recognition probs – shape (word_len, word_len)
    W = weight_matrix(
        word_len,
        drop_left=drop_left,
        drop_right=drop_right,
        floor=floor,
        dtype=dtype,
    )  # P(correct | fix,pos)

    # Broadcast:
    #   W[:, None, :] : (word_len,1,word_len)
    #   bits[None, :, :] : (1, num_masks, word_len)
    # Result → (word_len, num_masks, word_len); prod over last axis → (word_len, num_masks)
    W_exp = W[:, None, :]  # (word_len,1,word_len)
    bits_exp = bits[None, :, :]  # (1,num_masks,word_len)
    prob_tensor = np.where(bits_exp, W_exp, 1.0 - W_exp)
    probs = np.prod(prob_tensor, axis=2, dtype=dtype)  # (word_len,num_masks)
    return probs.astype(dtype, copy=False)


# --------------------------------------------------------------------------- #
# Disk cache helpers                                                          #
# --------------------------------------------------------------------------- #

def _cache_key(word_len: int, drop_left: float, drop_right: float, floor: float) -> str:
    """Stable filename-safe key based on parameters."""
    payload = json.dumps(
        {"L": word_len, "dl": drop_left, "dr": drop_right, "floor": floor},
        sort_keys=True,
    ).encode()
    return hashlib.sha1(payload).hexdigest()


def get_mask_prob_matrix(
    word_len: int,
    drop_left: float,
    drop_right: float,
    *,
    floor: float = 0.0,
    dtype: np.dtype | str = "float32",
    cache_dir: Path | None = None,
    allow_write: bool = True,
) -> np.ndarray:
    """Load probability matrix from cache or compute and optionally save.

    Parameters
    ----------
    word_len   : length of the word (4…11 in the target corpus)
    drop_left  : linear drop per letter to the left  of fixation
    drop_right : linear drop per letter to the right of fixation
    floor      : minimum recognition probability (clips tail)
    cache_dir  : directory for ``.npz`` files (defaults to ~/.cache/ilp_entropy)
    allow_write: if *False*, never saves new computations to disk

    Returns
    -------
    P : np.ndarray, shape = (word_len, 2**word_len), dtype = *dtype*
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = _cache_key(word_len, drop_left, drop_right, floor)
    path = cache_dir / f"P_L{word_len}_{key}.npz"

    if path.is_file():
        data = np.load(path, mmap_mode="r")  # read-only mem-map for low RAM
        arr = data["P"].astype(dtype, copy=False)
        return cast(np.ndarray, arr)

    # Cache miss – compute
    P = mask_prob_matrix(
        word_len,
        drop_left=drop_left,
        drop_right=drop_right,
        floor=floor,
        dtype=dtype,
    )

    if allow_write:
        np.savez_compressed(path, P=P.astype("float32", copy=False))  # ~8 MB for L=11

    return P

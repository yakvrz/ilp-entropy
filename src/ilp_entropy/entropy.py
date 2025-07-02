"""ilp_entropy.entropy
~~~~~~~~~~~~~~~~~~~~~~~
Full-spec Initial Landing-Position (ILP) entropy implementation.

This version follows the project road-map (§2.3-2.5):

*   Uses **bit-mask enumeration** for every visibility pattern.
*   Combines mask-level candidate-set entropies with mask probabilities
    pre-computed by :pyfunc:`ilp_entropy.probability.get_mask_prob_matrix`.
*   Works for any word length ≤ 11 and arbitrary `(drop_left, drop_right)`.

The module expects a *corpus index* providing, **per word length**:

    - ``codes``  - ``np.ndarray[int8]``   shape (N_L, L)
    - ``freqs``  - ``np.ndarray[float32]`` shape (N_L,)

The default implementation tries ``ilp_entropy.io.get_corpus_index`` but you
can pass your own via the public API.

Author: Koby Raz
"""
from __future__ import annotations

from typing import Mapping, cast

import numpy as np

from .masks import enumerate_masks, unpack_bits
from .probability import get_mask_prob_matrix

__all__ = [
    "position_entropy",
    "ilp_entropy",
    "optimal_fixation",
]

###############################################################################
# Corpus helpers                                                              #
###############################################################################

# --------------------------------------------------------------------------- #
# Optional import - allows the rest of the package to compile even if the     #
# corpus I/O layer is not implemented yet.                                    #
# --------------------------------------------------------------------------- #
try:
    # Expected to expose:  get_corpus_index(lang: str) -> CorpusIndex
    from .io import get_corpus_index  # type: ignore
except Exception:  # pragma: no cover - graceful degradation for dev installs
    get_corpus_index = None  # type: ignore


class CorpusIndex(Mapping[int, tuple[np.ndarray, np.ndarray]]):
    """Typed wrapper: ``ci[L] -> (codes, freqs)``."""

    def __getitem__(self, L: int) -> tuple[np.ndarray, np.ndarray]:  # noqa: D401
        raise NotImplementedError

    # ignore the remaining abstract-method clutter for brevity; Mapping[K,V]
    # provides default ``keys`` / ``__iter__`` / ``__len__`` via ``__getitem__``.


###############################################################################
# Utility functions                                                           #
###############################################################################

def _word_to_codes(word: str, /) -> np.ndarray:
    """Map *lower-case* ASCII letters → 0-25 codes (dtype = uint8)."""
    return (np.frombuffer(word.lower().encode("ascii"), dtype=np.uint8) - ord("a")).astype(
        "uint8",
    )


def _candidate_entropy_by_mask(
    word_codes: np.ndarray,  # shape (L,)
    masks_arr: np.ndarray,  # (M,) uintXX
    masks_bits: np.ndarray,  # (M,L) bool
    corpus_codes: np.ndarray,  # (N,L) uint8
    corpus_freqs: np.ndarray,  # (N,) float32
    *,
    eps: float = 1e-12,
) -> np.ndarray:  # (M,) float32
    """Return Shannon entropy **per mask** for a *single* target word.*"""

    N, L = corpus_codes.shape  # noqa: N806 (capital L is word length)
    masks_arr.shape[0]  # noqa: N806 (capital M = num masks)

    # Broadcast comparison: (N,1,L) vs (L,) → (N,1,L)
    eq = (corpus_codes[:, None, :] == word_codes[None, None, :])  # bool (N,1,L)
    vis = masks_bits[None, :, :]  # (1,M,L)

    # Accept if equal at visible positions *or* position is invisible.
    matches = np.all((~vis) | eq, axis=2)  # (N,M) bool

    # Fast weighted sums via matrix mult: each column j gives freqs of candidates
    freq_mat = matches.astype(np.float32) * corpus_freqs[:, None]  # (N,M)

    freq_sum = np.sum(freq_mat, axis=0)  # (M,)

    # Normalise and compute entropy; guard against log2(0).
    with np.errstate(divide="ignore"):
        probs_mat = freq_mat / (freq_sum + eps)  # broadcast divisor
        logp = np.log2(np.where(probs_mat > 0, probs_mat, 1.0))
    ent = -np.sum(probs_mat * logp, axis=0, dtype="float32")  # (M,)

    # Masks with *no* matching word (freq_sum==0) → entropy 0 by definition.
    ent = np.where(freq_sum > 0, ent, 0.0)
    return ent.astype("float32")


###############################################################################
# Public API                                                                  #
###############################################################################

def position_entropy(
    word: str,
    fixation_pos: int,
    drop_left: float,
    drop_right: float,
    *,
    corpus_index: CorpusIndex | None = None,
    language: str = "en",
) -> float:
    """Residual Shannon entropy (bits) for **one** landing position."""

    H_curve = ilp_entropy(
        word,
        drop_left,
        drop_right,
        corpus_index=corpus_index,
        language=language,
    )
    return float(H_curve[fixation_pos])


def ilp_entropy(
    word: str,
    drop_left: float,
    drop_right: float,
    *,
    corpus_index: CorpusIndex | None = None,
    language: str = "en",
) -> list[float]:
    """Return the ILP-entropy curve for *all* possible fixations in ``word``."""

    word_len = len(word)

    # ------------------------------------------------------------------ corpus
    if corpus_index is None:
        if get_corpus_index is None:
            raise RuntimeError(
                "No corpus index supplied and ilp_entropy.io.get_corpus_index "
                "is not available. Please pass `corpus_index=` explicitly.",
            )
        # Call get_corpus_index with default CSV path for the given language
        corpus_index = cast(CorpusIndex, get_corpus_index())

    try:
        corpus_codes, corpus_freqs = corpus_index[word_len]
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"Corpus index has no words of length {word_len}") from exc

    # ------------------------------------------------------------------- masks
    masks_arr = enumerate_masks(word_len)  # (M,)
    masks_bits = unpack_bits(masks_arr, length=word_len)  # (M,L) bool

    # ------------------------------------------------ candidate entropy (mask)
    word_codes = _word_to_codes(word)
    ent_by_mask = _candidate_entropy_by_mask(
        word_codes,
        masks_arr,
        masks_bits,
        corpus_codes,
        corpus_freqs,
    )  # (M,)

    # -------------------------------------------------------------- P(mask|fix)
    P_fix_mask = get_mask_prob_matrix(
        word_len=word_len,
        drop_left=drop_left,
        drop_right=drop_right,
    )  # shape (L,M)

    # -------------------------------------------------------------- final dot
    H_fix = np.dot(P_fix_mask, ent_by_mask).astype("float32")  # (L,)
    return cast(list[float], H_fix.tolist())


def optimal_fixation(
    word: str,
    drop_left: float,
    drop_right: float,
    *,
    corpus_index: CorpusIndex | None = None,
    language: str = "en",
) -> int:
    """Index (0-based) of the fixation that **minimises** ILP entropy."""

    H_curve = ilp_entropy(
        word,
        drop_left,
        drop_right,
        corpus_index=corpus_index,
        language=language,
    )
    return int(np.argmin(H_curve))

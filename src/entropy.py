"""
This module contains the core logic for calculating Initial Landing Position
(ILP) entropy, which measures the uncertainty in word recognition based on
a given fixation point.
"""
from __future__ import annotations

from typing import cast

import numpy as np
from numba import jit

from .masks import enumerate_masks, unpack_bits
from .probability import get_mask_prob_matrix

__all__ = [
    "ilp_entropy",
]


@jit(nopython=True, cache=True)
def _candidate_entropy_by_mask(
    word_codes: np.ndarray,
    masks_bits: np.ndarray,
    corpus_codes: np.ndarray,
    corpus_freqs: np.ndarray,
) -> np.ndarray:
    """
    Calculates the entropy for the set of candidate words for each mask.
    This Numba-jitted function is optimized for performance. It iterates over
    masks and uses an explicit loop to check for matching corpus words, as
    Numba does not support `np.all` with an `axis` argument.
    """
    n_masks = masks_bits.shape[0]
    n_corpus_words = corpus_codes.shape[0]
    entropies = np.zeros(n_masks, dtype=np.float32)

    for i in range(n_masks):
        mask_bit = masks_bits[i]
        visible_indices = np.where(mask_bit)[0]

        # Create a list of candidate frequencies. Numba works well with lists.
        candidate_freqs_list = []
        if visible_indices.size > 0:
            # Numba doesn't support np.all(axis=...). We must loop manually.
            # This gets compiled to fast machine code.
            for k in range(n_corpus_words):
                is_match = True
                # Check if this corpus word matches the visible letters
                for j in visible_indices:
                    if corpus_codes[k, j] != word_codes[j]:
                        is_match = False
                        break
                if is_match:
                    candidate_freqs_list.append(corpus_freqs[k])

            if len(candidate_freqs_list) > 0:
                candidate_freqs = np.array(candidate_freqs_list, dtype=np.float32)
            else:
                candidate_freqs = np.empty(0, dtype=np.float32)
        else:
            # If mask is all zeros, all words are candidates
            candidate_freqs = corpus_freqs

        total_freq = candidate_freqs.sum()
        if total_freq > 0:
            probs = candidate_freqs / total_freq
            # Loop for entropy calculation is robust and fast in Numba
            h = 0.0
            for p in probs:
                if p > 0:
                    h -= p * np.log2(p)
            entropies[i] = h

    return entropies


def ilp_entropy(
    word: str,
    drop_left: float,
    drop_right: float,
    *,
    corpus: dict[int, tuple[np.ndarray, np.ndarray]],
    mask_cache: dict[int, np.ndarray],
) -> list[float]:
    """
    Calculates the ILP entropy curve for a single word.

    This function computes the word recognition uncertainty (entropy) for every
    possible fixation position within the given word, using a pre-computed
    corpus index and a cache of visibility masks.

    Args:
        word: The target word for which to calculate entropy.
        drop_left: The linear acuity drop-off rate to the left of fixation.
        drop_right: The linear acuity drop-off rate to the right of fixation.
        corpus: A pre-loaded and structured corpus index.
        mask_cache: A dictionary caching the bit-unpacked visibility masks for word lengths.

    Returns:
        A list of floats representing the entropy at each fixation position.
    """
    word_len = len(word)

    # 1. Get pre-computed data for this word length from the corpus and mask caches.
    try:
        corpus_codes, corpus_freqs = corpus[word_len]
        masks_bits = mask_cache[word_len]
    except KeyError as exc:
        raise ValueError(
            f"Corpus or mask cache missing data for length {word_len}"
        ) from exc

    # 2. Calculate the entropy for each possible visibility mask.
    word_codes = np.fromiter((ord(c) - 97 for c in word), dtype=np.uint8, count=word_len)
    ent_by_mask = _candidate_entropy_by_mask(
        word_codes,
        masks_bits,
        corpus_codes,
        corpus_freqs,
    )

    # 3. Get the probability of each mask occurring for each fixation point.
    P_fix_mask = get_mask_prob_matrix(
        word_len=word_len,
        masks_bits=masks_bits,
        drop_left=drop_left,
        drop_right=drop_right,
    )

    # 4. The final entropy for each fixation is the weighted average (dot product)
    #    of the mask probabilities and the entropy of the candidates for each mask.
    H_fix = np.dot(P_fix_mask, ent_by_mask).astype("float32")
    return cast(list[float], H_fix.tolist())

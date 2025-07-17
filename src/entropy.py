"""
This module contains the core logic for calculating Initial Landing Position
(ILP) entropy, which measures the uncertainty in word recognition based on
a given fixation point.
"""
from __future__ import annotations

from typing import cast

import numpy as np

from .io import get_corpus_index
from .masks import enumerate_masks, unpack_bits
from .probability import get_mask_prob_matrix

__all__ = [
    "ilp_entropy",
]


def _word_to_codes(word: str) -> np.ndarray:
    """Converts a word into a NumPy array of integer codes (a=0, b=1, ...)."""
    return np.fromiter((ord(c) - 97 for c in word), dtype=np.uint8, count=len(word))


def _candidate_entropy_by_mask(
    word_codes: np.ndarray,
    masks_bits: np.ndarray,
    corpus_codes: np.ndarray,
    corpus_freqs: np.ndarray,
) -> np.ndarray:
    """
    Calculates the entropy for the set of candidate words for each mask.

    For each visibility mask, this function identifies all words in the corpus
    that match the visible letters of the target word and computes the
    Shannon entropy of that set based on their frequencies.
    """
    n_masks = len(masks_bits)
    entropies = np.zeros(n_masks, dtype="float32")

    for i in range(n_masks):
        mask_bit = masks_bits[i]
        visible_indices = np.where(mask_bit)[0]

        if visible_indices.size > 0:
            matches = np.all(
                corpus_codes[:, visible_indices] == word_codes[visible_indices],
                axis=1,
            )
            candidate_freqs = corpus_freqs[matches]
        else:
            # If mask is all zeros, all words are candidates
            candidate_freqs = corpus_freqs

        total_freq = candidate_freqs.sum()
        if total_freq > 0:
            probs = candidate_freqs / total_freq
            # Calculate entropy, avoiding log(0) for zero-probability candidates.
            non_zero_probs = probs[probs > 0]
            entropies[i] = -np.sum(non_zero_probs * np.log2(non_zero_probs))

    return entropies


def ilp_entropy(
    word: str,
    drop_left: float,
    drop_right: float,
    *,
    corpus: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
    corpus_path: str | None = None,
    min_freq: float = 1e-7,
) -> list[float]:
    """
    Calculates the ILP entropy curve for a single word.

    This function computes the word recognition uncertainty (entropy) for every
    possible fixation position within the given word.

    Args:
        word: The target word for which to calculate entropy.
        drop_left: The linear acuity drop-off rate to the left of fixation.
        drop_right: The linear acuity drop-off rate to the right of fixation.
        corpus: An optional pre-loaded corpus index. If None, corpus_path is used.
        corpus_path: Path to the corpus CSV file, used if corpus is not provided.
        min_freq: The minimum frequency for a word to be included in the corpus.

    Returns:
        A list of floats representing the entropy at each fixation position.
    """
    word_len = len(word)

    if corpus is None:
        if corpus_path is None:
            raise ValueError("Either 'corpus' or 'corpus_path' must be provided.")
        corpus = get_corpus_index(
            word_lengths=[word_len], csv_path=corpus_path, min_freq=min_freq
        )

    try:
        corpus_codes, corpus_freqs = corpus[word_len]
    except KeyError as exc:
        raise ValueError(f"Corpus has no words of length {word_len}") from exc

    masks_arr = enumerate_masks(word_len)
    masks_bits = unpack_bits(masks_arr, length=word_len)

    word_codes = _word_to_codes(word)
    ent_by_mask = _candidate_entropy_by_mask(
        word_codes,
        masks_bits,
        corpus_codes,
        corpus_freqs,
    )

    P_fix_mask = get_mask_prob_matrix(
        word_len=word_len,
        drop_left=drop_left,
        drop_right=drop_right,
    )

    # The final entropy for each fixation is the dot product of the
    # mask probabilities and the entropy of the candidates for each mask.
    H_fix = np.dot(P_fix_mask, ent_by_mask).astype("float32")
    return cast(list[float], H_fix.tolist())

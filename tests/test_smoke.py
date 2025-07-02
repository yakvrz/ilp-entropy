# tests/test_smoke.py
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ilp_entropy.acuity import acuity_weights, weight_matrix
from ilp_entropy.masks import enumerate_masks, unpack_bits
from ilp_entropy.probability import get_mask_prob_matrix
from ilp_entropy.entropy import position_entropy, ilp_entropy, optimal_fixation
from ilp_entropy.io import get_corpus_index


@pytest.fixture(scope="session")
def tiny_corpus():
    """Load just the 4-letter slice of the corpus—fast and deterministic."""
    codes, freqs = get_corpus_index(word_lengths=[4])[4]
    # take first 30 words to keep the test snappy
    return codes[:30], freqs[:30]


def test_acuity_helpers():
    w = acuity_weights(7, fixation_pos=3, drop_left=0.15, drop_right=0.10)
    assert w.shape == (7,) and np.all((0.0 <= w) & (w <= 1.0))

    W = weight_matrix(5, drop_left=0.2, drop_right=0.1)
    assert W.shape == (5, 5)
    assert np.all((0.0 <= W) & (W <= 1.0))


def test_mask_helpers():
    m = enumerate_masks(5)
    assert len(m) == 2 ** 5
    bits = unpack_bits(m, length=5)
    # bits should be bool matrix (32,5) with unique rows
    assert bits.dtype == bool and bits.shape == (32, 5)
    assert np.unique(bits, axis=0).shape[0] == 32


def test_probability_helpers():
    P = get_mask_prob_matrix(word_len=5, drop_left=0.2, drop_right=0.1)
    # shape = (fixations, masks)
    assert P.shape == (5, 2 ** 5)
    assert np.all((0.0 <= P) & (P <= 1.0))
    # rows should sum to 1
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)


def test_entropy_end_to_end(tiny_corpus):
    codes, _ = tiny_corpus
    sample_word = "".join(chr(97 + c) for c in codes[0])  # first word as str

    # single-fixation
    e_pos = position_entropy(sample_word, 1, 0.15, 0.10)
    assert np.isfinite(e_pos) and e_pos >= 0

    # full curve
    curve = ilp_entropy(sample_word, 0.15, 0.10)
    assert len(curve) == len(sample_word)
    curve_array = np.array(curve)
    assert np.all(np.isfinite(curve_array)) and np.all(curve_array >= 0)

    # optimum
    opt = optimal_fixation(sample_word, 0.15, 0.10)
    assert 0 <= opt < len(sample_word)

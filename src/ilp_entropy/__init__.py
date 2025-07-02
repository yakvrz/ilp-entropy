"""Initial Landing Position (ILP) Entropy Package

A Python package for computing Initial Landing Position Entropy, which quantifies
uncertainty in word recognition for each possible fixation position within a word
during reading.

Key Features:
- Efficient bit-mask based computation
- Separate left/right visual acuity drop parameters
- Vectorized NumPy operations for performance
- Disk caching of probability matrices
- Support for word lengths up to 11 letters

Basic Usage:
    >>> from ilp_entropy import ilp_entropy, position_entropy, optimal_fixation
    >>>
    >>> # Compute entropy curve for all fixation positions
    >>> curve = ilp_entropy("word", drop_left=0.15, drop_right=0.10)
    >>>
    >>> # Compute entropy for specific fixation position
    >>> entropy = position_entropy("word", fixation_pos=1, drop_left=0.15, drop_right=0.10)
    >>>
    >>> # Find optimal fixation position
    >>> opt_pos = optimal_fixation("word", drop_left=0.15, drop_right=0.10)

Author: Koby Raz
"""

from __future__ import annotations

# Visual acuity functions
from .acuity import acuity_weights, weight_matrix

# Configuration and validation
from .config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CORPUS_PATH,
    DEFAULT_DROP_LEFT,
    DEFAULT_DROP_RIGHT,
    DEFAULT_MIN_FREQ,
    DEFAULT_WORD_LENGTHS,
    MAX_WORD_LENGTH,
    MIN_WORD_LENGTH,
    validate_drop_rate,
    validate_parameters,
    validate_word,
    validate_word_length,
    validate_word_lengths,
)

# Core entropy computation
from .entropy import ilp_entropy, optimal_fixation, position_entropy

# Data loading
from .io import get_corpus_index, get_top_words

# Mask operations
from .masks import bit_dtype, enumerate_masks, unpack_bits

# Probability matrices
from .probability import get_mask_prob_matrix, mask_prob_matrix

__version__ = "0.1.0"

__all__ = [
    # Core API
    "ilp_entropy",
    "position_entropy",
    "optimal_fixation",

    # Visual acuity
    "acuity_weights",
    "weight_matrix",

    # Masks
    "enumerate_masks",
    "unpack_bits",
    "bit_dtype",

    # Probability matrices
    "mask_prob_matrix",
    "get_mask_prob_matrix",

    # Data loading
    "get_corpus_index",
    "get_top_words",

    # Configuration
    "DEFAULT_DROP_LEFT",
    "DEFAULT_DROP_RIGHT",
    "DEFAULT_WORD_LENGTHS",
    "DEFAULT_MIN_FREQ",
    "DEFAULT_CORPUS_PATH",
    "DEFAULT_CACHE_DIR",
    "MAX_WORD_LENGTH",
    "MIN_WORD_LENGTH",

    # Validation
    "validate_drop_rate",
    "validate_word",
    "validate_word_length",
    "validate_word_lengths",
    "validate_parameters",

    # Metadata
    "__version__",
]

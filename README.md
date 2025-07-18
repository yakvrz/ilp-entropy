# ILP Entropy Project

## Overview

This project provides a Python implementation for calculating Initial Landing Position (ILP) Entropy. ILP Entropy is a measure from cognitive science and reading research that quantifies the uncertainty (in bits) about a word's identity given a specific eye fixation point.

The core idea is to model how visual acuity drops off with distance from the fixation point and to calculate the resulting uncertainty based on a corpus of words.

## Project Structure

```
ilp-entropy/
├── data/
│   ├── opensubtitles_en.csv   # Corpus data
│   └── wikipedia_en.csv       # Additional corpus data
├── scripts/
│   └── main.py                # Main executable script for calculations
├── src/
│   ├── entropy.py             # Core entropy calculation logic
│   ├── io.py                  # Data loading and saving
│   ├── masks.py               # Visibility mask generation
│   └── probability.py         # Probability calculation helpers
├── visualization/
│   └── plot_results.py        # Script for plotting results
├── run_simple.sh              # Convenience script for a single run
├── run_sweep.sh               # Convenience script for a parameter sweep
├── word_list.txt              # A default list of words to process
└── README.md
```

## Core Calculation Pipeline

The calculation proceeds through the following steps:

1.  **Corpus Preprocessing**: The corpus is loaded, sanitized (lowercase, alphabetic words only), and filtered by a minimum frequency threshold. It is then indexed by word length for efficient lookups. (`src/io.py`)
2.  **Visual Acuity Modeling**: For a given word and fixation point, we model the probability of correctly identifying each letter based on a linear acuity drop-off. (`src/acuity.py`)
3.  **Mask Enumeration**: We generate all `2^L` possible visibility "masks" for a word of length `L`. (`src/masks.py`)
4.  **Mask Probability Calculation**: We calculate the probability of each mask occurring based on the letter acuity weights from the previous step. (`src/probability.py`)
5.  **Candidate Set Entropy**: For each mask, we find all corpus words consistent with the visible letters and compute the Shannon entropy of this "candidate set." (`src/entropy.py`)
6.  **Final ILP Entropy**: The final entropy for a fixation point is the weighted average of the candidate set entropies, using the mask probabilities as weights. This produces an entropy curve across the word. (`src/entropy.py`)

## Performance and Optimization

The script is designed for high performance on large datasets and is optimized in several ways:

*   **Parallel Processing**: It uses Python's `concurrent.futures.ProcessPoolExecutor` to run calculations in parallel, automatically leveraging all available CPU cores.
*   **Efficient I/O**: The corpus file is read into memory only once. All subsequent operations (filtering, indexing, etc.) are performed on the in-memory representation to minimize disk access.
*   **Just-In-Time (JIT) Compilation**: The most computationally intensive part of the entropy calculation is compiled to highly optimized machine code on its first run using the Numba library. Subsequent calls to this function are significantly faster.
*   **Pre-computation and Caching**: All data required by the worker processes, such as the corpus index and the bitwise visibility masks, are pre-calculated and cached before the main parallel computation begins. This avoids redundant work inside the main loop.

## Usage

The primary way to run calculations is via the `scripts/main.py` script. For convenience, two shell scripts are provided in the project root to execute common tasks.

You may need to make the scripts executable first:
```bash
chmod +x run_simple.sh
chmod +x run_sweep.sh
```

### Convenience Scripts

#### Single Run (`run_simple.sh`)
Executes a single calculation with a fixed set of parameters (`drop_left=0.1`, `drop_right=0.2`) for words of length 4-10. It saves the results and a `plot.png` visualization to a new timestamped directory in `output/`.
```bash
./run_simple.sh
```

#### Parameter Sweep (`run_sweep.sh`)
Executes a full parameter sweep over a grid of `drop_left` and `drop_right` values (from 0.1 to 0.9) for words of length 4-10. It saves the results and a `plot.png` visualization to a new timestamped directory in `output/`.
```bash
./run_sweep.sh
```

### Manual Execution

You can also call the Python script directly to run custom calculations.

**Example: Single Run for a Word List**
```bash
python scripts/main.py \
    --corpus-file data/opensubtitles_en.csv \
    --word-list word_list.txt \
    --min-freq 1e-6 \
    --drop-left 0.1 \
    --drop-right 0.2
```

**Example: Custom Sweep**
```bash
python scripts/main.py \
    --corpus-file data/opensubtitles_en.csv \
    --all-corpus-words \
    --min-freq 1e-6 \
    --sweep-left '0.0,0.5,0.1' \
    --sweep-right '0.0,0.5,0.1'
```

### All Arguments

*   `--corpus-file`: (Required) Path to the corpus file.
*   `--min-freq`: (Required) Minimum word frequency to include.
*   `--word-list` or `--all-corpus-words`: (Required) Specify either a file with a list of words or the flag to use all words from the corpus.
*   `--drop-left` or `--sweep-left`: (Required) Specify either a single value for the left-side acuity drop-off or a sweep range in the format `'START,END,STEP'`.
*   `--drop-right` or `--sweep-right`: (Required) Specify either a single value for the right-side acuity drop-off or a sweep range.
*   `--word-lengths`: (Optional) Filter words by specific lengths.
*   `--workers`: (Optional) Number of parallel processes to use. Defaults to the number of available CPU cores.
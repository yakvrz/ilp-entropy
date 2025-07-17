# ILP Entropy Project Architecture

## Overview

This project provides a Python implementation for calculating Initial Landing Position (ILP) Entropy. ILP Entropy is a measure from cognitive science and reading research that quantifies the uncertainty (in bits) about a word's identity given a specific eye fixation point.

The core idea is to model how visual acuity drops off with distance from the fixation point and to calculate the resulting uncertainty based on a corpus of words.

## Core Calculation Pipeline

The calculation proceeds through the following steps:

1.  **Corpus Preprocessing**: Before any calculations, the corpus is loaded and sanitized. This involves converting all words to lowercase, filtering to keep only strictly alphabetic words (`^[a-z]+$`), and removing words rarer than a specified frequency threshold (`--min-freq`). The filtered corpus is then indexed into a dictionary by word length for efficient retrieval. (`io.py`)

2.  **Visual Acuity Modeling**: For a given word and a fixation position, we first model the probability of correctly identifying each letter. This is based on a linear drop-off in acuity with distance from fixation, with potentially different rates for the left and right sides. (`acuity.py`)

3.  **Mask Enumeration**: To handle all possible patterns of letter visibility, we generate `2^L` "masks" for a word of length `L`. Each mask is a binary pattern representing which letters are seen and which are not. (`masks.py`)

4.  **Mask Probability Calculation**: For each fixation position, we calculate the probability of every possible visibility mask occurring. This depends on the letter acuity weights from step 2. (`probability.py`)

5.  **Candidate Set Entropy**: For each mask, we identify all words in the pre-processed reference corpus that are consistent with the visible letters of the target word. We then compute the Shannon entropy of this "candidate set" based on the frequencies of the matching words. (`entropy.py`)

6.  **Final ILP Entropy**: The final ILP entropy for a given fixation is the weighted average of the candidate set entropies, where the weights are the mask probabilities from step 4. This gives an entropy value for each possible fixation point in the word. (`entropy.py`)

## Project Structure

The project contains the following key components:

```
ilp-entropy/
├── data/
│   ├── opensubtitles.csv
│   └── words.txt
├── scripts/
│   └── run_entropy.py
├── src/
│   ├── acuity.py
│   ├── entropy.py
│   ├── io.py
│   ├── masks.py
│   └── probability.py
├── config.json
├── .gitignore
├── README.md
└── requirements.txt
```

## Usage

The primary entry point is the `scripts/run_entropy.py` script, which offers a flexible command-line interface for running entropy calculations.

### Configuration

A `config.json` file in the root directory holds the default parameters for the script, including file paths and calculation settings. All parameters in this file can be overridden by command-line arguments.

### Execution

The script can be run to process a list of words from a file, or to process all words from the corpus that meet the minimum frequency requirement. The calculations are run in parallel by default to maximize performance.

**Example Usage:**

*   **Run with all defaults from `config.json`:**
    ```bash
    python scripts/run_entropy.py
    ```

*   **Run for a specific word list and save the output:**
    ```bash
    python scripts/run_entropy.py --word-list data/my_words.txt --output-file results.csv
    ```

*   **Run for all words in the corpus using 4 worker processes:**
    ```bash
    python scripts/run_entropy.py --all-corpus-words --workers 4
    ```
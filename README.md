# ILP Entropy Calculator

This project provides a command-line tool to calculate the Initial Landing Position (ILP) entropy for a given set of words based on a text corpus. It is designed for researchers in psycholinguistics and related fields to analyze how information is distributed across words.

The calculation simulates reading by applying a linear acuity drop-off from a fixation point, measures the resulting uncertainty (entropy) at each possible landing position within a word, and uses a corpus of unigram frequencies to determine word probabilities.

The tool is configurable, supports parallel processing for efficiency, and allows for parameter sweeps to explore the effects of different acuity drop-off rates.

## Project Structure

```
ilp-entropy/
├── ARCHITECTURE.md         # Detailed explanation of the calculation pipeline
├── data/
│   ├── opensubtitles.csv   # All data files live here
│   └── words.txt
├── scripts/
│   └── run_entropy.py
├── src/
│   ├── ... (source files)
├── config.json
├── README.md
└── requirements.txt        # Python dependencies
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd ilp-entropy
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Corpus Data:**
    Place your corpus file (e.g., `opensubtitles.csv`) in the `data/` directory. The expected format is a CSV with `word` and `freq` columns.

### Corpus Preprocessing

Before any calculations, the corpus undergoes a critical sanitization and filtering process to ensure data quality and efficiency:

1.  **Lowercase Conversion**: All words from the corpus are converted to lowercase. This ensures that words like "The" and "the" are treated as identical.
2.  **Alphabetic Filtering**: Words are filtered using the regular expression `^[a-z]+$`. This removes any words containing numbers, hyphens, apostrophes, or any other non-alphabetic characters.
3.  **Minimum Frequency Filtering**: Words with a frequency below the `--min-freq` threshold (default `1e-7`) are removed from the corpus to exclude very rare words.

This pre-processed and indexed version of the corpus is then used for all subsequent entropy calculations.

## Usage

The primary entry point is `
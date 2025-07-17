# ILP Entropy Calculator

This project provides a command-line tool to calculate the Initial Landing Position (ILP) entropy for a given set of words based on a text corpus. It is designed for researchers in psycholinguistics and related fields to analyze how information is distributed across words.

The calculation simulates reading by applying a linear acuity drop-off from a fixation point, measures the resulting uncertainty (entropy) at each possible landing position within a word, and uses a corpus of unigram frequencies to determine word probabilities.

The tool is configurable, supports parallel processing for efficiency, and allows for parameter sweeps to explore the effects of different acuity drop-off rates.

## Project Structure

```
ilp-entropy/
├── ARCHITECTURE.md         # Detailed explanation of the calculation pipeline and theory
├── data/                   # Directory for input data files
│   ├── opensubtitles.csv   # Example corpus file with word frequencies
│   └── words.txt           # Example word list for targeted calculations
├── scripts/
│   └── run_entropy.py      # Main entry point for running entropy calculations
├── src/                    # Source code for the ILP entropy calculation logic
│   ├── acuity.py           # Models visual acuity drop-off
│   ├── entropy.py          # Core entropy calculation logic
│   ├── io.py               # Handles data loading and corpus processing
│   ├── masks.py            # Generates visibility masks for letters
│   └── probability.py      # Calculates probabilities of visibility masks
├── config.json             # Configuration file for default parameters
├── README.md               # This file
└── requirements.txt        # Python dependencies for the project
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

The primary entry point is `scripts/run_entropy.py`. All parameters can be configured in `config.json` or overridden with command-line arguments.

### Configuration

The `config.json` file holds the default parameters for the script.

```json
{
  "corpus_file": "data/opensubtitles.csv",
  "word_list": "data/words.txt",
  "drop_left": 0.1,
  "drop_right": 0.2,
  "min_freq": 1e-7,
  "output_file": "ilp_entropy_results.csv"
}
```

### Basic Calculation

To run the calculation for a predefined list of words:

```bash
python scripts/run_entropy.py --word-list data/words.txt --output-file results.csv
```

### Running on All Corpus Words

To process every word in the corpus that meets the frequency and length criteria (instead of using a word list):

```bash
python scripts/run_entropy.py --all-corpus-words --word-lengths 5 6 7 --output-file long_words_entropy.csv
```

### Parallel Execution

The script automatically uses all available CPU cores for processing. You can specify the number of workers:

```bash
python scripts/run_entropy.py --all-corpus-words --workers 4
```

### Parameter Sweeps

The tool can run a grid search over different `drop_left` and `drop_right` acuity parameters. This is useful for sensitivity analysis.

The sweep range is defined as a string: `"START,END,STEP"`.

**Example:**
This command will test `drop_left` values of `0.0`, `0.1`, and `0.2` against `drop_right` values of `0.0`, `0.1`, and `0.2` for the words in `data/words.txt`.

```bash
python scripts/run_entropy.py \
    --word-list data/words.txt \
    --sweep-left "0.0,0.2,0.1" \
    --sweep-right "0.0,0.2,0.1" \
    --output-file sweep_results.csv
```

The output CSV will contain results for every combination of parameters, allowing for easy analysis of how acuity rates affect ILP entropy.
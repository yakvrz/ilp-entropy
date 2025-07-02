# ILP Entropy

A Python package for calculating Initial Landing Position (ILP) Entropy - a measure of uncertainty in word recognition for each fixation position within a word.

## Overview

This package implements the ILP entropy calculation that quantifies how much uncertainty exists in word recognition when fixating at different positions within a word. The calculation takes into account:

- **Visual acuity drop**: Separate parameters for left and right visual field degradation
- **Word frequency**: Based on corpus statistics
- **Letter position**: Entropy calculated for each possible fixation position

## Installation

### Development Installation

```bash
# Clone the repository
git clone <repository-url>
cd ilp-entropy

# Install dependencies with Poetry
poetry install

# Or install in development mode with pip
pip install -e .
```

### Dependencies

- Python 3.8+
- NumPy
- Pandas
- Click (for CLI)

## Usage

### Command Line Interface

The package provides several CLI commands for different use cases:

#### Calculate entropy for individual words

```bash
# Single word
python -m ilp_entropy words "hello"

# Multiple words
python -m ilp_entropy words "hello" "world" "entropy"

# With custom parameters
python -m ilp_entropy words "hello" --drop-left 0.3 --drop-right 0.4
```

#### Process words from a file

```bash
# Process all words in a file
python -m ilp_entropy file words.txt

# Get entropy at every position (not just optimal)
python -m ilp_entropy file words.txt --no-optimal-only
```

#### Parameter sweep analysis

```bash
# Sweep over multiple parameter combinations
python -m ilp_entropy sweep words.txt \
  --drop-left-range 0.1 0.5 0.1 \
  --drop-right-range 0.1 0.5 0.1
```

#### Top frequency words

```bash
# Process top 1000 most frequent words
python -m ilp_entropy top-words 1000

# With parameter sweep
python -m ilp_entropy top-words 100 \
  --drop-left-range 0.2 0.4 0.1 \
  --drop-right-range 0.2 0.4 0.1
```

### Python API

```python
from ilp_entropy import calculate_word_entropy, get_top_words

# Calculate entropy for a single word
entropy_df = calculate_word_entropy("hello", drop_left=0.3, drop_right=0.4)
print(entropy_df)

# Get top frequent words
top_words = get_top_words(1000)

# Batch processing
from ilp_entropy.runner import process_words_batch

results = process_words_batch(
    words=["hello", "world"],
    drop_left_values=[0.2, 0.3],
    drop_right_values=[0.2, 0.3],
    max_workers=4
)
```

## Output Format

The entropy calculations produce CSV output with the following columns:

- `word`: The input word
- `position`: Fixation position (1-indexed)
- `entropy`: Calculated entropy value
- `drop_left`: Left visual field drop parameter
- `drop_right`: Right visual field drop parameter

Example output:
```
word,position,entropy,drop_left,drop_right
hello,1,2.145,0.3,0.4
hello,2,1.876,0.3,0.4
hello,3,1.654,0.3,0.4
hello,4,1.789,0.3,0.4
hello,5,2.001,0.3,0.4
```

## Testing

Run the test suite:

```bash
# Unit tests
poetry run pytest tests/test_smoke.py -v

# Integration tests
poetry run python tests/test_integration.py

# Performance tests (medium scale)
poetry run python tests/test_performance.py

# Performance tests (full 1000 words - slow)
poetry run python tests/test_performance.py --full

# All fast tests
poetry run pytest -v

# All tests including slow ones
poetry run pytest -v -m "slow"

# Skip slow tests
poetry run pytest -v -m "not slow"
```

## Project Structure

```
ilp-entropy/
├── src/ilp_entropy/          # Main package
│   ├── __init__.py          # Public API exports
│   ├── __main__.py          # CLI entry point
│   ├── acuity.py            # Visual acuity calculations
│   ├── cli.py               # Command-line interface
│   ├── config.py            # Configuration and constants
│   ├── entropy.py           # Core entropy calculations
│   ├── io.py                # Data loading and I/O
│   ├── masks.py             # Visual field masks
│   ├── probability.py       # Probability calculations
│   └── runner.py            # Batch processing and parallel execution
├── tests/                   # Test suite
│   ├── test_smoke.py        # Unit tests
│   └── test_integration.py  # Integration tests
├── data/                    # Corpus data files
├── ARCHITECTURE.md          # Detailed architecture documentation
└── reference_pipeline.R     # Original R implementation
```

## Algorithm Details

The ILP entropy calculation follows these steps:

1. **Data Loading**: Load word frequency corpus and filter for alphabetic words
2. **Mask Generation**: Create visual field masks based on acuity drop parameters
3. **Probability Calculation**: Compute recognition probabilities for each position
4. **Entropy Calculation**: Calculate Shannon entropy across all possible words

For detailed algorithm documentation, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Configuration

Default parameters can be modified in `src/ilp_entropy/config.py`:

- `DEFAULT_DROP_LEFT = 0.3`: Default left visual field drop
- `DEFAULT_DROP_RIGHT = 0.3`: Default right visual field drop  
- `MIN_WORD_LENGTH = 4`: Minimum word length to process
- `MAX_WORD_LENGTH = 15`: Maximum word length to process

## License

[License information to be added]

## Contributing

[Contributing guidelines to be added]

## Citation

[Citation information to be added]

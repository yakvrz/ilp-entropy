# ILP Entropy

A Python package for calculating Initial Landing Position (ILP) Entropy - a measure of uncertainty in word recognition for each fixation position within a word.

## Overview

This package implements the ILP entropy calculation that quantifies how much uncertainty exists in word recognition when fixating at different positions within a word. The calculation takes into account:

- **Visual acuity drop**: Separate parameters for left and right visual field degradation
- **Word frequency**: Based on corpus statistics
- **Letter position**: Entropy calculated for each possible fixation position

## Key Features

- **Transparent Progress Reporting**: Real-time progress bars with ETA calculations for all operations
- **High Performance**: Intelligent workload detection and parallel processing optimizations
- **Comprehensive CLI**: Full command-line interface with multiple output formats
- **Flexible API**: Both single-word and batch processing capabilities
- **Robust Testing**: Comprehensive test suite with performance benchmarks

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

- Python 3.9+
- NumPy
- Pandas
- Click (for CLI)

## Usage

### Command Line Interface

The package provides several CLI commands with automatic progress reporting:

#### Calculate entropy for individual words

```bash
# Single word
python -m ilp_entropy words "hello"

# Multiple words with progress bar
python -m ilp_entropy words "hello" "world" "entropy"

# With custom parameters
python -m ilp_entropy words "hello" --drop-left 0.3 --drop-right 0.4

# Save to file
python -m ilp_entropy words "hello" --output results.csv
```

#### Process words from a file

```bash
# Process all words in a file with progress tracking
python -m ilp_entropy file words.txt

# Save entropy at every position (default behavior)
python -m ilp_entropy file words.txt --output results.csv

# Find only optimal fixation positions
python -m ilp_entropy file words.txt --optimal-only
```

#### Parameter sweep analysis

```bash
# Sweep over multiple parameter combinations with nested progress tracking
python -m ilp_entropy sweep words.txt \
  --drop-left-range 0.1 0.5 0.1 \
  --drop-right-range 0.1 0.5 0.1 \
  --output-dir sweep_results/

# Control parallel processing
python -m ilp_entropy sweep words.txt \
  --drop-left-range 0.1 0.3 0.1 \
  --drop-right-range 0.1 0.3 0.1 \
  --workers 4
```

#### Top frequency words

```bash
# Process top 1000 most frequent words with progress
python -m ilp_entropy top-words 1000 --output top1k_results.csv

# Filter by word length
python -m ilp_entropy top-words 500 --word-lengths 4 5 6 7
```

#### Progress Reporting

All CLI commands show real-time progress with visual indicators:
```
Progress: [████████████░░░░░░░░] 1,234/2,000 (61.7%)
```

Parameter sweeps show nested progress:
```
Sweep: [██████░░░░░░░░░░░░] 12,500/25,000 (50.0%) | Combo 13/25 | Words 500/1000
```

### Python API

```python
# Core entropy functions
from ilp_entropy import ilp_entropy, position_entropy, optimal_fixation

# Single word entropy curve
entropy_curve = ilp_entropy("hello", drop_left=0.3, drop_right=0.4)
print(f"Entropy at each position: {entropy_curve}")

# Entropy at specific position
entropy_pos2 = position_entropy("hello", fixation_pos=2, drop_left=0.3, drop_right=0.4)
print(f"Entropy at position 2: {entropy_pos2}")

# Find optimal fixation position
optimal_pos = optimal_fixation("hello", drop_left=0.3, drop_right=0.4)
print(f"Optimal position: {optimal_pos}")

# Batch processing with progress tracking
from ilp_entropy.runner import batch_ilp_entropy, parameter_sweep

# Progress callback function
def progress_callback(completed, total):
    percent = 100 * completed / total
    print(f"Progress: {completed}/{total} ({percent:.1f}%)")

# Batch entropy calculation
words = ["hello", "world", "entropy", "calculation"]
df = batch_ilp_entropy(
    words,
    drop_left=0.3,
    drop_right=0.4,
    n_workers=4,
    progress_callback=progress_callback
)
print(df.head())

# Parameter sweep
sweep_df = parameter_sweep(
    words,
    drop_left_values=[0.2, 0.3, 0.4],
    drop_right_values=[0.2, 0.3, 0.4],
    output_dir="sweep_output/",
    progress_callback=progress_callback
)

# Load top frequency words
from ilp_entropy import get_top_words
top_words = get_top_words(1000, word_lengths=range(4, 8))
print(f"Loaded {len(top_words)} top words")
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

## Performance Features

### Intelligent Processing
- **Automatic workload detection**: Uses sequential processing for small jobs (≤5 words)
- **Smart worker management**: Automatically caps workers based on workload size
- **Corpus preloading**: Prevents concurrent file access issues
- **Memory efficient**: Processes large datasets without memory issues

### Progress Tracking
- **Real-time progress bars**: Visual indicators for all operations
- **ETA calculations**: Intelligent time estimates based on current performance
- **Nested progress**: Multi-level tracking for complex operations like parameter sweeps
- **Performance metrics**: Shows processing rates (words/sec)

## Testing

Run the comprehensive test suite:

```bash
# Quick smoke tests
poetry run pytest tests/test_smoke.py -v

# Integration tests
poetry run pytest tests/test_integration.py -v

# Performance tests (1000 words)
poetry run python tests/test_performance.py

# Performance tests with parameter sweep (25,000 computations)
poetry run python tests/test_performance.py --full

# All fast tests
poetry run pytest -v

# All tests including slow ones
poetry run pytest -v --runslow

# Skip slow tests
poetry run pytest -v -m "not slow"
```

### Test Organization

- **test_smoke.py**: Basic functionality and unit tests
- **test_integration.py**: End-to-end workflow tests
- **test_performance.py**: Large-scale performance benchmarks and stress tests

## Project Structure

```
ilp-entropy/
├── src/ilp_entropy/          # Main package
│   ├── __init__.py          # Public API exports
│   ├── __main__.py          # CLI entry point
│   ├── acuity.py            # Visual acuity calculations
│   ├── cli.py               # Command-line interface with progress reporting
│   ├── config.py            # Configuration and constants
│   ├── entropy.py           # Core entropy calculations
│   ├── io.py                # Data loading and I/O
│   ├── masks.py             # Visual field masks
│   ├── probability.py       # Probability calculations
│   ├── runner.py            # Batch processing and parallel execution
│   └── py.typed             # Type checking marker
├── tests/                   # Comprehensive test suite
│   ├── test_smoke.py        # Unit tests
│   ├── test_integration.py  # Integration tests
│   └── test_performance.py  # Performance benchmarks
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

## Performance Benchmarks

On a typical system, the package achieves:
- **Single word processing**: ~1000+ words/second
- **Batch processing**: ~500-1000 words/second (depending on word length)
- **Parameter sweeps**: ~25,000 word-parameter combinations in ~60 seconds
- **Memory usage**: Efficiently handles 1000+ word datasets

Performance automatically optimizes based on workload size and available CPU cores.

## License

[License information to be added]

## Contributing

[Contributing guidelines to be added]

## Citation

[Citation information to be added]

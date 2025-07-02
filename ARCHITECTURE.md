# ILP Entropy Package Architecture

## Overview

The Initial Landing Position (ILP) Entropy package quantifies uncertainty in word recognition for each possible fixation position within a word during reading. This Python implementation transforms the original R pipeline into an efficient, modular package suitable for research and production use.

## Core Concept

ILP Entropy measures the uncertainty (in bits) about word identity given:
- A target word
- A fixation position (where the eye lands)
- Visual acuity drop parameters (how recognition probability declines with distance from fixation)

The key innovation is modeling separate linear acuity drops to the left and right of fixation, reflecting empirical asymmetries in reading research.

## Mathematical Foundation

### Visual Acuity Model
For a fixation at position `f` in a word of length `L`:
```
P(recognize letter at position p) = max(0, 1 - |p - f| × drop_direction)
```
where `drop_direction` is `drop_left` if `p < f`, else `drop_right`.

### Mask-Based Computation
- Each possible pattern of letter visibility is a "mask" (2^L total for word length L)
- Mask probability: `P(mask|fixation) = ∏ᵢ P(see letter i)` assuming independence
- Entropy: `H = Σ P(mask) × H(candidates|mask)` where candidates match the visible letters

## Package Structure

```
src/ilp_entropy/
├── __init__.py          # Package initialization, main exports
├── config.py            # Global constants, CLI defaults, validation
├── io.py                # Corpus loading, caching, data cleaning
├── masks.py             # Binary mask enumeration and bit operations
├── acuity.py            # Visual acuity weight computation
├── probability.py       # Mask probability matrices with caching
├── entropy.py           # Core ILP entropy calculation
├── runner.py            # Parallel processing orchestration
├── cli.py               # Command-line interface
└── tests/               # Comprehensive test suite
```

## Module Specifications

### 1. config.py
**Purpose**: Centralized configuration and validation
- Default parameters (drop rates, word lengths, corpus paths)
- Path resolution for data files and cache directories
- Parameter validation functions
- CLI argument defaults

### 2. io.py ✅ (Implemented, needs bug fix)
**Purpose**: Corpus data loading and preprocessing
- Load and clean unigram frequency data
- Filter by word length, frequency thresholds
- Encode words as integer arrays (a=0, b=1, ..., z=25)
- Memory-efficient caching with @lru_cache
- Support for multiple languages/corpora

**Key Functions**:
- `get_corpus_index()`: Main entry point returning {length: (codes, freqs)}
- `_encode_word()`: String to integer array conversion
- `_get_corpus_index_cached()`: LRU cached loading

### 3. masks.py ✅ (Implemented)
**Purpose**: Efficient binary mask operations
- Enumerate all 2^L visibility patterns for word length L
- Use appropriate integer dtypes (uint8, uint16, uint32, uint64)
- Vectorized bit unpacking for NumPy operations

**Key Functions**:
- `enumerate_masks()`: Generate all possible masks
- `unpack_bits()`: Convert integer masks to boolean matrices
- `bit_dtype()`: Choose optimal dtype for word length

### 4. acuity.py ✅ (Implemented)
**Purpose**: Visual acuity weight computation
- Linear drop-off model with separate left/right parameters
- Vectorized computation for efficiency
- Precomputed weight matrices for batch operations

**Key Functions**:
- `acuity_weights()`: Single fixation position weights
- `weight_matrix()`: Precompute all fixation positions

### 5. probability.py ✅ (Implemented)
**Purpose**: Mask probability computation and caching
- Combine acuity weights with mask patterns
- Disk caching for expensive computations
- Memory-mapped loading for large matrices

**Key Functions**:
- `mask_prob_matrix()`: Core probability computation
- `get_mask_prob_matrix()`: Cached wrapper with disk persistence

### 6. entropy.py ✅ (Implemented)
**Purpose**: ILP entropy calculation
- Single-word entropy curves
- Candidate word matching using bit operations
- Vectorized Shannon entropy computation

**Key Functions**:
- `position_entropy()`: Single fixation position
- `ilp_entropy()`: Full entropy curve
- `optimal_fixation()`: Find minimum entropy position

### 7. runner.py ❌ (Empty - needs implementation)
**Purpose**: Parallel processing orchestration
- Batch processing of word lists
- Parameter sweep support
- Progress reporting and checkpointing
- Memory-efficient chunking
- Multi-core/distributed processing

**Required Functions**:
- `process_word_batch()`: Process chunk of words
- `parameter_sweep()`: Grid search over drop parameters
- `save_results()`: Structured output with metadata

### 8. cli.py ❌ (Empty - needs implementation)
**Purpose**: Command-line interface
- Argument parsing with validation
- Progress bars and logging
- Output format options (CSV, JSON, HDF5)
- Resume functionality for long runs

**Required Functions**:
- `main()`: CLI entry point
- `validate_args()`: Parameter validation
- `setup_logging()`: Configurable logging

## Performance Requirements

### Memory Efficiency
- Support words up to 11 letters (2^11 = 2048 masks maximum)
- Use appropriate dtypes: uint8 for small words, up to uint64 for 11 letters
- Memory-mapped file loading for large probability matrices
- Chunked processing to handle large word lists

### Computational Efficiency
- Vectorized NumPy operations throughout
- Minimal Python loops
- Efficient bit operations for mask handling
- Parallel processing support
- Disk caching of expensive computations

### Scalability
- Handle 10K+ words efficiently
- Support parameter sweeps (e.g., 17×17 drop parameter grid)
- Multi-core processing with progress tracking
- Checkpointing for long-running jobs

## Data Formats

### Input Corpus Format
CSV with columns:
- `unigram`: word (string, lowercase)
- `unigram_freq`: frequency (float)

### Internal Representation
- Words as uint8 arrays (0-25 for a-z)
- Frequencies as float32 arrays
- Grouped by word length for efficiency

### Cache Format
- Probability matrices: compressed NPZ files
- Filenames include parameter hash for invalidation
- Metadata stored separately

### Output Format
- Structured results with metadata
- Support multiple formats (CSV, JSON, Parquet)
- Include parameter provenance

## Testing Strategy

### Unit Tests
- Each module tested independently
- Edge cases (empty words, extreme parameters)
- Numerical precision validation
- Memory usage checks

### Integration Tests
- End-to-end pipeline validation
- Comparison with R reference implementation
- Performance benchmarks
- Cross-platform compatibility

### Regression Tests
- Golden outputs for standard test cases
- Parameter sensitivity analysis
- Reproducibility across platforms

## Error Handling

### Input Validation
- Parameter range checking (0 ≤ drop ≤ 1)
- Word length validation (1 ≤ length ≤ 11)
- Corpus format validation

### Graceful Degradation
- Handle missing corpus files
- Skip invalid words with warnings
- Partial results on interruption

### Debugging Support
- Verbose logging options
- Intermediate result inspection
- Memory usage monitoring

## Extension Points

### Multi-Language Support
- Language-specific alphabet handling
- Configurable corpus paths
- Character set validation

### Alternative Acuity Models
- Plugin architecture for acuity functions
- Non-linear drop models
- Position-dependent asymmetries

### Custom Corpora
- Support for domain-specific word lists
- User-provided frequency distributions
- Dynamic corpus updates

## Dependencies

### Core Dependencies
- `numpy >= 1.20`: Numerical computations
- `pandas >= 1.3`: Data loading and manipulation
- `pathlib`: Cross-platform path handling

### Optional Dependencies
- `numba`: JIT compilation for hot paths
- `dask`: Distributed computing
- `h5py`: HDF5 output format
- `matplotlib`: Visualization utilities

### Development Dependencies
- `pytest`: Testing framework
- `black`: Code formatting
- `mypy`: Type checking
- `ruff`: Linting
- `pre-commit`: Git hooks

## Deployment

### Package Distribution
- PyPI package with proper metadata
- Conda-forge recipe for scientific users
- Docker container for reproducible environments

### Documentation
- Sphinx-generated documentation
- Jupyter notebook tutorials
- API reference with examples
- Performance optimization guide

### Continuous Integration
- Automated testing on multiple Python versions
- Cross-platform compatibility checks
- Performance regression detection
- Code quality enforcement 
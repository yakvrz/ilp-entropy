"""ilp_entropy.runner
~~~~~~~~~~~~~~~~~~~~

Parallel processing orchestration for ILP entropy computation.

This module provides high-level functions for batch processing large numbers
of words, parameter sweeps, and parallel computation with progress tracking.

Key Features:
- Multi-core processing with configurable worker counts
- Progress tracking and reporting
- Memory-efficient chunking for large word lists
- Parameter sweep support (grid search)
- Structured output with metadata
- Checkpointing and resume functionality

Author: Koby Raz
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_DROP_LEFT,
    DEFAULT_DROP_RIGHT,
    validate_parameters,
    validate_word,
)
from .entropy import ilp_entropy, optimal_fixation, position_entropy
from .io import get_corpus_index

__all__ = [
    "process_word_batch",
    "parameter_sweep",
    "batch_ilp_entropy",
    "batch_optimal_fixation",
    "save_results",
    "load_results",
]

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Core batch processing functions                                             #
# --------------------------------------------------------------------------- #

def _process_single_word(
    word: str,
    drop_left: float,
    drop_right: float,
    include_curve: bool = True,
    include_optimal: bool = True,
) -> dict[str, Any]:
    """Process a single word and return results."""
    try:
        word = validate_word(word)
        results = {"word": word, "length": len(word)}

        if include_curve:
            curve = ilp_entropy(word, drop_left, drop_right)
            results["entropy_curve"] = curve
            results["mean_entropy"] = float(np.mean(curve))
            results["min_entropy"] = float(np.min(curve))
            results["max_entropy"] = float(np.max(curve))

        if include_optimal:
            opt_pos = optimal_fixation(word, drop_left, drop_right)
            results["optimal_position"] = opt_pos
            if include_curve:
                results["optimal_entropy"] = float(curve[opt_pos])
            else:
                results["optimal_entropy"] = position_entropy(
                    word, opt_pos, drop_left, drop_right
                )

        results["success"] = True
        results["error"] = None

    except Exception as e:
        results = {
            "word": word,
            "length": len(word) if isinstance(word, str) else None,
            "success": False,
            "error": str(e),
        }
        logger.warning(f"Failed to process word '{word}': {e}")

    return results

def process_word_batch(
    words: Sequence[str],
    drop_left: float = DEFAULT_DROP_LEFT,
    drop_right: float = DEFAULT_DROP_RIGHT,
    *,
    include_curve: bool = True,
    include_optimal: bool = True,
    n_workers: int | None = None,
    chunk_size: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """Process a batch of words in parallel.

    Parameters
    ----------
    words : Sequence[str]
        List of words to process
    drop_left : float
        Linear drop rate to the left of fixation
    drop_right : float
        Linear drop rate to the right of fixation
    include_curve : bool, default True
        Whether to compute full entropy curves
    include_optimal : bool, default True
        Whether to find optimal fixation positions
    n_workers : int, optional
        Number of worker processes (default: CPU count)
    chunk_size : int, optional
        Chunk size for parallel processing
    progress_callback : callable, optional
        Function called with (completed, total) for progress updates

    Returns
    -------
    list[dict]
        Results for each word
    """
    # Validate parameters
    validate_parameters(drop_left=drop_left, drop_right=drop_right)

    if not words:
        return []

    # Preload corpus in main process to avoid concurrent file access issues
    # This prevents "No data left in file" errors when workers try to load corpus simultaneously
    try:
        logger.debug("Preloading corpus to avoid concurrent access issues...")
        # Get all unique word lengths to preload relevant corpus data
        word_lengths = set(len(word) for word in words)
        get_corpus_index(word_lengths=word_lengths)
        logger.debug(f"Corpus preloaded for word lengths: {sorted(word_lengths)}")
    except Exception as e:
        logger.warning(f"Failed to preload corpus: {e}. Continuing anyway...")

    # For very small workloads, sequential processing is faster due to overhead
    if len(words) <= 5:
        logger.info(f"Using sequential processing for {len(words)} words (small workload)")
        results = []
        for i, word in enumerate(words):
            result = _process_single_word(word, drop_left, drop_right, include_curve, include_optimal)
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, len(words))
        return results

    n_workers = n_workers or mp.cpu_count()
    # Cap workers at reasonable level for workload size
    n_workers = min(n_workers, len(words), 8)  # Max 8 workers, no more than words
    chunk_size = chunk_size or max(1, len(words) // (n_workers * 2))  # Larger chunks

    logger.info(
        f"Processing {len(words)} words with {n_workers} workers, "
        f"chunk_size={chunk_size}"
    )

    results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit tasks
        future_to_word = {
            executor.submit(
                _process_single_word,
                word,
                drop_left,
                drop_right,
                include_curve,
                include_optimal,
            ): word
            for word in words
        }

        # Collect results as they complete
        for future in as_completed(future_to_word):
            word = future_to_word[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1

                if progress_callback:
                    progress_callback(completed, len(words))

            except Exception as e:
                logger.error(f"Unexpected error processing '{word}': {e}")
                results.append({
                    "word": word,
                    "success": False,
                    "error": f"Unexpected error: {e}",
                })
                completed += 1

    # Sort results to match input order
    word_to_index = {word: i for i, word in enumerate(words)}
    results.sort(key=lambda r: word_to_index.get(r["word"], len(words)))

    return results

def batch_ilp_entropy(
    words: Sequence[str],
    drop_left: float = DEFAULT_DROP_LEFT,
    drop_right: float = DEFAULT_DROP_RIGHT,
    **kwargs: Any,
) -> pd.DataFrame:
    """Compute ILP entropy curves for a batch of words.

    Parameters
    ----------
    words : Sequence[str]
        List of words to process
    drop_left : float
        Linear drop rate to the left of fixation
    drop_right : float
        Linear drop rate to the right of fixation
    **kwargs
        Additional arguments passed to process_word_batch

    Returns
    -------
    pd.DataFrame
        Results with columns: word, position, entropy, drop_left, drop_right
    """
    results = process_word_batch(
        words,
        drop_left=drop_left,
        drop_right=drop_right,
        include_curve=True,
        include_optimal=False,
        **kwargs,
    )

    # Convert to long format DataFrame
    rows = []
    for result in results:
        if result["success"] and "entropy_curve" in result:
            word = result["word"]
            curve = result["entropy_curve"]
            for pos, entropy in enumerate(curve):
                rows.append({
                    "word": word,
                    "position": pos + 1,  # 1-indexed positions
                    "entropy": entropy,
                    "drop_left": drop_left,
                    "drop_right": drop_right,
                })

    return pd.DataFrame(rows)

def batch_optimal_fixation(
    words: Sequence[str],
    drop_left: float = DEFAULT_DROP_LEFT,
    drop_right: float = DEFAULT_DROP_RIGHT,
    **kwargs: Any,
) -> pd.DataFrame:
    """Find optimal fixation positions for a batch of words.

    Parameters
    ----------
    words : Sequence[str]
        List of words to process
    drop_left : float
        Linear drop rate to the left of fixation
    drop_right : float
        Linear drop rate to the right of fixation
    **kwargs
        Additional arguments passed to process_word_batch

    Returns
    -------
    pd.DataFrame
        Results with columns: word, optimal_position, optimal_entropy, drop_left, drop_right
    """
    results = process_word_batch(
        words,
        drop_left=drop_left,
        drop_right=drop_right,
        include_curve=False,
        include_optimal=True,
        **kwargs,
    )

    # Convert to DataFrame
    rows = []
    for result in results:
        if result["success"]:
            rows.append({
                "word": result["word"],
                "optimal_position": result.get("optimal_position", None),
                "optimal_entropy": result.get("optimal_entropy", None),
                "drop_left": drop_left,
                "drop_right": drop_right,
            })

    return pd.DataFrame(rows)

# --------------------------------------------------------------------------- #
# Parameter sweep functions                                                   #
# --------------------------------------------------------------------------- #

def parameter_sweep(
    words: Sequence[str],
    drop_left_values: Sequence[float],
    drop_right_values: Sequence[float],
    *,
    output_dir: str | Path | None = None,
    save_individual: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Perform parameter sweep over drop rate combinations.

    Parameters
    ----------
    words : Sequence[str]
        List of words to process
    drop_left_values : Sequence[float]
        Left drop rate values to test
    drop_right_values : Sequence[float]
        Right drop rate values to test
    output_dir : str | Path, optional
        Directory to save results
    save_individual : bool, default True
        Whether to save individual parameter combination results
    progress_callback : callable, optional
        Function called with (completed, total) for overall progress
    **kwargs
        Additional arguments passed to process_word_batch

    Returns
    -------
    pd.DataFrame
        Combined results from all parameter combinations
    """
    from datetime import datetime

    # Validate parameters
    validate_parameters(drop_left=drop_left_values[0], drop_right=drop_right_values[0])

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate total combinations and words
    total_combinations = len(drop_left_values) * len(drop_right_values)
    total_words = len(words) * total_combinations
    processed_words = 0
    
    logger.info(
        f"Starting parameter sweep: {len(drop_left_values)} × "
        f"{len(drop_right_values)} = {total_combinations} combinations "
        f"({total_words:,} total word computations)"
    )

    all_results = []
    combination_count = 0
    overall_start_time = time.time()

    for drop_left in drop_left_values:
        for drop_right in drop_right_values:
            combination_count += 1
            
            # Calculate combination-level progress and ETA
            if combination_count > 1:
                elapsed_so_far = time.time() - overall_start_time
                avg_time_per_combo = elapsed_so_far / (combination_count - 1)
                remaining_combos = total_combinations - combination_count + 1
                eta_seconds = avg_time_per_combo * remaining_combos
                eta_str = f" (ETA: {eta_seconds/60:.1f}m)" if eta_seconds > 60 else f" (ETA: {eta_seconds:.0f}s)"
            else:
                eta_str = ""
            
            logger.info(
                f"Processing combination {combination_count}/{total_combinations}: "
                f"drop_left={drop_left:.3f}, drop_right={drop_right:.3f}{eta_str}"
            )

            start_time = time.time()

            # Create nested progress callback for this combination
            combo_start_words = processed_words
            def nested_progress(completed: int, total: int) -> None:
                if progress_callback:
                    overall_completed = combo_start_words + completed
                    progress_callback(overall_completed, total_words)

            # Process this parameter combination
            df = batch_ilp_entropy(
                words,
                drop_left=drop_left,
                drop_right=drop_right,
                progress_callback=nested_progress,
                **kwargs,
            )

            processed_words += len(words)
            elapsed = time.time() - start_time
            rate = len(words) / elapsed if elapsed > 0 else 0
            logger.info(f"Completed combination in {elapsed:.1f}s ({rate:.1f} words/sec)")

            # Save individual results if requested
            if save_individual and output_dir:
                filename = f"entropy_dropL_{drop_left:.3f}_dropR_{drop_right:.3f}.csv"
                df.to_csv(output_dir / filename, index=False)

            all_results.append(df)

    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Calculate final statistics
    total_sweep_time = time.time() - overall_start_time
    overall_rate = total_words / total_sweep_time
    
    logger.info(
        f"Parameter sweep completed: {total_combinations} combinations, "
        f"{len(words)} words each, {total_words:,} total computations "
        f"in {total_sweep_time:.1f}s ({overall_rate:.1f} words/sec overall)"
    )

    # Final progress update
    if progress_callback:
        progress_callback(total_words, total_words)

    # Save combined results and metadata
    if output_dir:
        combined_df.to_csv(output_dir / "combined_results.csv", index=False)

        # Save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "total_words": len(words),
            "total_combinations": total_combinations,
            "drop_left_values": list(drop_left_values),
            "drop_right_values": list(drop_right_values),
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    return combined_df

# --------------------------------------------------------------------------- #
# I/O functions                                                               #
# --------------------------------------------------------------------------- #

def save_results(
    results: pd.DataFrame | list[dict],
    output_path: str | Path,
    *,
    format: str = "csv",
    include_metadata: bool = True,
) -> None:
    """Save results to file with optional metadata."""
    output_path = Path(output_path)

    if isinstance(results, list):
        df = pd.DataFrame(results)
    else:
        df = results

    if format.lower() == "csv":
        df.to_csv(output_path, index=False)
    elif format.lower() == "json":
        df.to_json(output_path, orient="records", indent=2)
    elif format.lower() == "parquet":
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

    if include_metadata:
        metadata = {
            "n_rows": len(df),
            "columns": list(df.columns),
            "timestamp": time.time(),
            "format": format,
        }

        metadata_path = output_path.with_suffix(".metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

def load_results(input_path: str | Path) -> pd.DataFrame:
    """Load results from file, auto-detecting format."""
    input_path = Path(input_path)

    if input_path.suffix.lower() == ".csv":
        return pd.read_csv(input_path)
    elif input_path.suffix.lower() == ".json":
        return pd.read_json(input_path, orient="records")
    elif input_path.suffix.lower() == ".parquet":
        return pd.read_parquet(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

# --------------------------------------------------------------------------- #
# Convenience functions                                                       #
# --------------------------------------------------------------------------- #

def chunk_sequence(sequence: Sequence, chunk_size: int) -> Iterator[Sequence]:
    """Split a sequence into chunks of specified size."""
    for i in range(0, len(sequence), chunk_size):
        yield sequence[i:i + chunk_size]

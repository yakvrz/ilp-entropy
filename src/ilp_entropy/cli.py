"""ilp_entropy.cli
~~~~~~~~~~~~~~~~~~

Command-line interface for the ILP Entropy package.

This module provides a command-line interface for batch processing words,
parameter sweeps, and various output formats.

Usage Examples:
    # Process a list of words
    python -m ilp_entropy words word1 word2 word3 --drop-left 0.15 --drop-right 0.10

    # Process words from file
    python -m ilp_entropy file words.txt --output results.csv

    # Parameter sweep
    python -m ilp_entropy sweep words.txt \\
        --drop-left-range 0.1 0.2 0.05 --drop-right-range 0.1 0.2 0.05

Author: Koby Raz
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import (
    DEFAULT_DROP_LEFT,
    DEFAULT_DROP_RIGHT,
    DEFAULT_WORD_LENGTHS,
    validate_parameters,
)
from .io import get_top_words
from .runner import batch_ilp_entropy, batch_optimal_fixation, parameter_sweep

__all__ = ["main", "setup_logging"]

logger = logging.getLogger(__name__)

def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the CLI."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="ilp_entropy",
        description="Compute Initial Landing Position (ILP) Entropy for words",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s words hello world --drop-left 0.15 --drop-right 0.10
  %(prog)s file words.txt --output results.csv --format csv
  %(prog)s sweep words.txt --drop-left-range 0.1 0.3 0.05 --drop-right-range 0.1 0.3 0.05
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Increase verbosity (use -v, -vv, or -vvv)",
    )

    # Global parameters
    parser.add_argument(
        "--drop-left",
        type=float,
        default=DEFAULT_DROP_LEFT,
        help=f"Linear drop rate to the left of fixation (default: {DEFAULT_DROP_LEFT})",
    )

    parser.add_argument(
        "--drop-right",
        type=float,
        default=DEFAULT_DROP_RIGHT,
        help=f"Linear drop rate to the right of fixation (default: {DEFAULT_DROP_RIGHT})",
    )

    parser.add_argument(
        "--workers", "-j",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count)",
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file path",
    )

    parser.add_argument(
        "--format", "-f",
        choices=["csv", "json", "parquet"],
        default="csv",
        help="Output format (default: csv)",
    )

    # Subcommands
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        required=True,
    )

    # Words command
    words_parser = subparsers.add_parser(
        "words",
        help="Process individual words",
    )
    words_parser.add_argument(
        "words",
        nargs="+",
        help="Words to process",
    )
    words_parser.add_argument(
        "--optimal-only",
        action="store_true",
        help="Only compute optimal fixation positions",
    )

    # File command
    file_parser = subparsers.add_parser(
        "file",
        help="Process words from file",
    )
    file_parser.add_argument(
        "input_file",
        type=Path,
        help="Input file with one word per line",
    )
    file_parser.add_argument(
        "--optimal-only",
        action="store_true",
        help="Only compute optimal fixation positions",
    )

    # Sweep command
    sweep_parser = subparsers.add_parser(
        "sweep",
        help="Parameter sweep over drop rates",
    )
    sweep_parser.add_argument(
        "input_file",
        type=Path,
        help="Input file with one word per line",
    )
    sweep_parser.add_argument(
        "--drop-left-range",
        nargs=3,
        type=float,
        metavar=("START", "STOP", "STEP"),
        default=[0.05, 0.25, 0.05],
        help="Drop left range: start stop step (default: 0.05 0.25 0.05)",
    )
    sweep_parser.add_argument(
        "--drop-right-range",
        nargs=3,
        type=float,
        metavar=("START", "STOP", "STEP"),
        default=[0.05, 0.25, 0.05],
        help="Drop right range: start stop step (default: 0.05 0.25 0.05)",
    )
    sweep_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for sweep results",
    )

    # Top words command
    top_words_parser = subparsers.add_parser(
        "top-words",
        help="Process top N most frequent words from corpus",
    )
    top_words_parser.add_argument(
        "n",
        type=int,
        help="Number of top words to process",
    )
    top_words_parser.add_argument(
        "--corpus-path",
        type=Path,
        help="Path to corpus CSV file (default: data/unigrams_en.csv)",
    )
    top_words_parser.add_argument(
        "--min-freq",
        type=float,
        default=0.0,
        help="Minimum frequency threshold (default: 0.0)",
    )
    top_words_parser.add_argument(
        "--word-lengths",
        nargs="+",
        type=int,
        default=list(DEFAULT_WORD_LENGTHS),
        help=f"Word lengths to include (default: {list(DEFAULT_WORD_LENGTHS)})",
    )
    top_words_parser.add_argument(
        "--optimal-only",
        action="store_true",
        help="Only compute optimal fixation positions",
    )

    return parser

def load_words_from_file(file_path: Path) -> list[str]:
    """Load words from a text file (one word per line)."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    words = []
    with open(file_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            word = line.strip().lower()
            if word and word.isalpha():
                words.append(word)
            elif word:
                logger.warning(f"Skipping invalid word on line {line_num}: '{word}'")

    if not words:
        raise ValueError(f"No valid words found in {file_path}")

    logger.info(f"Loaded {len(words)} words from {file_path}")
    return words

def progress_callback(completed: int, total: int) -> None:
    """Enhanced progress callback for batch processing."""
    if total > 0:
        percent = 100 * completed / total
        bar_length = 40
        filled_length = int(bar_length * completed // total)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"\rProgress: [{bar}] {completed:,}/{total:,} ({percent:.1f}%)", end="", flush=True)
        if completed == total:
            print()  # New line when complete

def handle_words_command(args: argparse.Namespace) -> None:
    """Handle the 'words' subcommand."""
    validate_parameters(drop_left=args.drop_left, drop_right=args.drop_right)

    logger.info(f"Processing {len(args.words)} words")

    if args.optimal_only:
        df = batch_optimal_fixation(
            args.words,
            drop_left=args.drop_left,
            drop_right=args.drop_right,
            n_workers=args.workers,
            progress_callback=progress_callback,
        )
    else:
        df = batch_ilp_entropy(
            args.words,
            drop_left=args.drop_left,
            drop_right=args.drop_right,
            n_workers=args.workers,
            progress_callback=progress_callback,
        )

    if args.output:
        output_path = args.output
        if output_path.suffix == "":
            output_path = output_path.with_suffix(f".{args.format}")

        if args.format == "csv":
            df.to_csv(output_path, index=False)
        elif args.format == "json":
            df.to_json(output_path, orient="records", indent=2)
        elif args.format == "parquet":
            df.to_parquet(output_path, index=False)

        logger.info(f"Results saved to {output_path}")
    else:
        print(df.to_string(index=False))

def handle_file_command(args: argparse.Namespace) -> None:
    """Handle the 'file' subcommand."""
    words = load_words_from_file(args.input_file)

    # Update args with loaded words and delegate to words handler
    args.words = words
    handle_words_command(args)

def handle_sweep_command(args: argparse.Namespace) -> None:
    """Handle the 'sweep' subcommand."""
    words = load_words_from_file(args.input_file)

    # Generate parameter ranges
    dl_start, dl_stop, dl_step = args.drop_left_range
    dr_start, dr_stop, dr_step = args.drop_right_range

    drop_left_values = np.arange(dl_start, dl_stop + dl_step/2, dl_step)
    drop_right_values = np.arange(dr_start, dr_stop + dr_step/2, dr_step)

    total_combinations = len(drop_left_values) * len(drop_right_values)
    total_computations = len(words) * total_combinations

    logger.info(
        f"Parameter sweep: {len(drop_left_values)} × {len(drop_right_values)} = "
        f"{total_combinations} combinations ({total_computations:,} total computations)"
    )

    # Set up output directory
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"ilp_entropy_sweep_{timestamp}")

    # Run parameter sweep with progress reporting
    start_time = time.time()
    df = parameter_sweep(
        words,
        drop_left_values,
        drop_right_values,
        output_dir=output_dir,
        n_workers=args.workers,
        progress_callback=progress_callback,
    )
    elapsed = time.time() - start_time

    logger.info(
        f"Parameter sweep completed in {elapsed:.1f}s "
        f"({total_computations / elapsed:.1f} words/sec overall)"
    )

    if args.output:
        output_path = args.output
        if output_path.suffix == "":
            output_path = output_path.with_suffix(f".{args.format}")

        if args.format == "csv":
            df.to_csv(output_path, index=False)
        elif args.format == "json":
            df.to_json(output_path, orient="records", indent=2)
        elif args.format == "parquet":
            df.to_parquet(output_path, index=False)

        logger.info(f"Combined results saved to {output_path}")

def handle_top_words_command(args: argparse.Namespace) -> None:
    """Handle the 'top-words' subcommand."""
    validate_parameters(drop_left=args.drop_left, drop_right=args.drop_right)

    logger.info(f"Loading top {args.n} words from corpus")

    # Load top N words from corpus
    words = get_top_words(
        args.n,
        csv_path=args.corpus_path,
        word_lengths=args.word_lengths,
        min_freq=args.min_freq,
    )

    logger.info(f"Loaded {len(words)} words")

    if len(words) < args.n:
        logger.warning(
            f"Only found {len(words)} words matching criteria "
            f"(requested {args.n})"
        )

    # Process the words
    if args.optimal_only:
        df = batch_optimal_fixation(
            words,
            drop_left=args.drop_left,
            drop_right=args.drop_right,
            n_workers=args.workers,
            progress_callback=progress_callback,
        )
    else:
        df = batch_ilp_entropy(
            words,
            drop_left=args.drop_left,
            drop_right=args.drop_right,
            n_workers=args.workers,
            progress_callback=progress_callback,
        )

    if args.output:
        output_path = args.output
        if output_path.suffix == "":
            output_path = output_path.with_suffix(f".{args.format}")

        if args.format == "csv":
            df.to_csv(output_path, index=False)
        elif args.format == "json":
            df.to_json(output_path, orient="records", indent=2)
        elif args.format == "parquet":
            df.to_parquet(output_path, index=False)

        logger.info(f"Results saved to {output_path}")
    else:
        print(df.to_string(index=False))

def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    # Set up logging
    if args.verbose == 0:
        log_level = "WARNING"
    elif args.verbose == 1:
        log_level = "INFO"
    else:
        log_level = "DEBUG"

    setup_logging(log_level)

    try:
        if args.command == "words":
            handle_words_command(args)
        elif args.command == "file":
            handle_file_command(args)
        elif args.command == "sweep":
            handle_sweep_command(args)
        elif args.command == "top-words":
            handle_top_words_command(args)
        else:
            parser.error(f"Unknown command: {args.command}")

        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose >= 2:
            logger.exception("Full traceback:")
        return 1

if __name__ == "__main__":
    sys.exit(main())

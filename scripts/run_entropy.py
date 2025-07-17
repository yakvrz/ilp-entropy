"""
This script serves as a command-line entry point to run the ILP entropy computation.

It loads its default parameters from a 'config.json' file and allows any parameter
to be overridden via command-line arguments. It supports parallel execution and
parameter sweeps for drop rates.
"""
import argparse
import json
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import os
import numpy as np

from src.entropy import ilp_entropy
from src.io import get_corpus_words, get_corpus_index


def load_config(config_path):
    """Loads a JSON configuration file."""
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Info: config.json not found. Using command-line arguments and defaults.")
        return {}
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {config_path}. Using defaults.")
        return {}


def parse_sweep_range(range_str: str) -> np.ndarray:
    """Parses a 'START,END,STEP' string into a NumPy arange."""
    try:
        parts = [float(p) for p in range_str.split(',')]
        if len(parts) != 3:
            raise ValueError
        # Add a small epsilon to the end to make it inclusive
        return np.arange(parts[0], parts[1] + parts[2] / 2, parts[2])
    except (ValueError, IndexError):
        raise argparse.ArgumentTypeError(
            "Sweep range must be in 'START,END,STEP' format (e.g., '0.1,0.9,0.1')."
        )


def process_word(word, drop_left, drop_right, min_freq, corpus_index):
    """Worker function to process a single word with specific parameters."""
    try:
        entropy_curve = ilp_entropy(
            word=word,
            drop_left=drop_left,
            drop_right=drop_right,
            corpus=corpus_index,
            min_freq=min_freq,
        )
        word_results = []
        for i, entropy_value in enumerate(entropy_curve):
            word_results.append(
                {
                    "word": word,
                    "pos": i + 1,
                    "entropy": entropy_value,
                    "drop_left": drop_left,
                    "drop_right": drop_right,
                }
            )
        return word_results
    except ValueError as e:
        return {"word": word, "error": str(e)}


def main():
    """Parses command-line arguments and runs the entropy computation."""

    config = load_config("config.json")

    parser = argparse.ArgumentParser(
        description="Calculate ILP entropy for a given list of words, with optional parameter sweeps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--word-list",
        type=str,
        default=config.get("word_list"),
        help="Path to a file containing a list of words to process. Ignored if --all-corpus-words is set.",
    )
    parser.add_argument(
        "--corpus-file",
        type=str,
        default=config.get("corpus_file", "opensubtitles.csv"),
        help="Path to the corpus CSV file.",
    )
    parser.add_argument(
        "--drop-left",
        type=float,
        default=config.get("drop_left", 0.1),
        help="Linear acuity drop-off rate to the left of fixation. Used if --sweep-left is not set.",
    )
    parser.add_argument(
        "--drop-right",
        type=float,
        default=config.get("drop_right", 0.2),
        help="Linear acuity drop-off rate to the right of fixation. Used if --sweep-right is not set.",
    )
    parser.add_argument(
        "--sweep-left",
        type=parse_sweep_range,
        default=None,
        help="Define a sweep range for drop_left as 'START,END,STEP'.",
    )
    parser.add_argument(
        "--sweep-right",
        type=parse_sweep_range,
        default=None,
        help="Define a sweep range for drop_right as 'START,END,STEP'.",
    )
    parser.add_argument(
        "--min-freq",
        type=float,
        default=config.get("min_freq", 1e-7),
        help="Minimum word frequency to include from the corpus.",
    )
    parser.add_argument(
        "--word-lengths",
        type=int,
        nargs="*",
        default=config.get("word_lengths"),
        help="A list of word lengths to process. If not specified, all words are processed.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=config.get("output_file"),
        help="Path to save the results to a CSV file. If not specified, results are printed to the console.",
    )
    parser.add_argument(
        "--all-corpus-words",
        action="store_true",
        help="If set, process all words in the corpus file that meet the minimum frequency requirement.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Number of worker processes to use for parallel execution.",
    )

    args = parser.parse_args()

    # Determine the source of words to process
    if args.all_corpus_words:
        print(f"Loading all words from corpus: {args.corpus_file}...")
        words_to_process = get_corpus_words(
            args.corpus_file, min_freq=args.min_freq
        )
    else:
        if not args.word_list:
            parser.error("Either --word-list must be specified, or --all-corpus-words must be set.")
        try:
            with open(args.word_list, "r") as f:
                words_to_process = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Error: Word list file not found at {args.word_list}")
            return

    if args.word_lengths:
        words_to_process = [w for w in words_to_process if len(w) in args.word_lengths]

    # --- Pre-load the entire corpus index into memory once ---
    all_lengths = set(len(w) for w in words_to_process)
    print("Pre-loading corpus index for all required word lengths...")
    corpus_index = get_corpus_index(
        word_lengths=all_lengths,
        csv_path=args.corpus_file,
        min_freq=args.min_freq,
    )
    print("Corpus pre-loaded.")

    # --- Generate parameter grid for sweep ---
    left_drops = args.sweep_left if args.sweep_left is not None else [args.drop_left]
    right_drops = args.sweep_right if args.sweep_right is not None else [args.drop_right]

    param_grid = [
        {"drop_left": dl, "drop_right": dr}
        for dl in left_drops
        for dr in right_drops
    ]

    print(f"Starting calculation for {len(words_to_process)} words across {len(param_grid)} parameter combinations using {args.workers} workers...")
    
    all_results = []
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for params in tqdm(param_grid, desc="Parameter Sweep"):
            current_drop_left = params["drop_left"]
            current_drop_right = params["drop_right"]

            worker_func = partial(
                process_word,
                drop_left=current_drop_left,
                drop_right=current_drop_right,
                min_freq=args.min_freq,
                corpus_index=corpus_index,
            )
            
            futures = {executor.submit(worker_func, word) for word in words_to_process}
            
            sweep_desc = f"Sweep (L={current_drop_left:.2f}, R={current_drop_right:.2f})"
            for future in tqdm(as_completed(futures), total=len(words_to_process), desc=sweep_desc, leave=False):
                result = future.result()
                if isinstance(result, list):
                    all_results.extend(result)
                elif isinstance(result, dict) and 'error' in result:
                    print(f"\nCould not process word '{result['word']}': {result['error']}")

    if not all_results:
        print("No results to show.")
        return

    results_df = pd.DataFrame(all_results).sort_values(by=["word", "pos"]).reset_index(drop=True)

    if args.output_file:
        results_df.to_csv(args.output_file, index=False)
        print(f"\nResults saved to {args.output_file}")
    else:
        print("\n--- Results ---")
        print(results_df.to_string(index=False))


if __name__ == "__main__":
    main() 
"""
This script serves as a command-line entry point to run the ILP entropy computation.

It loads its default parameters from a 'config.json' file and allows any parameter
to be overridden via command-line arguments. It supports parallel execution and
parameter sweeps for drop rates.
"""
import argparse
import json
import os
import sys
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.entropy import ilp_entropy
from src.io import load_corpus, load_words
from src.masks import enumerate_masks, unpack_bits

# Globals set inside each worker to avoid re-pickling large objects per task
GLOBAL_CORPUS_INDEX = None
GLOBAL_MASK_CACHE = None
GLOBAL_CHAR_MAP = None


def build_corpus_index(
    corpus_df: pd.DataFrame, word_lengths: list[int], *, char_map: dict[str, int] | None = None
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Builds a structured dictionary (index) from the corpus for efficient lookups.

    For each specified word length, this function creates a tuple containing:
    - A NumPy array of word character codes.
    - A NumPy array of corresponding word frequencies.

    Args:
        corpus_df: The pre-loaded and filtered corpus DataFrame.
        word_lengths: A list of word lengths to be included in the index.
        char_map: Optional mapping of characters to integer codes. If None,
            a default a=0, b=1, ... encoding is used (suitable for English).

    Returns:
        A dictionary where keys are word lengths and values are (codes, freqs) tuples.
    """
    corpus_index: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for L in word_lengths:
        # Filter for words of a specific length
        sub = corpus_df[corpus_df["word"].str.len() == L].copy()
        if sub.empty:
            continue

        if char_map is None:
            # Default English-centric encoding.
            codes = (
                sub["word"]
                .apply(lambda s: [ord(ch) - 97 for ch in s])
                .explode()
                .astype("uint8")
                .to_numpy()
                .reshape(-1, L)
            )
        else:
            # General unicode-friendly encoding driven by provided mapping.
            codes = (
                sub["word"]
                .apply(lambda s: [char_map[ch] for ch in s])
                .explode()
                .astype("uint16")
                .to_numpy()
                .reshape(-1, L)
            )

        # Convert frequencies to a NumPy array
        freqs = sub["freq"].to_numpy(dtype="float32")

        corpus_index[L] = (codes, freqs)
    return corpus_index


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


def init_worker(corpus_index, mask_cache, char_map):
    """Initializer to set heavy, read-only state in each worker process."""
    global GLOBAL_CORPUS_INDEX, GLOBAL_MASK_CACHE, GLOBAL_CHAR_MAP
    GLOBAL_CORPUS_INDEX = corpus_index
    GLOBAL_MASK_CACHE = mask_cache
    GLOBAL_CHAR_MAP = char_map


def process_word(word: str, drop_left: float, drop_right: float):
    """Worker function to process a single word using globals set by init_worker."""
    try:
        entropy_curve = ilp_entropy(
            word=word,
            drop_left=drop_left,
            drop_right=drop_right,
            corpus=GLOBAL_CORPUS_INDEX,
            mask_cache=GLOBAL_MASK_CACHE,
            char_map=GLOBAL_CHAR_MAP,
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


def save_metadata(directory, params):
    """Saves the run parameters to a JSON file, rounding floats for clarity."""
    params_to_save = params.copy()

    # If all corpus words are used, word_list is irrelevant and should be removed for clarity.
    if params_to_save.get("all_corpus_words") and "word_list" in params_to_save:
        del params_to_save["word_list"]

    metadata_path = os.path.join(directory, "metadata.json")
    with open(metadata_path, "w") as f:
        # Convert numpy arrays and round floats for clean serialization
        serializable_params = {}
        for k, v in params_to_save.items():
            if isinstance(v, np.ndarray):
                # Round array elements to a few decimal places
                serializable_params[k] = [round(x, 8) for x in v.tolist()]
            elif isinstance(v, float):
                serializable_params[k] = round(v, 8)
            else:
                serializable_params[k] = v

        json.dump(serializable_params, f, indent=4)
    print(f"Run metadata saved to {metadata_path}")


def main():
    """Parses command-line arguments and runs the entropy computation."""

    parser = argparse.ArgumentParser(
        description="Calculate ILP entropy for a given list of words, with optional parameter sweeps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--corpus-file",
        type=str,
        required=True,
        help="Path to the corpus file (e.g., data/opensubtitles_en.csv)",
    )
    parser.add_argument(
        "--min-freq",
        type=float,
        required=True,
        help="Minimum word frequency to include from the corpus (e.g., 1e-6).",
    )

    # Mutually exclusive group for word source
    word_source_group = parser.add_mutually_exclusive_group(required=True)
    word_source_group.add_argument(
        "--word-list",
        type=str,
        help="Path to a file containing words to process (e.g., word_list.txt).",
    )
    word_source_group.add_argument(
        "--all-corpus-words",
        action="store_true",
        help="Process all words in the corpus that meet the frequency requirement.",
    )

    # Mutually exclusive group for left drop
    left_drop_group = parser.add_mutually_exclusive_group(required=True)
    left_drop_group.add_argument(
        "--drop-left",
        type=float,
        help="Linear acuity drop-off rate to the left of fixation (e.g., 0.1).",
    )
    left_drop_group.add_argument(
        "--sweep-left",
        type=parse_sweep_range,
        help="Define a sweep range for drop_left as 'START,END,STEP'.",
    )

    # Mutually exclusive group for right drop
    right_drop_group = parser.add_mutually_exclusive_group(required=True)
    right_drop_group.add_argument(
        "--drop-right",
        type=float,
        help="Linear acuity drop-off rate to the right of fixation (e.g., 0.2).",
    )
    right_drop_group.add_argument(
        "--sweep-right",
        type=parse_sweep_range,
        help="Define a sweep range for drop_right as 'START,END,STEP'.",
    )
    
    # Optional arguments that still have defaults
    parser.add_argument(
        "--word-lengths",
        type=int,
        nargs="*",
        default=None,
        help="A list of word lengths to process. If not specified, all words are processed.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Number of worker processes to use for parallel execution.",
    )

    args = parser.parse_args()

    # --- Create a unique, timestamped output directory ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("output", f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Outputs will be saved to: {output_dir}")

    # Save run parameters to a metadata file
    save_metadata(output_dir, vars(args))


    # --- Pre-load the corpus and create the corpus index ---
    print(f"Loading and processing corpus from {args.corpus_file}...")
    corpus_df, char_map = load_corpus(
        corpus_file=args.corpus_file, min_freq=args.min_freq
    )

    # Determine the source of words to process
    if args.all_corpus_words:
        print("Using all words from the processed corpus...")
        words_to_process = corpus_df["word"].tolist()
    else:
        try:
            words_to_process = load_words(args.word_list)
        except FileNotFoundError:
            print(f"Error: Word list file not found at {args.word_list}")
            return

    if args.word_lengths:
        words_to_process = [w for w in words_to_process if len(w) in args.word_lengths]

    # --- Pre-calculate all required data to pass to workers ---
    all_lengths = sorted(list(set(len(w) for w in words_to_process)))

    print("Pre-building corpus index for all required word lengths...")
    corpus_index = build_corpus_index(
        corpus_df=corpus_df,
        word_lengths=all_lengths,
        char_map=char_map,
    )
    print("Corpus index created.")

    print("Pre-calculating visibility masks for all required word lengths...")
    mask_cache = {
        L: unpack_bits(enumerate_masks(L), length=L) for L in all_lengths
    }
    print("Mask cache created.")


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
    
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(corpus_index, mask_cache, char_map),
    ) as executor:
        for params in tqdm(param_grid, desc="Parameter Sweep"):
            current_drop_left = params["drop_left"]
            current_drop_right = params["drop_right"]

            futures = [
                executor.submit(process_word, word, current_drop_left, current_drop_right)
                for word in words_to_process
            ]

            sweep_desc = f"Sweep (L={current_drop_left:.2f}, R={current_drop_right:.2f})"
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=sweep_desc,
                leave=False,
            ):
                result = future.result()
                if isinstance(result, list):
                    all_results.extend(result)
                elif isinstance(result, dict) and 'error' in result:
                    print(f"\nCould not process word '{result['word']}': {result['error']}")

    if not all_results:
        print("No results to show.")
        return

    results_df = pd.DataFrame(all_results).sort_values(by=["word", "pos"]).reset_index(drop=True)

    # Save results to the run-specific output directory
    output_csv_path = os.path.join(output_dir, "results.csv")
    results_df.to_csv(output_csv_path, index=False)
    print(f"\nResults saved to {output_csv_path}")


if __name__ == "__main__":
    main()

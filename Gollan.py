#!/usr/bin/env python3
"""
Generate ILP entropy tables for the top-K words per length, saving one row per
word per position. Designed for the Hebrew OpenSubtitles corpus but works for
any corpus file accepted by src/io.load_corpus.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from scripts.main import build_corpus_index
from src.entropy import ilp_entropy
from src.io import load_corpus
from src.masks import enumerate_masks, unpack_bits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ILP entropy tables (one row per word-position).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--corpus-file",
        type=str,
        default="data/opensubtitles_he.csv",
        help="Input corpus CSV with columns word,freq.",
    )
    parser.add_argument(
        "--min-freq",
        type=float,
        default=1e-6,
        help="Frequency cutoff applied before processing.",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[5, 6, 7, 8],
        help="Word lengths to process.",
    )
    parser.add_argument(
        "--drop-left",
        type=float,
        default=0.25,
        help="Drop-off to the left of fixation.",
    )
    parser.add_argument(
        "--drop-right",
        type=float,
        default=0.25,
        help="Drop-off to the right of fixation.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=6000,
        help="Process at most this many most-frequent words per length.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of words to keep per target position (after filtering for unique minima).",
    )
    parser.add_argument(
        "--positions",
        type=int,
        nargs="+",
        default=[2, 6],
        help="Fixation positions to filter on.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Gollan_he",
        help="Directory to store the generated CSVs.",
    )
    return parser.parse_args()


def iter_records(
    words: Iterable[str],
    freqs: Iterable[float],
    *,
    drop_left: float,
    drop_right: float,
    corpus_index,
    mask_cache,
    char_map,
) -> list[tuple[str, float, np.ndarray, int, float, int]]:
    records = []
    for word, freq in zip(words, freqs):
        h = np.array(
            ilp_entropy(
                word,
                drop_left,
                drop_right,
                corpus=corpus_index,
                mask_cache=mask_cache,
                char_map=char_map,
            ),
            dtype=float,
        )
        min_val = float(h.min())
        min_pos = int(h.argmin()) + 1
        tie_count = int(np.isclose(h, min_val).sum())
        records.append((word, float(freq), h, min_pos, min_val, tie_count))
    return records


def write_subset(
    records: list[tuple[str, float, np.ndarray, int, float, int]],
    *,
    pos: int,
    length: int,
    top_k: int,
    drop_left: float,
    drop_right: float,
    out_path: Path,
) -> dict[str, int]:
    filtered = [r for r in records if r[3] == pos and r[5] == 1]
    filtered.sort(key=lambda r: r[1], reverse=True)
    top = filtered[:top_k]

    rows = []
    for word, freq, h, min_pos, min_val, _ in top:
        for i, ent in enumerate(h, start=1):
            rows.append(
                {
                    "word": word,
                    "length": length,
                    "frequency": freq,
                    "pos": i,
                    "entropy": float(ent),
                    "min_entropy_pos": min_pos,
                    "min_entropy": min_val,
                    "drop_left": drop_left,
                    "drop_right": drop_right,
                }
            )

    pd.DataFrame(rows).to_csv(out_path, index=False)
    return {
        "candidates": len(filtered),
        "saved_words": len(top),
        "rows_written": len(rows),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_df, char_map = load_corpus(args.corpus_file, min_freq=args.min_freq)

    summary = {}
    for L in args.lengths:
        sub = corpus_df[corpus_df["word"].str.len() == L].copy()
        if sub.empty:
            for pos in args.positions:
                (out_dir / f"entropy_pos{pos}_len{L}_he.csv").write_text("")
                summary[(L, pos)] = {"candidates": 0, "saved_words": 0, "rows_written": 0}
            continue

        # Build corpus index for this length
        corpus_index = build_corpus_index(sub, [L], char_map=char_map)
        mask_cache = {L: unpack_bits(enumerate_masks(L), length=L)}

        targets = sub.nlargest(args.max_words, "freq") if args.max_words else sub
        records = iter_records(
            targets["word"],
            targets["freq"],
            drop_left=args.drop_left,
            drop_right=args.drop_right,
            corpus_index=corpus_index,
            mask_cache=mask_cache,
            char_map=char_map,
        )

        for pos in args.positions:
            if pos > L:
                # Write empty file for impossible positions.
                pd.DataFrame(
                    columns=[
                        "word",
                        "length",
                        "frequency",
                        "pos",
                        "entropy",
                        "min_entropy_pos",
                        "min_entropy",
                        "drop_left",
                        "drop_right",
                    ]
                ).to_csv(out_dir / f"entropy_pos{pos}_len{L}_he.csv", index=False)
                summary[(L, pos)] = {"candidates": 0, "saved_words": 0, "rows_written": 0}
                continue

            stats = write_subset(
                records,
                pos=pos,
                length=L,
                top_k=args.top_k,
                drop_left=args.drop_left,
                drop_right=args.drop_right,
                out_path=out_dir / f"entropy_pos{pos}_len{L}_he.csv",
            )
            summary[(L, pos)] = stats

    for key, val in summary.items():
        L, pos = key
        print(f"len={L}, pos={pos}: {val}")


if __name__ == "__main__":
    main()

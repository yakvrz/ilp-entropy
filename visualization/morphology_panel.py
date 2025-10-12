"""
Create a minimal panel figure illustrating how ILP entropy reflects morphology.

The script generates a vertical stack of entropy curves for selected exemplar
words. Each row displays the word label and its entropy trajectory, with a short
description highlighting the relevant morphological feature.

Output: a PNG file (default `figures/morphology_entropy.png`).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.entropy import ilp_entropy
from src.io import load_corpus
from src.masks import enumerate_masks, unpack_bits


@dataclass(frozen=True)
class MorphExample:
    word: str
    span: tuple[int, int]
    note: str


DEFAULT_EXAMPLES: tuple[MorphExample, ...] = (
    MorphExample("running", (4, 7), "Suffix '-ing' keeps late uncertainty elevated"),
    MorphExample("unknown", (0, 3), "Prefix 'un-' lifts initial entropy"),
    MorphExample("preview", (0, 3), "Prefix 'pre-' shifts curve leftward"),
    MorphExample("teacher", (5, 7), "Suffix '-er' raises ending entropy"),
)


def build_corpus_index(
    corpus_df: pd.DataFrame, lengths: Iterable[int]
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    index: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for L in sorted(set(lengths)):
        subset = corpus_df[corpus_df["word"].str.len() == L].copy()
        if subset.empty:
            continue
        codes = (
            subset["word"]
            .apply(lambda s: [ord(ch) - 97 for ch in s])
            .explode()
            .astype("uint8")
            .to_numpy()
            .reshape(-1, L)
        )
        freqs = subset["freq"].to_numpy(dtype="float32")
        index[L] = (codes, freqs)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a minimal morphology-focused ILP entropy panel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--corpus-file",
        required=True,
        help="Path to the corpus CSV used for entropy calculations.",
    )
    parser.add_argument(
        "--min-freq",
        type=float,
        required=True,
        help="Minimum frequency threshold passed to load_corpus.",
    )
    parser.add_argument(
        "--drop-left",
        type=float,
        default=0.3,
        help="Left acuity drop parameter.",
    )
    parser.add_argument(
        "--drop-right",
        type=float,
        default=0.3,
        help="Right acuity drop parameter.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/morphology_entropy.png",
        help="Path to save the generated PNG figure.",
    )
    args = parser.parse_args()

    corpus_df, _ = load_corpus(args.corpus_file, min_freq=args.min_freq)
    examples = DEFAULT_EXAMPLES
    lengths = [len(ex.word) for ex in examples]
    corpus_index = build_corpus_index(corpus_df, lengths)
    mask_cache = {L: unpack_bits(enumerate_masks(L), length=L) for L in corpus_index.keys()}

    for ex in examples:
        L = len(ex.word)
        if L not in corpus_index or L not in mask_cache:
            raise ValueError(f"No corpus data available for the word '{ex.word}'.")

    fig_height = 2.4 * len(examples)
    fig, axes = plt.subplots(len(examples), 1, figsize=(4.6, fig_height), sharex=False)
    if len(examples) == 1:
        axes = [axes]

    for ax, example in zip(axes, examples):
        word = example.word.lower()
        L = len(word)
        corpus_codes, corpus_freqs = corpus_index[L]
        entropy_curve = ilp_entropy(
            word=word,
            drop_left=args.drop_left,
            drop_right=args.drop_right,
            corpus={L: (corpus_codes, corpus_freqs)},
            mask_cache={L: mask_cache[L]},
        )
        positions = np.arange(1, L + 1)
        entropy = np.asarray(entropy_curve, dtype=float)

        ax.plot(positions, entropy, color="#1b6ca8", linewidth=2.3)
        ax.set_xlim(0.5, len(positions) + 0.5)

        y_lower = max(0.0, entropy.min() - 0.1)
        y_upper = entropy.max() + 0.3
        lower_int = int(np.floor(y_lower))
        upper_int = int(np.ceil(y_upper))
        if upper_int == lower_int:
            upper_int = lower_int + 1
        y_ticks = np.arange(lower_int, upper_int + 1)

        ax.set_ylim(y_lower, y_upper)
        ax.set_xticks(positions)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{int(tick)}" for tick in y_ticks])
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        ax.text(
            0.0,
            1.19,
            word.upper(),
            transform=ax.transAxes,
            fontsize=12.5,
            fontweight="bold",
            color="#1b1b1b",
            ha="left",
            va="bottom",
        )
        ax.text(
            0.0,
            1.07,
            example.note,
            transform=ax.transAxes,
            fontsize=10.0,
            color="#555555",
            ha="left",
            va="bottom",
        )

        if ax is axes[-1]:
            ax.set_xlabel("Fixation position")
        else:
            ax.set_xlabel("")

    axes[0].set_ylabel("Entropy (bits)")
    fig.tight_layout(rect=(0, 0, 1, 0.98), h_pad=1.2)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved morphology panel to {output_path}")


if __name__ == "__main__":
    main()

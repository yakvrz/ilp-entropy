"""
Generate an HTML page that highlights the minimal-entropy position within each
word in a paragraph. Uses a fixed symmetrical drop of 0.25 by default.

Usage (defaults shown):
  python3 scripts/highlight_mep.py \
      --corpus-file data/opensubtitles_en.csv \
      --min-freq 1e-9 \
      --meco-csv data/meco_l2_texts.csv \
      --row 1 \
      --output output/highlight_mep.html
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np

import sys

# Ensure project root is importable
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.io import load_corpus
from src.masks import enumerate_masks, unpack_bits
from src.entropy import ilp_entropy


def extract_paragraph(meco_csv_path: Path, row_index_one_based: int) -> str:
    """Read the MECO CSV and return the text in the 'text' column for the row.

    Args:
        meco_csv_path: Path to the MECO CSV file.
        row_index_one_based: Row index starting at 1 (excluding header).
    """
    with meco_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            text_idx = header.index("text")
        except ValueError:
            text_idx = 1 if len(header) > 1 else 0

        for i, row in enumerate(reader, start=1):
            if i == row_index_one_based:
                return row[text_idx]
    raise IndexError(f"Row {row_index_one_based} not found in {meco_csv_path}")


def find_word_lengths_to_prepare(paragraph: str) -> list[int]:
    """Return sorted unique lengths for alphabetic tokens in paragraph."""
    words = re.findall(r"[A-Za-z]+", paragraph)
    return sorted(set(len(w.lower()) for w in words if w))


def build_corpus_index_for_lengths(
    corpus_df, word_lengths: Iterable[int]
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Build an index compatible with ilp_entropy for specific lengths."""
    index: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for L in sorted(set(word_lengths)):
        sub = corpus_df[corpus_df["word"].str.len() == L].copy()
        if sub.empty:
            continue
        codes = (
            sub["word"]
            .apply(lambda s: [ord(ch) - 97 for ch in s])
            .explode()
            .astype("uint8")
            .to_numpy()
            .reshape(-1, L)
        )
        freqs = sub["freq"].to_numpy(dtype="float32")
        index[L] = (codes, freqs)
    return index


def compute_min_entropy_position(word: str, corpus: dict, mask_cache: dict) -> int | None:
    """Return 0-based index of minimal-entropy position for a word, or None if unavailable."""
    L = len(word)
    if L not in corpus or L not in mask_cache:
        return None
    curve = ilp_entropy(word=word, drop_left=0.25, drop_right=0.25, corpus=corpus, mask_cache=mask_cache)
    if not curve:
        return None
    # curve positions are per index 0..L-1 (returned list matches that order)
    return int(np.argmin(np.asarray(curve)))


def compute_max_entropy_position(word: str, corpus: dict, mask_cache: dict) -> int | None:
    """Return 0-based index of maximal-entropy position for a word, or None if unavailable."""
    L = len(word)
    if L not in corpus or L not in mask_cache:
        return None
    curve = ilp_entropy(word=word, drop_left=0.25, drop_right=0.25, corpus=corpus, mask_cache=mask_cache)
    if not curve:
        return None
    return int(np.argmax(np.asarray(curve)))


def highlight_paragraph(paragraph: str, corpus: dict, mask_cache: dict, *, mode: str = "min") -> str:
    """Return HTML string for the paragraph with entropy positions highlighted.

    mode: "min" for minimal-entropy position, "max" for highest-entropy position.
    """
    # Replace alphabetic sequences with highlighted spans
    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        norm = token.lower()
        if mode == "max":
            pos = compute_max_entropy_position(norm, corpus, mask_cache)
        else:
            pos = compute_min_entropy_position(norm, corpus, mask_cache)
        if pos is None or pos < 0 or pos >= len(token):
            return token
        # Wrap the word and its parts to enable optional de-emphasis of neighbors
        pre = token[:pos]
        mid = token[pos:pos+1]
        post = token[pos+1:]
        pre_html = f"<span class=\"pre\">{pre}</span>" if pre else ""
        post_html = f"<span class=\"post\">{post}</span>" if post else ""
        return f"<span class=\"word\">{pre_html}<span class=\"mep\">{mid}</span>{post_html}</span>"

    highlighted = re.sub(r"[A-Za-z]+", repl, paragraph)
    return highlighted


def render_html(content_min: str, content_max: str) -> str:
    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Minimal Entropy Positions</title>
  <style>
    :root {{
      /* Defaults: bold + red, no highlight */
      --mep-color: #d00000;
      --mep-weight: 700;
      --mep-bg: transparent;
    }}
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; padding: 24px; }}
    .container {{ max-width: 900px; margin: 0 auto; }}
    p {{ font-size: 18px; }}
    .controls {{ display: flex; gap: 16px; align-items: center; flex-wrap: wrap; margin-bottom: 16px; color: #333; }}
    .controls label {{ display: inline-flex; align-items: center; gap: 6px; }}
    .mep {{
      color: var(--mep-color);
      font-weight: var(--mep-weight);
      background: var(--mep-bg);
      border-radius: 3px;
      padding: 0 1px;
      box-shadow: 0 0 0 1px rgba(0,0,0,0.05) inset;
      text-decoration: underline;
      text-decoration-color: var(--mep-color);
      text-decoration-thickness: 0.12em;
      text-underline-offset: 0.18em;
    }}
    /* De-emphasize neighbors when enabled */
    body[data-deemph="on"] .word .pre,
    body[data-deemph="on"] .word .post {{ opacity: .45; }}

    /* Invert highlighting: apply style to neighbors instead of MEP */
    body[data-invert="on"] .mep {{
      color: inherit;
      font-weight: inherit;
      background: transparent;
      text-decoration: none;
    }}
    body[data-invert="on"] .word .pre,
    body[data-invert="on"] .word .post {{
      color: var(--mep-color);
      font-weight: var(--mep-weight);
      background: var(--mep-bg);
      text-decoration: underline;
      text-decoration-color: var(--mep-color);
      text-decoration-thickness: 0.12em;
      text-underline-offset: 0.18em;
    }}
  </style>
  </head>
  <body>
    <div class=\"container\">
      <h2>Minimal Entropy Positions (drop_left = drop_right = 0.25)</h2>
      <div class=\"controls\">
        <label>Color <input id=\"mep-color\" type=\"color\" value=\"#d00000\" /></label>
        <label><input id=\"mep-bold\" type=\"checkbox\" checked /> Bold</label>
        <label><input id=\"mep-highlight\" type=\"checkbox\" /> Highlight</label>
        <label><input id=\"mep-deemph\" type=\"checkbox\" /> De-emphasize neighbors</label>
        <label><input id=\"mep-invert\" type=\"checkbox\" /> Invert</label>
      </div>
      <p>{content_min}</p>
      <h3 style=\"margin-top:28px;\">Highest Entropy Positions (counter-example)</h3>
      <p>{content_max}</p>
      <p style=\"margin-top:24px;color:#666;font-size:14px;\">Above: minimal-entropy position. Below: highest-entropy position.</p>
    </div>
    <script>
      (function() {{
        const root = document.documentElement;
        const body = document.body;
        const color = document.getElementById('mep-color');
        const bold = document.getElementById('mep-bold');
        const hl = document.getElementById('mep-highlight');
        const deemph = document.getElementById('mep-deemph');
        const invert = document.getElementById('mep-invert');
        function apply() {{
          root.style.setProperty('--mep-color', color.value || '#d00000');
          root.style.setProperty('--mep-weight', bold.checked ? '700' : '400');
          root.style.setProperty('--mep-bg', hl.checked ? '#ffe08a' : 'transparent');
          body.setAttribute('data-deemph', deemph.checked ? 'on' : 'off');
          body.setAttribute('data-invert', invert.checked ? 'on' : 'off');
        }}
        color.addEventListener('input', apply);
        bold.addEventListener('change', apply);
        hl.addEventListener('change', apply);
        deemph.addEventListener('change', apply);
        invert.addEventListener('change', apply);
        apply();
      }})();
    </script>
  </body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Highlight minimal-entropy positions in a paragraph.")
    parser.add_argument("--corpus-file", type=str, default="data/opensubtitles_en.csv")
    parser.add_argument("--min-freq", type=float, default=1e-9)
    parser.add_argument("--meco-csv", type=str, default="data/meco_l2_texts.csv")
    parser.add_argument("--row", type=int, default=1, help="1-based row index to select from MECO CSV")
    parser.add_argument("--output", type=str, default="output/highlight_mep.html")
    args = parser.parse_args()

    paragraph = extract_paragraph(Path(args.meco_csv), args.row)

    # Load corpus and prepare indices/masks only for needed lengths
    corpus_df, _ = load_corpus(args.corpus_file, min_freq=args.min_freq)
    lengths = find_word_lengths_to_prepare(paragraph)
    corpus_index = build_corpus_index_for_lengths(corpus_df, lengths)
    mask_cache = {L: unpack_bits(enumerate_masks(L), length=L) for L in lengths}

    highlighted_min = highlight_paragraph(paragraph, corpus_index, mask_cache, mode="min")
    highlighted_max = highlight_paragraph(paragraph, corpus_index, mask_cache, mode="max")
    html = render_html(highlighted_min, highlighted_max)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()



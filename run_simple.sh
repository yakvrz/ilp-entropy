#!/bin/bash

# This script executes a single ILP entropy calculation with a fixed set of parameters
# and then generates a visualization of the results.

set -e # Exit immediately if a command exits with a non-zero status.

# Run the calculation
echo "Starting simple run..."
python3 scripts/main.py \
    --corpus-file data/opensubtitles_en.csv \
    --all-corpus-words \
    --min-freq 1e-6 \
    --drop-left 0.1 \
    --drop-right 0.2 \
    --word-lengths 4 5 6 7 8 9 10
echo "Simple run completed."

# Find the most recent output directory
LATEST_RUN_DIR=$(ls -td output/run_* | head -n 1)

if [ -z "$LATEST_RUN_DIR" ]; then
    echo "Error: No output directory found. Cannot generate visualization."
    exit 1
fi

echo "Found latest run directory: $LATEST_RUN_DIR"

# Run the visualization script on the latest results
echo "Generating visualization..."
python3 visualization/plot_results.py "$LATEST_RUN_DIR"

echo "Simple run and visualization completed successfully." 
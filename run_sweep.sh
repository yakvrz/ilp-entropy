#!/bin/bash

# This script executes a full parameter sweep for the ILP entropy calculation
# and then generates a visualization of the results.

set -e # Exit immediately if a command exits with a non-zero status.

# Run the parameter sweep
echo "Starting parameter sweep..."
python3 scripts/main.py \
    --corpus-file data/opensubtitles_en.csv \
    --all-corpus-words \
    --min-freq 1e-6 \
    --sweep-left '0.1,0.9,0.1' \
    --sweep-right '0.1,0.9,0.1' \
    --word-lengths 4 5 6 7 8 9 10
echo "Parameter sweep completed."

# Find the most recent output directory
# 'ls -td' lists directories in reverse chronological order, and 'head -n 1' gets the first one.
LATEST_RUN_DIR=$(ls -td output/run_* | head -n 1)

if [ -z "$LATEST_RUN_DIR" ]; then
    echo "Error: No output directory found. Cannot generate visualization."
    exit 1
fi

echo "Found latest run directory: $LATEST_RUN_DIR"

# Run the visualization script on the latest results
echo "Generating visualization..."
python3 visualization/plot_results.py "$LATEST_RUN_DIR"

echo "Sweep and visualization completed successfully." 
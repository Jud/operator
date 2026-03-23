#!/usr/bin/env bash
# Setup venv and install dependencies for BERT disfluency benchmark.
# Usage: cd docs/hypotheses/bert-disfluency/working && bash setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d .venv ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Installing dependencies..."
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet transformers torch

echo ""
echo "Setup complete. Run the benchmark with:"
echo "  cd $SCRIPT_DIR"
echo "  .venv/bin/python classify_bench.py"

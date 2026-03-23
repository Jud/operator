#!/usr/bin/env bash
# Setup venv and install dependencies for BERT disfluency benchmark.
# Usage: cd docs/hypotheses/bert-disfluency/working && bash setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Find the main project root (works in both main checkout and worktrees)
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --path-format=absolute --git-common-dir 2>/dev/null | xargs dirname)"
cd "$SCRIPT_DIR"

# Reuse project-root .venv-poc if it exists, otherwise create a local .venv
if [ -d "$REPO_ROOT/.venv-poc" ]; then
    VENV="$REPO_ROOT/.venv-poc"
    echo "Reusing existing venv at $VENV"
elif [ -d .venv ]; then
    VENV="$SCRIPT_DIR/.venv"
else
    echo "Creating virtual environment..."
    python3 -m venv .venv
    VENV="$SCRIPT_DIR/.venv"
fi

echo "Installing dependencies..."
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet "transformers>=4.48.0" torch

echo ""
echo "Setup complete. Run the benchmark with:"
echo "  cd $SCRIPT_DIR"
echo "  $VENV/bin/python classify_bench.py"

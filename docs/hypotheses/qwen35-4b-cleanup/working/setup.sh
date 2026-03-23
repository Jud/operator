#!/usr/bin/env bash
# Setup script for Qwen3.5-4B STT cleanup hypothesis test.
# Creates a Python venv with mlx-lm and mlx-vlm installed.
# Idempotent — skips creation if venv already exists.
#
# Note: Qwen3.5-4B is a unified VLM with early fusion, so the
# mlx-community variant (Qwen3.5-4B-MLX-8bit) may require mlx-vlm.
# We install both mlx-lm and mlx-vlm to handle either loading path.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/python" ]; then
    echo "Venv already exists at $VENV_DIR — skipping creation."
else
    echo "Creating Python venv at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    echo "Venv created."
fi

echo "Installing mlx-lm and mlx-vlm..."
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet mlx-lm mlx-vlm
echo "Packages installed."

echo "Running smoke test (mlx_lm)..."
"$VENV_DIR/bin/python" -c "from mlx_lm import load; print('mlx_lm: ok')"
echo "Running smoke test (mlx_vlm)..."
"$VENV_DIR/bin/python" -c "from mlx_vlm import load; print('mlx_vlm: ok')"
echo "Setup complete."

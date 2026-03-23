#!/usr/bin/env bash
# Setup script for Qwen3.5-2B STT cleanup hypothesis test.
# Creates a Python venv with mlx-lm installed.
# Idempotent — skips creation if venv already exists.

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

echo "Installing mlx-lm..."
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet mlx-lm
echo "mlx-lm installed."

echo "Running smoke test..."
"$VENV_DIR/bin/python" -c "from mlx_lm import load; print('ok')"
echo "Setup complete."

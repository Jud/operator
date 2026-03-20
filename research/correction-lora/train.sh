#!/bin/bash
set -euo pipefail

# LoRA fine-tuning for speech correction on Qwen3-0.6B
#
# Prerequisites:
#   pip install mlx-lm
#
# Usage:
#   ./train.sh                    # Train with defaults
#   ./train.sh --iters 400        # More iterations
#   MODEL=Qwen/Qwen3-0.6B ./train.sh  # Different model

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
ADAPTER_DIR="$SCRIPT_DIR/adapters"

# Activate venv
source "$SCRIPT_DIR/.venv/bin/activate"

# Model — default to Qwen3-0.6B 4-bit from mlx-community
MODEL="${MODEL:-mlx-community/Qwen3-0.6B-4bit}"

# Training hyperparameters
ITERS="${ITERS:-200}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
LORA_LAYERS="${LORA_LAYERS:-16}"

echo "=== Correction LoRA Training ==="
echo "  Model:    $MODEL"
echo "  Data:     $DATA_DIR"
echo "  Adapters: $ADAPTER_DIR"
echo "  Iters:    $ITERS"
echo ""

# Check prerequisites
if ! command -v mlx_lm.lora &>/dev/null; then
    echo "ERROR: mlx_lm not installed. Run: pip install mlx-lm"
    exit 1
fi

if [ ! -f "$DATA_DIR/train.jsonl" ]; then
    echo "ERROR: No training data. Run generate-training-data.py first."
    exit 1
fi

TRAIN_COUNT=$(wc -l < "$DATA_DIR/train.jsonl" | tr -d ' ')
echo "  Training examples: $TRAIN_COUNT"

if [ -f "$DATA_DIR/valid.jsonl" ]; then
    VALID_COUNT=$(wc -l < "$DATA_DIR/valid.jsonl" | tr -d ' ')
    echo "  Validation examples: $VALID_COUNT"
fi
echo ""

# Train
mlx_lm.lora \
    --model "$MODEL" \
    --train \
    --data "$DATA_DIR" \
    --adapter-path "$ADAPTER_DIR" \
    --iters "$ITERS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --num-layers "$LORA_LAYERS" \
    --use-chat-template \
    --mask-prompt \
    "$@"

echo ""
echo "=== Training complete ==="
echo "  Adapter saved to: $ADAPTER_DIR"
echo ""
echo "Test with:"
echo "  mlx_lm.generate \\"
echo "    --model $MODEL \\"
echo "    --adapter-path $ADAPTER_DIR \\"
echo "    --prompt 'Correct: um so i was thinking about the uh the meeting'"

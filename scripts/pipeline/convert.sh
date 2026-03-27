#!/usr/bin/env bash
set -euo pipefail

# Convert any Qwen 3.5 model to CoreML.
#
# Usage:
#   ./scripts/pipeline/convert.sh Qwen/Qwen3.5-0.8B
#   ./scripts/pipeline/convert.sh Qwen/Qwen3.5-2B
#   ./scripts/pipeline/convert.sh Qwen/Qwen3.5-4B
#   ./scripts/pipeline/convert.sh Qwen/Qwen3.5-2B --output models/qwen2b

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-/tmp/coreml-venv/bin/python}"
MODEL_ID="${1:?Usage: $0 <model_id> [--output <dir>]}"

# Parse optional --output
OUTPUT_DIR=""
shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Default output dir: models/<slug>_fp16/
if [ -z "$OUTPUT_DIR" ]; then
    SLUG=$(echo "$MODEL_ID" | tr '/' '-' | tr '[:upper:]' '[:lower:]')
    OUTPUT_DIR="models/${SLUG}_fp16"
fi

DECODE_DIR="$OUTPUT_DIR"
PREFILL_DIR="${OUTPUT_DIR/monolith/batch_prefill}"
if [ "$DECODE_DIR" = "$PREFILL_DIR" ]; then
    PREFILL_DIR="${OUTPUT_DIR}_prefill"
fi

echo "═══════════════════════════════════════════════"
echo "  Converting: $MODEL_ID"
echo "  Decode:     $DECODE_DIR"
echo "  Prefill:    $PREFILL_DIR"
echo "═══════════════════════════════════════════════"

# Step 1: Convert decode model
echo ""
echo "==> Step 1/3: Convert decode model..."
$PYTHON "$SCRIPT_DIR/convert_decode.py" "$MODEL_ID" --output "$DECODE_DIR" --fp16

# Step 2: Convert prefill model
echo ""
echo "==> Step 2/3: Convert prefill model..."
$PYTHON "$SCRIPT_DIR/convert_prefill.py" "$MODEL_ID" --output "$PREFILL_DIR" --fp16

# Step 3: Generate prompt cache
echo ""
echo "==> Step 3/3: Generate prompt cache..."
$PYTHON "$SCRIPT_DIR/gen_prompt_cache.py" --model-id "$MODEL_ID" --output "$DECODE_DIR/prompt_cache/"

# Copy tokenizer
echo ""
echo "==> Copying tokenizer..."
mkdir -p "$DECODE_DIR/tokenizer"
$PYTHON -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('$MODEL_ID')
tok.save_pretrained('$DECODE_DIR/tokenizer/')
print(f'Saved tokenizer to $DECODE_DIR/tokenizer/')
"

# Summary
echo ""
echo "═══════════════════════════════════════════════"
echo "  Done!"
echo "  Decode:     $(du -sh "$DECODE_DIR" | cut -f1) at $DECODE_DIR"
echo "  Prefill:    $(du -sh "$PREFILL_DIR" | cut -f1) at $PREFILL_DIR"
echo ""
echo "  Run with:"
echo "    swift run -c release CleanupCLI --model-dir $DECODE_DIR --prefill-dir $PREFILL_DIR"
echo "═══════════════════════════════════════════════"

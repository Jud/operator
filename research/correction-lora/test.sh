#!/bin/bash
set -euo pipefail

# Test the fine-tuned correction model
#
# Usage:
#   ./test.sh                           # Run default test cases
#   ./test.sh "your test transcription"  # Test a specific input

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ADAPTER_DIR="$SCRIPT_DIR/adapters"
MODEL="${MODEL:-mlx-community/Qwen3-0.6B-4bit}"

# Activate venv
source "$SCRIPT_DIR/.venv/bin/activate"

if [ ! -d "$ADAPTER_DIR" ]; then
    echo "ERROR: No adapter found at $ADAPTER_DIR. Run train.sh first."
    exit 1
fi

run_test() {
    local input="$1"
    echo "INPUT:  \"$input\""

    # Build the chat-template prompt manually for Qwen
    local prompt="<|im_start|>system
Correct the speech transcription. Output only the corrected text.<|im_end|>
<|im_start|>user
${input}<|im_end|>
<|im_start|>assistant
"

    local output
    output=$(mlx_lm.generate \
        --model "$MODEL" \
        --adapter-path "$ADAPTER_DIR" \
        --prompt "$prompt" \
        --max-tokens 128 \
        --temp 0.0 2>/dev/null | tail -1)

    echo "OUTPUT: \"$output\""
    echo ""
}

echo "=== Correction Model Test ==="
echo "  Model:   $MODEL"
echo "  Adapter: $ADAPTER_DIR"
echo ""

if [ $# -gt 0 ]; then
    run_test "$*"
    exit 0
fi

# Default test cases
run_test "um so i was i was thinking about uh the the meeting"
run_test "yeah so like we need to to fix the the bug in the uh authentication module"
run_test "can you uh list all the files in the current directory"
run_test "i think we should we should probably um deploy this to staging first"
run_test "the the CI run failed again"

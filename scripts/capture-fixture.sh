#!/bin/bash
set -euo pipefail

# Capture a transcription test fixture from an audio trace.
#
# Copies the WAV file to the test-fixtures directory and generates
# the ground truth transcription via batch decode.
#
# Usage:
#   ./scripts/capture-fixture.sh <name> [wav-path]
#
# If wav-path is omitted, uses the most recent audio trace.
#
# Examples:
#   ./scripts/capture-fixture.sh timestamp-drift
#   ./scripts/capture-fixture.sh short-utterance ~/path/to/audio.wav

FIXTURE_DIR="$HOME/Library/Application Support/Operator/test-fixtures"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <name> [wav-path]"
    echo ""
    echo "Captures a WAV file as a transcription test fixture."
    echo "If wav-path is omitted, uses the most recent audio trace."
    echo ""
    echo "Existing fixtures:"
    if [ -d "$FIXTURE_DIR" ]; then
        ls "$FIXTURE_DIR"/*.wav 2>/dev/null | while read f; do
            basename "$f" .wav
        done
    fi
    exit 1
fi

NAME="$1"
WAV_PATH="${2:-}"

if [ -z "$WAV_PATH" ]; then
    TRACE_DIR="$HOME/Library/Application Support/Operator/audio-traces"
    WAV_PATH=$(ls -t "$TRACE_DIR"/*.wav 2>/dev/null | head -1)
    if [ -z "$WAV_PATH" ]; then
        echo "ERROR: No audio traces found"
        exit 1
    fi
    echo "Using latest trace: $(basename "$WAV_PATH")"
fi

if [ ! -f "$WAV_PATH" ]; then
    echo "ERROR: File not found: $WAV_PATH"
    exit 1
fi

DEST="$FIXTURE_DIR/$NAME.wav"
EXPECTED="$FIXTURE_DIR/$NAME.expected.txt"
METADATA="$FIXTURE_DIR/$NAME.meta.json"

cp "$WAV_PATH" "$DEST"
echo "Copied: $DEST"

# Record commit and timestamp
COMMIT=$(git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null || echo "unknown")
BRANCH=$(git -C "$(dirname "$0")/.." rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
CAPTURED_AT=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
AUDIO_DURATION=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$DEST" 2>/dev/null || echo "unknown")

cat > "$METADATA" <<METAEOF
{
    "commit": "$COMMIT",
    "branch": "$BRANCH",
    "capturedAt": "$CAPTURED_AT",
    "sourceFile": "$(basename "$WAV_PATH")",
    "audioDuration": $AUDIO_DURATION
}
METAEOF
echo "Metadata: $METADATA"

# Generate ground truth via batch decode
echo "Generating ground truth..."
RESULT=$(OPERATOR_STT_WAV="$DEST" ./scripts/run-benchmarks.sh --no-build stt-replay-batch 2>&1)
TEXT=$(echo "$RESULT" | grep "Text:" | sed 's/.*Text: "//' | sed 's/"$//')

if [ -z "$TEXT" ]; then
    echo "WARNING: Batch decode returned empty text"
    echo "(empty)" > "$EXPECTED"
else
    echo "$TEXT" > "$EXPECTED"
    echo "Ground truth: \"$(head -c 100 "$EXPECTED")...\""
fi

echo ""
echo "Fixture saved:"
echo "  WAV:      $DEST"
echo "  Expected: $EXPECTED"
echo ""
echo "Run transcription tests with: make test-transcription"

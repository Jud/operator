#!/bin/bash
set -euo pipefail

# Watch transcription quality logs and auto-capture fixtures when quality drops.
#
# Monitors the TranscriptionQuality log category for warnings (< 85% match),
# finds the corresponding audio trace, and captures it as a test fixture.
#
# Usage:
#   ./scripts/watch-quality.sh           # Watch and auto-capture
#   ./scripts/watch-quality.sh --dry-run # Watch only, don't capture

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TRACE_DIR="$HOME/Library/Application Support/Operator/audio-traces"
FIXTURE_DIR="$HOME/Library/Application Support/Operator/test-fixtures"
DRY_RUN="${1:-}"

echo "=== Transcription Quality Watcher ==="
echo "  Watching for quality drops..."
echo "  Audio traces: $TRACE_DIR"
echo "  Fixtures: $FIXTURE_DIR"
echo ""

# Stream quality logs from Operator
/usr/bin/log stream \
    --predicate 'subsystem == "com.operator.app" AND category == "TranscriptionQuality"' \
    --level info 2>/dev/null | while IFS= read -r line; do

    # Print all quality lines
    if echo "$line" | grep -q "Quality:"; then
        echo "$line"
    fi

    # Act on warnings (< 85% match)
    if echo "$line" | grep -q "⚠️ Quality:"; then
        # Extract the percentage
        pct=$(echo "$line" | grep -oE '[0-9]+% match' | grep -oE '[0-9]+')

        # Find the most recent audio trace (the one that just finished)
        latest_trace=$(ls -t "$TRACE_DIR"/*.wav 2>/dev/null | head -1)

        if [ -z "$latest_trace" ]; then
            echo "  ⚠ No audio trace found to capture"
            continue
        fi

        # Generate a fixture name from timestamp and percentage
        trace_name=$(basename "$latest_trace" .wav)
        fixture_name="auto-${pct}pct-${trace_name}"

        # Check if this trace is already captured
        if [ -f "$FIXTURE_DIR/$fixture_name.wav" ]; then
            echo "  Already captured: $fixture_name"
            continue
        fi

        if [ "$DRY_RUN" = "--dry-run" ]; then
            echo "  [DRY RUN] Would capture: $fixture_name"
            echo "    Source: $latest_trace"
        else
            echo "  Capturing fixture: $fixture_name"
            cd "$PROJECT_DIR" && ./scripts/capture-fixture.sh "$fixture_name" "$latest_trace" 2>&1 | sed 's/^/    /'
            echo ""
            echo "  Run tests: make test-transcription"
        fi
    fi
done

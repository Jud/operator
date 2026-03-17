#!/bin/bash
set -euo pipefail

APP_PATH="$(cd "$(dirname "$0")/.." && pwd)/build/Operator.app"

# Build
echo "Building..."
cd "$(dirname "$0")/.."
make app

# Kill existing
echo "Stopping Operator..."
pkill -9 -f 'Operator.app/Contents/MacOS/Operator' 2>/dev/null || true
sleep 1

# Verify dead
if pgrep -f 'Operator.app/Contents/MacOS/Operator' >/dev/null 2>&1; then
    echo "ERROR: Operator still running after kill"
    pgrep -af 'Operator.app/Contents/MacOS/Operator'
    exit 1
fi

# Launch
echo "Starting $APP_PATH"
open "$APP_PATH"
sleep 1

# Verify running
PID=$(pgrep -f 'Operator.app/Contents/MacOS/Operator' || true)
if [[ -z "$PID" ]]; then
    echo "ERROR: Operator failed to start"
    exit 1
fi

echo "Operator running (PID $PID)"

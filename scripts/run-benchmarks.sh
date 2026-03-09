#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_DIR="$ROOT_DIR/Operator"
CONFIG="debug"
NO_BUILD=0
FORCE_METALLIB=0

print_usage() {
    cat <<'EOF'
Usage: ./scripts/run-benchmarks.sh [--debug|--release] [--no-build] [--force-metallib] [target...]

This runner does not require `make app` or a macOS `.app` bundle.

Targets:
  all               Run every benchmark suite
  routing           Run routing latency and routing accuracy benchmarks
  routing-latency   Run only routing latency benchmark
  routing-accuracy  Run only routing accuracy benchmark
  tts               Run TTS time-to-first-audio benchmark
  tts-ttfa          Run only TTS time-to-first-audio benchmark
  stt               Run STT latency, long-utterance, and streaming benchmarks
  stt-latency       Run only STT 5s transcription benchmark
  stt-long          Run only STT long-utterance benchmark
  stt-streaming     Run only STT streaming/finalize benchmark
  stt-soak          Run a long paced STT soak benchmark (use OPERATOR_STT_SOAK_MINUTES)
  memory            Run routing-model memory benchmark
  list              Show this target list

Options:
  --debug           Use the debug build (default)
  --release         Use the release build
  --no-build        Reuse the existing binary and metallib for faster reruns
  --force-metallib  Rebuild mlx.metallib even if the source hash matches
EOF
}

TARGET_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)
            CONFIG="debug"
            ;;
        --release)
            CONFIG="release"
            ;;
        --no-build)
            NO_BUILD=1
            ;;
        --force-metallib)
            FORCE_METALLIB=1
            ;;
        list|help|--help|-h)
            print_usage
            exit 0
            ;;
        *)
            TARGET_ARGS+=("$1")
            ;;
    esac
    shift
done

if ! xcrun --find metal >/dev/null 2>&1; then
    echo "The Metal toolchain is not available via xcrun." >&2
    echo "Install it with: xcodebuild -downloadComponent MetalToolchain" >&2
    exit 1
fi

cd "$PACKAGE_DIR"

if [[ "$NO_BUILD" != "1" ]]; then
    swift build --product Benchmarks -c "$CONFIG"
fi

METALLIB_ARGS=("$CONFIG")
if [[ "$FORCE_METALLIB" == "1" ]]; then
    METALLIB_ARGS+=("--force")
fi
"$ROOT_DIR/scripts/build-mlx-metallib.sh" "${METALLIB_ARGS[@]}"

EXECUTABLE="$PACKAGE_DIR/.build/arm64-apple-macosx/$CONFIG/Benchmarks"
if [[ ! -x "$EXECUTABLE" ]]; then
    EXECUTABLE="$(find "$PACKAGE_DIR/.build" -maxdepth 4 -type f -path "*/$CONFIG/Benchmarks" | head -n 1 || true)"
fi
if [[ -z "$EXECUTABLE" || ! -x "$EXECUTABLE" ]]; then
    echo "error: failed to locate Benchmarks executable for config=$CONFIG" >&2
    exit 1
fi

exec env OPERATOR_BENCHMARK_RUNNER=1 "$EXECUTABLE" "${TARGET_ARGS[@]}"

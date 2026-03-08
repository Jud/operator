#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./scripts/build-mlx-metallib.sh [debug|release] [--force]

Builds MLX's Metal shader library and places `mlx.metallib` next to the
SwiftPM-built binaries under `Operator/.build/.../<config>/`.
No `.app` bundle is required.

Examples:
  ./scripts/build-mlx-metallib.sh debug
  ./scripts/build-mlx-metallib.sh release --force
EOF
}

FORCE=0
CONFIG="debug"

for arg in "$@"; do
    case "$arg" in
        debug|release)
            CONFIG="$arg"
            ;;
        --force)
            FORCE=1
            ;;
        -h|--help|help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            echo "" >&2
            usage >&2
            exit 2
            ;;
    esac
done

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_DIR="$ROOT_DIR/Operator"
BUILD_DIR="$PACKAGE_DIR/.build"
MLX_SWIFT_DIR="$BUILD_DIR/checkouts/mlx-swift"
KERNELS_DIR="$MLX_SWIFT_DIR/Source/Cmlx/mlx/mlx/backend/metal/kernels"
TOOLCHAIN_NAME="${TOOLCHAINS:-}"

if [[ ! -d "$BUILD_DIR" ]]; then
    echo "error: $BUILD_DIR not found (run swift build first)" >&2
    exit 1
fi

if [[ ! -d "$KERNELS_DIR" ]]; then
    echo "error: MLX kernels dir not found at $KERNELS_DIR" >&2
    exit 1
fi

OUT_DIR="$BUILD_DIR/arm64-apple-macosx/$CONFIG"
if [[ ! -d "$OUT_DIR" ]]; then
    OUT_DIR="$(find "$BUILD_DIR" -maxdepth 4 -type d -path "*/$CONFIG" | head -n 1 || true)"
fi
if [[ -z "$OUT_DIR" || ! -d "$OUT_DIR" ]]; then
    echo "error: failed to locate build output directory for config=$CONFIG" >&2
    exit 1
fi

METAL_SRCS=()
while IFS= read -r line; do
    METAL_SRCS+=("$line")
done < <(find "$KERNELS_DIR" -type f -name '*.metal' ! -name '*_nax.metal' | LC_ALL=C sort)

if [[ "${#METAL_SRCS[@]}" -eq 0 ]]; then
    echo "error: no Metal sources found under $KERNELS_DIR" >&2
    exit 1
fi

OUT_METALLIB="$OUT_DIR/mlx.metallib"
OUT_DEFAULT_METALLIB="$OUT_DIR/default.metallib"
HASH_FILE="$OUT_DIR/.mlx.metallib.sha"
XCODE_DEFAULT_METALLIB="$PACKAGE_DIR/.build/xcode/Build/Products/$(tr '[:lower:]' '[:upper:]' <<< "${CONFIG:0:1}")${CONFIG:1}/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
XCODE_RELEASE_METALLIB="$PACKAGE_DIR/.build/xcode-release/Build/Products/Release/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
CURRENT_HASH="$(
    find "$KERNELS_DIR" -type f \( -name '*.metal' -o -name '*.h' \) ! -name '*_nax.metal' \
        | LC_ALL=C sort \
        | xargs cat \
        | shasum -a 256 \
        | awk '{print $1}'
)"

if [[ "$FORCE" != "1" && -f "$OUT_METALLIB" && -f "$HASH_FILE" ]]; then
    PREV_HASH="$(cat "$HASH_FILE" 2>/dev/null || true)"
    if [[ "$CURRENT_HASH" == "$PREV_HASH" ]]; then
        echo "mlx.metallib is up to date at $OUT_METALLIB"
        if [[ ! -f "$OUT_DEFAULT_METALLIB" ]]; then
            cp "$OUT_METALLIB" "$OUT_DEFAULT_METALLIB"
        fi
        exit 0
    fi
fi

if [[ "$FORCE" != "1" ]]; then
    CANDIDATE_METALLIB=""
    if [[ "$CONFIG" == "debug" && -f "$XCODE_DEFAULT_METALLIB" ]]; then
        CANDIDATE_METALLIB="$XCODE_DEFAULT_METALLIB"
    elif [[ "$CONFIG" == "release" && -f "$XCODE_RELEASE_METALLIB" ]]; then
        CANDIDATE_METALLIB="$XCODE_RELEASE_METALLIB"
    fi

    if [[ -n "$CANDIDATE_METALLIB" ]]; then
        cp "$CANDIDATE_METALLIB" "$OUT_METALLIB"
        cp "$CANDIDATE_METALLIB" "$OUT_DEFAULT_METALLIB"
        printf '%s' "$CURRENT_HASH" > "$HASH_FILE"
        echo "Reused Xcode-built default.metallib from $CANDIDATE_METALLIB"
        exit 0
    fi
fi

TMPDIR_ROOT="${TMPDIR:-/tmp}"
TMP_DIR="$(mktemp -d "$TMPDIR_ROOT/operator-mlx-metallib.XXXXXX")"
cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

XCRUN=(xcrun -sdk macosx)
if [[ -n "$TOOLCHAIN_NAME" ]]; then
    XCRUN=(xcrun --toolchain "$TOOLCHAIN_NAME" -sdk macosx)
fi

AIR_FILES=()
METAL_FLAGS=(
    -x metal
    -Wall
    -Wextra
    -fno-fast-math
    -Wno-c++17-extensions
    -Wno-c++20-extensions
)

echo "Compiling ${#METAL_SRCS[@]} Metal sources for $CONFIG..."
for src in "${METAL_SRCS[@]}"; do
    rel="${src#"$KERNELS_DIR/"}"
    key="$(printf '%s' "$rel" | shasum -a 256 | awk '{print $1}' | cut -c1-16)"
    out_air="$TMP_DIR/$key.air"

    if ! "${XCRUN[@]}" metal "${METAL_FLAGS[@]}" -c "$src" -I"$KERNELS_DIR" -I"$MLX_SWIFT_DIR/Source/Cmlx/mlx" -o "$out_air" 2>"$TMP_DIR/metal.err"; then
        cat "$TMP_DIR/metal.err" >&2
        exit 1
    fi

    AIR_FILES+=("$out_air")
done

echo "Linking mlx.metallib -> $OUT_METALLIB"
"${XCRUN[@]}" metallib "${AIR_FILES[@]}" -o "$OUT_METALLIB"
cp "$OUT_METALLIB" "$OUT_DEFAULT_METALLIB"
printf '%s' "$CURRENT_HASH" > "$HASH_FILE"

echo "OK: wrote $OUT_METALLIB"

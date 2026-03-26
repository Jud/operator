#!/usr/bin/env bash
set -euo pipefail

# Build the Rust qwen-tokenizer crate and generate Swift/FFI bindings.
# Produces an XCFramework for SPM integration.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUST_DIR="$PROJECT_ROOT/rust/qwen-tokenizer"
SPM_SWIFT="$PROJECT_ROOT/Operator/Sources/QwenTokenizerRust"
XCFRAMEWORK="$PROJECT_ROOT/Operator/Frameworks/QwenTokenizerFFI.xcframework"

echo "==> Building Rust crate (release)..."
cd "$RUST_DIR"
cargo build --release

echo "==> Generating Swift bindings..."
cargo run --release --bin uniffi-bindgen generate \
    --library target/release/libqwen_tokenizer.a \
    --language swift \
    --out-dir generated

echo "==> Copying Swift bindings..."
cp generated/qwen_tokenizer.swift "$SPM_SWIFT/"

echo "==> Creating XCFramework..."
rm -rf "$XCFRAMEWORK"
mkdir -p "$(dirname "$XCFRAMEWORK")"

# Prepare headers directory for XCFramework
HEADERS_DIR="$RUST_DIR/target/xcframework-headers"
rm -rf "$HEADERS_DIR"
mkdir -p "$HEADERS_DIR"
cp generated/qwen_tokenizerFFI.h "$HEADERS_DIR/"
cp generated/qwen_tokenizerFFI.modulemap "$HEADERS_DIR/module.modulemap"

xcodebuild -create-xcframework \
    -library target/release/libqwen_tokenizer.a \
    -headers "$HEADERS_DIR" \
    -output "$XCFRAMEWORK"

LIB_SIZE="$(du -h target/release/libqwen_tokenizer.a | cut -f1)"
echo ""
echo "Done."
echo "  XCFramework: $XCFRAMEWORK"
echo "  Static lib size: $LIB_SIZE"

#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Full verify: MCP Server ==="
pushd "$ROOT_DIR/mcp-server" >/dev/null
npx tsc --noEmit
npx eslint 'src/**/*.ts'
npx prettier --check 'src/**/*.ts'
popd >/dev/null

echo "=== Full verify: Swift ==="
pushd "$ROOT_DIR/Operator" >/dev/null
swift build
swift test
swiftlint lint --strict
swift-format lint --strict --recursive Sources Tests
popd >/dev/null

echo "=== Full verify passed ==="

#!/bin/bash
set -e

# operator-setup: Register Operator MCP server with all Claude Code accounts.
# Auto-detects Claude Code config directories by fingerprint.

MCP_CMD="${OPERATOR_MCP_CMD:-operator-mcp}"
TOKEN_DIR="$HOME/.operator"
REGISTERED=0
SKIPPED=0

# Resolve MCP command to absolute path
if ! command -v "$MCP_CMD" &>/dev/null; then
  # Try common Homebrew locations
  for candidate in /opt/homebrew/bin/operator-mcp /usr/local/bin/operator-mcp; do
    if [ -x "$candidate" ]; then
      MCP_CMD="$candidate"
      break
    fi
  done
fi

MCP_CMD="$(command -v "$MCP_CMD" 2>/dev/null || echo "$MCP_CMD")"

if [ ! -x "$MCP_CMD" ]; then
  echo "Error: operator-mcp not found. Is Operator installed?"
  exit 1
fi

# Generate auth token if needed
if [ ! -f "$TOKEN_DIR/token" ]; then
  mkdir -p "$TOKEN_DIR"
  chmod 700 "$TOKEN_DIR"
  openssl rand -hex 32 > "$TOKEN_DIR/token"
  chmod 600 "$TOKEN_DIR/token"
  echo "Generated auth token at $TOKEN_DIR/token"
fi

# Find all Claude Code config directories by scanning for .claude.json
# files that contain "numStartups" (Claude Code fingerprint).
CONFIG_DIRS=()
while IFS= read -r -d '' config_file; do
  dir="$(dirname "$config_file")"
  if grep -q '"numStartups"' "$config_file" 2>/dev/null; then
    CONFIG_DIRS+=("$dir")
  fi
done < <(find "$HOME" -maxdepth 2 -name ".claude.json" -print0 2>/dev/null)

if [ ${#CONFIG_DIRS[@]} -eq 0 ]; then
  echo "No Claude Code configs found."
  echo "Run Claude Code at least once, then re-run: operator-setup"
  exit 1
fi

echo "Found ${#CONFIG_DIRS[@]} Claude Code config(s):"
for d in "${CONFIG_DIRS[@]}"; do
  echo "  $d"
done
echo ""

# Register MCP server in each config
for d in "${CONFIG_DIRS[@]}"; do
  CONFIG_FILE="$d/.claude.json"
  LABEL="$(basename "$d")"

  # Check if already registered with correct command
  if python3 -c "
import json, sys
with open('$CONFIG_FILE') as f:
    d = json.load(f)
s = d.get('mcpServers', {}).get('operator', {})
if s.get('command') == '$MCP_CMD':
    sys.exit(0)
sys.exit(1)
" 2>/dev/null; then
    echo "  $LABEL: already registered ✓"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  # Add/update MCP server entry
  python3 -c "
import json
with open('$CONFIG_FILE') as f:
    d = json.load(f)
servers = d.setdefault('mcpServers', {})
servers['operator'] = {
    'type': 'stdio',
    'command': '$MCP_CMD',
    'args': [],
    'env': {}
}
with open('$CONFIG_FILE', 'w') as f:
    json.dump(d, f, indent=2)
"

  if [ $? -eq 0 ]; then
    echo "  $LABEL: registered ✓"
    REGISTERED=$((REGISTERED + 1))
  else
    echo "  $LABEL: failed ✗"
  fi
done

echo ""
if [ $REGISTERED -gt 0 ]; then
  echo "Registered Operator MCP in $REGISTERED config(s)."
fi
if [ $SKIPPED -gt 0 ]; then
  echo "$SKIPPED config(s) already up to date."
fi
echo ""
echo "Start the daemon: brew services start operator"

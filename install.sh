#!/bin/bash
set -euo pipefail

# Operator installer - works with or without Homebrew
# Usage: curl -fsSL https://raw.githubusercontent.com/Jud/operator/main/install.sh | bash

REPO="Jud/operator"
INSTALL_DIR="${INSTALL_DIR:-/usr/local}"
CONFIG_DIR="$HOME/.operator"

info()  { echo "==> $*"; }
warn()  { echo "==> WARNING: $*" >&2; }
error() { echo "==> ERROR: $*" >&2; exit 1; }

# Check prerequisites
check_prereqs() {
    [[ "$(uname)" == "Darwin" ]] || error "Operator requires macOS"

    sw_vers -productVersion | grep -qE '^1[5-9]\.' || \
        error "Operator requires macOS 15 (Sequoia) or later"

    command -v swift >/dev/null || error "Swift toolchain required. Install Xcode or Xcode Command Line Tools."
    command -v node >/dev/null  || error "Node.js required. Install via: brew install node"
    command -v npm >/dev/null   || error "npm required. Install via: brew install node"
}

# Try Homebrew first
try_brew() {
    if command -v brew >/dev/null; then
        info "Homebrew detected. Installing via brew..."
        brew tap jud/plugins https://github.com/Jud/operator.git 2>/dev/null || true
        brew install --HEAD operator
        return 0
    fi
    return 1
}

# Manual install from source
install_from_source() {
    local tmpdir
    tmpdir=$(mktemp -d)
    trap 'rm -rf "$tmpdir"' EXIT

    info "Cloning $REPO..."
    git clone --depth 1 "https://github.com/$REPO.git" "$tmpdir/operator"

    info "Building Swift daemon..."
    cd "$tmpdir/operator/Operator"
    swift build -c release --disable-sandbox
    cp .build/release/Operator "$INSTALL_DIR/bin/operator-daemon"

    info "Building MCP server..."
    cd "$tmpdir/operator/mcp-server"
    npm ci --ignore-scripts
    npm run build

    local mcp_dir="$INSTALL_DIR/lib/operator-mcp"
    mkdir -p "$mcp_dir"
    cp -r build package.json node_modules "$mcp_dir/"

    cat > "$INSTALL_DIR/bin/operator-mcp" <<WRAPPER
#!/bin/bash
exec node "$mcp_dir/build/index.js" "\$@"
WRAPPER
    chmod +x "$INSTALL_DIR/bin/operator-mcp"

    info "Binaries installed to $INSTALL_DIR/bin/"
}

# Set up config and auth token
setup_config() {
    mkdir -p "$CONFIG_DIR"
    chmod 700 "$CONFIG_DIR"

    if [[ ! -f "$CONFIG_DIR/token" ]]; then
        openssl rand -hex 32 > "$CONFIG_DIR/token"
        chmod 600 "$CONFIG_DIR/token"
        info "Auth token generated at $CONFIG_DIR/token"
    fi
}

# Register MCP server with Claude Code
register_mcp() {
    if command -v claude >/dev/null; then
        local mcp_bin
        mcp_bin=$(command -v operator-mcp)
        claude mcp add operator --transport stdio -- "$mcp_bin" && \
            info "MCP server registered with Claude Code" || \
            warn "Could not auto-register MCP server. Run: claude mcp add operator -- $mcp_bin"
    else
        warn "Claude Code CLI not found. Register manually after installing Claude Code:"
        echo "  claude mcp add operator -- $(command -v operator-mcp || echo '/usr/local/bin/operator-mcp')"
    fi
}

main() {
    info "Installing Operator..."
    check_prereqs
    try_brew || install_from_source
    setup_config
    register_mcp

    echo ""
    info "Installation complete!"
    echo ""
    echo "  Start the daemon:  operator-daemon"
    echo "  MCP server:        auto-registered with Claude Code"
    echo "  Config:            ~/.operator/"
    echo ""
}

main "$@"

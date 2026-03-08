#!/bin/bash
set -euo pipefail

# Operator installer
# Usage: curl -fsSL https://raw.githubusercontent.com/Jud/operator/main/install.sh | bash

REPO="Jud/operator"

info()  { echo "==> $*"; }
error() { echo "==> ERROR: $*" >&2; exit 1; }

# Check prerequisites
check_prereqs() {
    [[ "$(uname)" == "Darwin" ]] || error "Operator requires macOS"

    sw_vers -productVersion | grep -qE '^1[5-9]\.' || \
        error "Operator requires macOS 15 (Sequoia) or later"
}

main() {
    info "Installing Operator..."
    check_prereqs

    if command -v brew >/dev/null; then
        info "Installing via Homebrew Cask..."
        brew tap jud/operator https://github.com/Jud/homebrew-operator.git 2>/dev/null || true
        brew install --cask operator
    else
        error "Homebrew required. Install from https://brew.sh then re-run."
    fi

    echo ""
    info "Installation complete!"
    echo ""
    echo "  Launch Operator from Applications or Spotlight."
    echo "  It runs as a menu bar app (no Dock icon)."
    echo "  The Claude Code plugin is registered automatically on launch."
    echo ""
}

main "$@"

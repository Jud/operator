# Operator

Voice-first orchestration layer for multiple concurrent Claude Code sessions on macOS.

See [ARCHITECTURE.md](ARCHITECTURE.md) for full system design and [README.md](README.md) for user-facing docs.

## Build & Test

```bash
# Build Operator.app (Swift daemon + MCP server)
make app

# Create distributable zip
make zip

# Swift only
cd Operator && swift build
cd Operator && swift test

# MCP server only
cd mcp-server && npm ci && npm run build
cd mcp-server && npm test

# E2E (requires real iTerm)
cd Operator && swift run E2ETests

# Benchmarks
# Does not require `make app` or a `.app` bundle.
./scripts/run-benchmarks.sh list
./scripts/run-benchmarks.sh routing
./scripts/run-benchmarks.sh routing-latency
./scripts/run-benchmarks.sh --no-build routing-latency
./scripts/run-benchmarks.sh --release tts
./scripts/run-benchmarks.sh stt-long
make bench-routing
make bench BENCH_ARGS="--no-build routing-latency"
```

## Install

```bash
# Via Homebrew Cask
brew tap jud/operator https://github.com/Jud/homebrew-operator.git
brew install --cask operator

# Launch Operator — it auto-registers the Claude Code plugin on startup
```

## Key Design Decisions

- **MCP server is intentionally minimal** — just the `speak` tool (~60 lines). Registration is handled by iTerm discovery polling + Claude Code hooks, NOT by MCP tools. The `register`, `update_context`, and `get_state` tools were intentionally removed (commit `aefb8ba`).
- **Sessions keyed by TTY** — TTY paths are globally unique and stable. More reliable than PIDs or session names.
- **No API keys** — routing uses `claude -p` (print mode) which piggybacks on the user's existing Claude CLI auth.
- **File-based JXA delivery** — message text goes through a temp file to avoid shell + JavaScript double-escaping.

## Project Structure

```
Operator/Sources/    # Swift daemon
├── App/             # SwiftUI entry, AppDelegate bootstrap
├── Bridge/          # ITermBridge (JXA delivery)
├── Router/          # MessageRouter, ClaudePipe, InterruptionHandler
├── Server/          # HTTP server (Hummingbird, localhost:7420)
├── Shared/          # Protocols, Log factory
├── State/           # StateMachine, SessionRegistry, DiscoveryService
├── Trigger/         # FNKeyTrigger (push-to-talk)
├── UI/              # WaveformPanel, SettingsView
└── Voice/           # SpeechTranscriber, SpeechManager, AudioQueue

mcp-server/src/      # MCP server (TypeScript, ~60 lines)
└── index.ts         # Single `speak` tool
```

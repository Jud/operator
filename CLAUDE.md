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
./scripts/verify-all.sh
make verify

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
OPERATOR_STT_SOAK_MINUTES=5 ./scripts/run-benchmarks.sh stt-soak
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

## Swift Style Rules

All Swift code must pass both SwiftLint (`--strict`) and swift-format (`--strict`) with zero violations. The pre-commit hook and CI enforce this.

**Key rules to follow when generating Swift code:**
- **Access control on all top-level declarations** — use `private` for file-scoped helpers, `internal` or `public` for API surface. Never leave top-level `let`, `func`, `class`, `enum`, or `struct` without an access level keyword.
- **No `try!` or force unwraps (`!`)** — use `try` with `throws` propagation, or `guard let ... else` with meaningful errors. Even in CLI tools.
- **No semicolons** — one statement per line.
- **One variable per declaration** — `var a = 1; var b = 2`, not `var a = 1, b = 2`.
- **Number separators** — use underscores for readability: `1_000`, `6_144`, `248_044`.
- **Line length ≤ 120 chars** (warning), **≤ 150** (error). Break long strings into concatenated parts.
- **Function body ≤ 75 lines**, **cyclomatic complexity ≤ 10** — extract helpers when functions grow.
- **Identifier names ≥ 2 chars** — single-letter names (`i`, `j`, `x`, `y`, `n`, etc.) are excluded by config but prefer descriptive names where clarity helps.
- **Blank line before variable declarations** that follow non-declaration statements.

**Excluded from linting** (generated/vendored code):
- `Sources/Tokenizer/` — vendored HuggingFace swift-transformers
- `Sources/QwenTokenizerRust/` — UniFFI-generated Rust bindings

**Run locally:** `cd Operator && swiftlint lint --strict --quiet && swift-format lint --strict --recursive Sources Tests`

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

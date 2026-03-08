# Architecture

Operator has two components: a Swift daemon and a minimal MCP server.

## Components

### Operator Daemon (`Operator/`, Swift, macOS 15+)

A macOS MenuBarExtra app (no dock icon) that handles voice I/O, message routing, session management, and iTerm delivery.

**Concurrency model:**
- Swift 6 strict concurrency (zero warnings)
- Actors for shared mutable state: `SessionRegistry`, `AudioQueue`
- `@MainActor` for UI and AVFoundation-bound code (StateMachine, SpeechManager, WaveformPanel)
- Protocol boundaries at every I/O seam for testability: `TerminalBridge`, `RoutingEngine`, `SpeechManaging`, `SpeechTranscribing`, `AudioFeedbackProviding`, `TriggerSource`, `TranscriptionEngine`

### MCP Server (`mcp-server/`, TypeScript)

A single `speak` tool (~60 lines) that POSTs to the daemon's `/speak` endpoint. Claude Code spawns this per-instance as a child process. The MCP server identifies itself using `basename(process.cwd())`.

## Session Lifecycle

Sessions do NOT register through the MCP server. There are two registration paths:

### 1. iTerm Discovery Polling (primary)

`SessionDiscoveryService` polls iTerm via JXA every 10s to discover live terminal sessions by TTY. `SessionRegistry` reconciles the results — new sessions are registered, disappeared sessions are deregistered and announced.

### 2. Claude Code Hooks (supplementary)

The daemon exposes hook endpoints that Claude Code can call for lifecycle events:

| Endpoint | Payload | Effect |
|----------|---------|--------|
| `POST /hook/session-start` | `{session_id, tty, cwd}` | Registers session with Claude session ID |
| `POST /hook/stop` | `{session_id, tty, last_assistant_message?}` | Captures last assistant message as routing context |
| `POST /hook/session-end` | `{session_id, tty}` | Deregisters session |

The `/hook/stop` endpoint is especially valuable — it feeds the last assistant message into `SessionState.context`, which the routing prompt uses to make better routing decisions.

### Why Not MCP Registration?

The `register`, `update_context`, and `get_state` MCP tools were intentionally removed (commit `aefb8ba`). Since Claude Code spawns one MCP server per instance, the MCP server already knows its identity implicitly (TTY from parent process, name from cwd). Discovery polling handles the rest. This avoids requiring LLM cooperation for registration.

## State Machine

The daemon's core is a `StateMachine` (~695 lines, `@MainActor`) with 7 states:

```
IDLE → LISTENING → TRANSCRIBING → ROUTING → DELIVERING → IDLE
                                          → CLARIFYING → (user responds) → ROUTING
                                 → ERROR → (2s auto-recovery) → IDLE
```

- **Latest utterance wins**: triggering push-to-talk from any non-IDLE state cancels in-flight work
- **Timeouts**: per-state timeout management with auto-recovery
- **10 protocol-based dependencies** injected via constructor

## Message Routing

`MessageRouter` implements a six-step priority chain (first match wins):

1. **Operator commands** — "operator status", "list agents", "replay", "what did I miss", "help"
2. **Keyword extraction** — "tell sudo to..." / "hey frontend, ..." / "@backend ..."
3. **Single-session bypass** — only one session registered, route directly
4. **Session affinity** — same session as last message if <15s ago
5. **`claude -p` smart routing** — spawns `claude -p` subprocess to pick best session based on session names, cwds, contexts, and routing history
6. **Clarification fallback** — asks user "which agent?" via voice (max 2 attempts, 8s timeout per attempt)

## Message Delivery

`ITermBridge` delivers messages to specific iTerm panes via JXA `session.write()`:

- Works on **inactive tabs** without stealing focus (validated in `research/2026-03-06-iterm-session-targeting.md`)
- **File-based text injection**: writes message to a temp file, JXA reads it via `NSString.stringWithContentsOfFile`. This completely avoids shell + JavaScript double-escaping.
- **Delivery sequence**: Escape (clear partial input) → injected text → CR (submit)
- Sessions are targeted by TTY path match against iTerm's session objects

## Voice I/O

### Input
- Push-to-talk via FN/Globe key (`CGEventTap` on `maskSecondaryFn`). Configurable secondary hotkey.
- On-device transcription via `SFSpeechRecognizer` (no network needed)

### Output
- `AVSpeechSynthesizer` with premium neural voice fallback chain
- Distinct pitch per agent (7 pitch values from 0.85 to 1.15) via `VoiceManager`
- `AudioQueue` (actor): FIFO with urgent priority insertion, per-session prefixes ("From sudo:"), mute-on-talk, delegate-driven sequential playback

### Interruption Handling
- Word-level tracking of what the user heard vs. unheard portions
- Fast-path keywords: "skip" / "replay"
- `claude -p` fallback for ambiguous interruptions
- Tracks heard/unheard split for intelligent replay

### Audio Feedback
- Five bundled .caf tones: listening, processing, delivered, error, pending
- Played at every state transition — the product is audio-first, not visual-first

## HTTP API

Hummingbird 2.x server on `localhost:7420`. All endpoints require bearer token auth (`~/.operator/token`, 0600 perms).

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/speak` | POST | Queue speech: `{message, session?, priority?}` |
| `/state` | GET | Daemon state, queue depth, all sessions |
| `/hook/session-start` | POST | Claude Code hook: `{session_id, tty, cwd}` |
| `/hook/stop` | POST | Claude Code hook: `{session_id, tty, last_assistant_message?}` |
| `/hook/session-end` | POST | Claude Code hook: `{session_id, tty}` |

## Dependencies

| Dependency | Purpose |
|------------|---------|
| Hummingbird 2.x | HTTP server |
| iTerm2 JXA API | Session discovery + message delivery |
| SFSpeechRecognizer | On-device speech transcription |
| AVSpeechSynthesizer | Text-to-speech output |
| `claude -p` | Smart routing (uses existing Claude CLI auth) |
| @modelcontextprotocol/sdk | MCP protocol for speak tool |

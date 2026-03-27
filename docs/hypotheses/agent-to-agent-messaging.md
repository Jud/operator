# Hypothesis: Agent-to-Agent Messaging via Operator Daemon

**Date:** 2026-03-27
**Status:** Idea — not yet tested

## Observation

The Operator daemon already acts as a central hub for multiple Claude Code sessions:
- SessionRegistry tracks all active sessions (TTY, cwd, context, status)
- HTTP server accepts messages and routes them
- AudioQueue manages speech output to the user

Currently, all communication is agent→user (via speech) or user→agent (via voice).
There's no reason agents couldn't communicate with each other through the same hub.

## The Idea

Add a message delivery path between agents:
- Agent A sends a message targeting Agent B (by session name or routing hint)
- Daemon delivers it as text to Agent B's terminal (same as user voice commands)
- Agent B receives it as input and can act on it

The daemon becomes a **message bus**, not just a voice pipeline.

## What Already Exists

- `SessionRegistry` — knows all sessions, their contexts, what they're working on
- `TerminalBridge.writeToSession()` — delivers text to any terminal session
- `MessageRouter` — routes messages to the right session based on content/context
- `AudioQueue` — could announce inter-agent messages to the user as low-priority pings

## Open Questions

1. **When should agents talk to each other?** Unsolicited messages could be noisy.
   Options: explicit request ("tell the frontend agent..."), event-driven (schema
   changed → notify consumers), or discovery (agent asks "who's working on X?").

2. **How does the user stay in the loop?** Low-priority audio pings ("Backend told
   Frontend about the schema change") or a message log? Or fully transparent —
   user sees everything in the terminals?

3. **Protocol:** Simple text injection? Structured JSON? A new MCP tool that agents
   can call to send messages to other agents?

4. **Trust boundary:** Should agents be able to send arbitrary commands to each other,
   or only informational messages? An agent that can write to another agent's terminal
   can effectively control it.

5. **Feedback loops:** What prevents two agents from getting into an infinite
   conversation loop? Need a circuit breaker.

## Why This Matters

This is the "orchestration" part of "voice orchestration" that isn't built yet.
Voice I/O is the user's interface to the agent swarm. Agent-to-agent messaging
is the swarm's interface to itself. Together, they make the "audio control plane"
real — the user supervises via voice while agents coordinate via the daemon.

## Smallest Testable Version

1. Add `POST /message` endpoint: `{from: "session-a", to: "session-b", text: "..."}`
2. Daemon looks up session B in registry, delivers text via TerminalBridge
3. Low-priority audio ping to user: "Backend sent Frontend a message"
4. Test: two Claude Code sessions, one tells the other about a file change

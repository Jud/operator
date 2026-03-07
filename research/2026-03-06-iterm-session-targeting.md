# iTerm2 Session Targeting: Cross-Tab, No-Focus Text Delivery to TUI Applications

**Date**: 2026-03-06
**Status**: Proven, ready for implementation
**Scope**: Sending text input to Claude Code (and other TUI apps) running in iTerm2 panes, including on inactive tabs, without stealing focus or disrupting the user.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Background: iTerm2's Session Model](#background-iterm2s-session-model)
3. [Terminal State: Why This Is Hard](#terminal-state-why-this-is-hard)
4. [What Worked: The Proven Recipe](#what-worked-the-proven-recipe)
5. [Empirical Test Results](#empirical-test-results)
6. [Session Discovery and Addressing](#session-discovery-and-addressing)
7. [Reading Session Output](#reading-session-output)
8. [Architecture: MCP Server Design](#architecture-mcp-server-design)
9. [Implementation Plan](#implementation-plan)
10. [What Did NOT Work](#what-did-not-work)
11. [Known Limitations and Edge Cases](#known-limitations-and-edge-cases)
12. [Anti-Patterns: Do Not Attempt](#anti-patterns-do-not-attempt)

---

## Problem Statement

We need to programmatically type text into specific iTerm2 panes and submit it, where:

- The target pane may be on an **inactive tab** (not currently visible).
- The target pane is running a **TUI application** (Claude Code via Ink/React), not a bare shell prompt.
- The operation must **never steal focus** from whatever the user is currently doing.
- The operation must **never move the cursor** or activate the iTerm window.
- Pane targeting must be **robust to reorganization** (user can drag/split panes and it still works).
- It must work as part of a **single Harness installation** with no per-app configuration.

The primary use case is: one AI agent sends a prompt to another AI agent running Claude Code in a different iTerm pane, potentially on a different tab.

---

## Background: iTerm2's Session Model

iTerm2 exposes a clean object hierarchy via its scripting APIs (AppleScript, JXA, and Python):

```
Application
  Window (one or more)
    Tab (one or more per window)
      Session (one or more per tab; multiple = split panes)
```

Each **Session** represents one terminal pane and has:

| Property | Description | Example |
|----------|-------------|---------|
| `id` | UUID, globally unique, stable across pane moves | `87425DF0-1CB2-4C42-BE16-EC8B8AE9F826` |
| `tty` | PTY device path | `/dev/ttys004` |
| `name` | Session title (set by shell integration) | `Claude Code (claude)` |
| `profileName` | iTerm2 profile name | `Default` |
| `contents` | Visible terminal buffer text | (full screen contents) |
| `isProcessing` | True if output received in last 2 seconds | `true` / `false` |
| `isAtShellPrompt` | True if shell integration reports idle prompt | `true` / `false` |
| `columns` / `rows` | Terminal dimensions | `88` / `19` |

Session variables (via shell integration) provide additional context:

| Variable | Description | Example |
|----------|-------------|---------|
| `session.path` | Current working directory | `/Users/jud/Projects/harness` |
| `session.jobName` | Foreground process name | `claude` |
| `session.commandLine` | Full command line | `claude` |

### Key Insight: Sessions Survive Tab Switches and Pane Moves

A session's UUID and TTY remain stable when:
- The user switches to a different tab
- The user drags panes to rearrange them
- The user moves a pane to a different tab
- The window is minimized or hidden

This makes UUID/TTY the correct stable identifier for targeting, NOT screen coordinates or AX tree paths.

---

## Terminal State: Why This Is Hard

When Claude Code (or any Ink-based TUI) is running, it puts the terminal into **raw mode**. Here are the actual TTY settings observed on a Claude Code session:

```
speed 38400 baud; 19 rows; 88 columns;
lflags: -icanon -isig -iexten -echo
iflags: -istrip -icrnl -inlcr -igncr -ixon -ixoff ixany imaxbel iutf8
```

Critical flags:
- **`-icanon`**: No line buffering. Each byte delivered immediately to the reading process.
- **`-echo`**: No local echo. The TUI handles all display.
- **`-icrnl`**: Carriage return (`\r`) is NOT translated to newline (`\n`). They are distinct bytes.
- **`-isig`**: Ctrl+C does NOT generate SIGINT at the terminal driver level. The TUI handles it.

This means:
1. Text must be delivered as individual keystrokes, not line-buffered input.
2. "Enter" must be sent as `\r` (carriage return, 0x0D), NOT `\n` (line feed, 0x0A).
3. Special keys (Escape, Ctrl+U, etc.) must be sent as their raw byte values.

### Process Tree Context

A typical Claude Code session has this process tree on its TTY:

```
login → zsh → claude → sourcekit-lsp
                    → caffeinate
                    → node (subprocesses)
```

The `claude` process is the foreground process group leader and is the one reading stdin via the TUI framework.

---

## What Worked: The Proven Recipe

### Transport: iTerm2 JXA `session.write()`

iTerm2's JavaScript for Automation (JXA) API provides a `write` method on session objects:

```javascript
session.write({ text: "hello", newline: false });
```

This method:
- Delivers text through iTerm's internal input pipeline (not raw PTY write).
- Works on **any session**, regardless of which tab is active.
- Does **NOT** steal focus, activate the window, or switch tabs.
- Properly delivers keystrokes to TUI applications in raw mode.

### The Three-Step Sequence

To type and submit a prompt in Claude Code:

```javascript
const app = Application("iTerm2");
// ... resolve session by TTY or UUID ...

// Step 1: Clear any existing input
sess.write({ text: "\u001b", newline: false }); // Escape
delay(0.3);

// Step 2: Type the prompt text
sess.write({ text: "your prompt here", newline: false });
delay(0.2);

// Step 3: Submit with carriage return
sess.write({ text: "\r", newline: false });
```

**Why each step matters:**

| Step | Byte(s) | Purpose |
|------|---------|---------|
| Escape (`\u001b`) | `0x1B` | Dismisses any open overlay (help, menus) and clears input field in Claude Code |
| Text with `newline: false` | UTF-8 bytes | Types into the TUI's input field. `newline: false` is critical — default adds `\n` which doesn't submit |
| Carriage return (`\r`) | `0x0D` | Submits the input. Must be `\r` not `\n` because terminal is in raw mode with `-icrnl` |

### Why `newline: false` Is Critical

iTerm's `write text` defaults to appending a newline (`\n`, 0x0A) after the text. In a normal shell with `icrnl` enabled, this would work fine. But Claude Code's TUI has `icrnl` disabled, so:

- `\n` (0x0A) = line feed. The TUI does NOT interpret this as "submit".
- `\r` (0x0D) = carriage return. The TUI interprets this as Enter/submit.

You must always use `newline: false` and send `\r` explicitly.

---

## Empirical Test Results

### Test 1: Direct PTY Write (FAILED for TUI)

```bash
printf 'hello' > /dev/ttys004
```

**Result**: Text appeared as raw characters in the terminal window but was NOT picked up by Claude Code's TUI input field. The bytes went to the PTY device but bypassed iTerm's input processing pipeline.

**Conclusion**: Direct PTY write does not work for TUI applications. iTerm processes input through its own pipeline that the TUI depends on.

### Test 2: iTerm `write text` with Default Newline (PARTIAL)

```javascript
sess.write({ text: "/help" }); // newline defaults to true
```

**Result**: Text appeared in Claude Code's input field, but the appended `\n` did NOT trigger submission. The text just sat there with a newline character that the TUI didn't interpret as Enter.

### Test 3: iTerm `write text` with `newline: false` (TYPING WORKS)

```javascript
sess.write({ text: "test from harness", newline: false });
```

**Result**: Text appeared correctly in Claude Code's input field. No submission occurred (as expected).

### Test 4: Escape + Text + Carriage Return (FULL SUCCESS)

```javascript
sess.write({ text: "\u001b", newline: false }); // clear
delay(0.3);
sess.write({ text: "/help", newline: false });   // type
delay(0.2);
sess.write({ text: "\r", newline: false });       // submit
```

**Result**: Input field was cleared, `/help` was typed, Enter was triggered, and Claude Code executed the `/help` command. Help output appeared in the session. **The active tab did not change. Focus did not move.**

### Test 5: Full Prompt Submission (FULL SUCCESS)

```javascript
// After escape + clear:
sess.write({ text: "This is a test prompt. Use the say command to say I got it after receiving this prompt.", newline: false });
delay(0.2);
sess.write({ text: "\r", newline: false });
```

**Result**: Claude Code received the prompt, processed it, and responded with "I got it." The prompt was submitted and executed on a completely inactive tab. The user's active tab was never disturbed.

---

## Session Discovery and Addressing

### Listing All Sessions

```javascript
const app = Application("iTerm2");
const results = [];
for (const win of app.windows()) {
    for (const tab of win.tabs()) {
        for (const sess of tab.sessions()) {
            results.push({
                id: sess.id(),
                tty: sess.tty(),
                name: sess.name(),
                isProcessing: sess.isProcessing(),
                isAtShellPrompt: sess.isAtShellPrompt(),
                // Requires shell integration:
                workingDirectory: sess.variable({ named: "session.path" })
            });
        }
    }
}
```

### Finding a Session by TTY

```javascript
function findSessionByTTY(tty) {
    const app = Application("iTerm2");
    for (const win of app.windows()) {
        for (const tab of win.tabs()) {
            for (const sess of tab.sessions()) {
                if (sess.tty() === tty) return sess;
            }
        }
    }
    return null;
}
```

### Auto-Detecting Own TTY

Any process can determine its own TTY with:

```bash
tty
# Output: /dev/ttys004
```

Or programmatically in Node.js:

```javascript
const tty = require('child_process').execSync('tty', { stdio: ['inherit', 'pipe', 'pipe'] }).toString().trim();
```

Or by reading `/dev/stdin`:

```javascript
const fs = require('fs');
const tty = fs.readlinkSync('/dev/fd/0'); // may not work in all contexts
```

---

## Reading Session Output

### Full Buffer Read

```javascript
const contents = sess.contents();
// Returns the full visible terminal buffer as a string
```

### Parsing for Meaningful Content

The buffer contains the full screen (including blank lines). To extract meaningful content:

```javascript
const lines = sess.contents().split("\n").filter(l => l.trim().length > 0);
const lastN = lines.slice(-10); // Last 10 non-empty lines
```

### Polling for Command Completion

```javascript
// Wait for the session to stop processing (no output for 2 seconds)
function waitForIdle(sess, maxWaitMs = 30000) {
    const start = Date.now();
    while (sess.isProcessing() && (Date.now() - start) < maxWaitMs) {
        delay(0.5);
    }
    return !sess.isProcessing();
}
```

Note: `isProcessing` returns `true` if the session received any output in the last ~2 seconds. `isAtShellPrompt` requires iTerm2 shell integration and returns `true` when at a bare shell prompt (not useful for Claude Code which is a TUI).

---

## Architecture: MCP Server Design

### Overview

A standalone MCP server that wraps iTerm2's JXA API into MCP tools. Any MCP client (Claude Code, custom agents, etc.) can use it to discover and interact with iTerm sessions.

```
Claude Code A ─── MCP ──┐
Claude Code B ─── MCP ──┤
Custom Agent  ─── MCP ──┼── iTerm Session Bridge (MCP Server)
                         │         │
                         │    osascript/JXA
                         │         │
                         │    iTerm2 Application
                         │    ├── Tab 1: Session A, Session B
                         │    └── Tab 2: Session C
                         └── Registry (label → TTY → UUID)
```

### MCP Tools

#### `list_sessions`

Discovers all iTerm2 sessions across all windows and tabs.

**Parameters**: None (or optional `window` filter)

**Returns**:
```json
[
  {
    "id": "87425DF0-1CB2-4C42-BE16-EC8B8AE9F826",
    "tty": "/dev/ttys004",
    "name": "Claude Code (claude)",
    "tab": 0,
    "paneIndex": 1,
    "workingDirectory": "/Users/jud/Projects/sudo",
    "isProcessing": false,
    "label": "sudo-agent"
  }
]
```

#### `register_terminal`

Registers the calling agent's session with a human-readable label. The agent auto-detects its TTY.

**Parameters**:
- `label` (string, required): Human-readable name, e.g. `"harness-agent"` or `"sudo-builder"`
- `tty` (string, optional): Override TTY detection. Normally auto-detected.

**Effect**: Maps `label → TTY → iTerm session UUID` in the server's in-memory registry.

#### `write_to_session`

Sends text to a target session and optionally submits it.

**Parameters**:
- `target` (string, required): Label, TTY path, or session UUID
- `text` (string, required): Text to type
- `submit` (boolean, default `true`): Whether to send carriage return after text
- `clear_first` (boolean, default `true`): Whether to send Escape before typing

**Implementation**:
```javascript
// Resolve target to iTerm session
// If clear_first: send \u001b, delay 300ms
// Send text with newline: false
// If submit: delay 200ms, send \r with newline: false
```

#### `read_session`

Reads the current visible contents of a target session.

**Parameters**:
- `target` (string, required): Label, TTY path, or session UUID
- `last_n_lines` (number, optional): Only return the last N non-empty lines

**Returns**:
```json
{
  "contents": "... terminal buffer ...",
  "isProcessing": true,
  "lines": ["line1", "line2", "..."]
}
```

#### `wait_for_idle`

Blocks until a session stops processing (useful after submitting a prompt).

**Parameters**:
- `target` (string, required): Label, TTY path, or session UUID
- `timeout_ms` (number, default `30000`): Maximum wait time

**Returns**:
```json
{
  "idle": true,
  "contents": "... final buffer contents ..."
}
```

### Transport Layer

All iTerm interaction happens via `osascript -l JavaScript` subprocess calls from the MCP server (Node.js). This is simple and reliable:

```javascript
const { execSync } = require('child_process');

function writeToSession(tty, text, newline = false) {
    const escaped = text.replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/"/g, '\\"');
    const script = `
        const app = Application("iTerm2");
        for (const win of app.windows()) {
            for (const tab of win.tabs()) {
                for (const sess of tab.sessions()) {
                    if (sess.tty() === "${tty}") {
                        sess.write({ text: "${escaped}", newline: ${newline} });
                        return "ok";
                    }
                }
            }
        }
        return "session_not_found";
    `;
    return execSync(`osascript -l JavaScript -e '${script}'`).toString().trim();
}
```

---

## Implementation Plan

### Phase 1: Standalone MCP Server (Minimal Viable)

1. **New repo/package**: `iterm-mcp` (or similar)
2. **Runtime**: Node.js + `@modelcontextprotocol/sdk`
3. **Tools**: `list_sessions`, `write_to_session`, `read_session`
4. **Transport**: `osascript -l JavaScript` subprocesses
5. **No registration needed initially**: Just target by TTY or session name

Estimated size: ~200-300 lines of TypeScript.

### Phase 2: Registration and Labels

6. **In-memory registry**: Map of label → TTY
7. **`register_terminal` tool**: Agents self-register with labels
8. **TTY auto-detection helper**: Document how agents detect their own TTY

### Phase 3: Robustness

9. **Session validation**: Before writing, verify session still exists (TTY → UUID lookup)
10. **`wait_for_idle` tool**: Poll `isProcessing` for command completion
11. **Error handling**: Session closed, iTerm not running, permission denied
12. **Stale registration cleanup**: Periodically verify registered TTYs still exist

---

## What Did NOT Work

### 1. Direct PTY Write (`printf > /dev/ttys*`)

```bash
printf 'hello' > /dev/ttys004
```

Text appeared as raw characters in the terminal display but was NOT captured by Claude Code's TUI input field. The characters bypassed iTerm's input processing and went directly to the PTY output side, appearing as if they were program output rather than user input.

**Root cause**: Writing to `/dev/ttys*` from another process delivers bytes to the terminal's output stream (what gets displayed), not the input stream (what the foreground process reads from stdin). iTerm's `write text` uses a fundamentally different path — it injects into iTerm's internal input pipeline, which then delivers to the PTY's input side.

### 2. Default `newline: true` on `write text`

```javascript
sess.write({ text: "/help" }); // newline defaults to true
```

Text was typed into the input field, but the appended `\n` (line feed, 0x0A) was not interpreted as Enter/submit by Claude Code. The TUI continued waiting for input.

**Root cause**: Claude Code's TUI runs in raw mode with `-icrnl` (no CR-to-NL translation). The TUI expects `\r` (carriage return, 0x0D) for Enter. iTerm's default newline sends `\n` (0x0A), which is a different byte that the TUI ignores or treats as a literal.

### 3. Accessibility (AX) Based Approach for Inactive Tabs

The macOS Accessibility API exposes iTerm panes as:
```
AXWindow → AXGroup → AXSplitGroup → AXScrollArea → AXTextArea (role: "shell")
```

However, the AX tree **only exposes elements for the currently visible tab**. Panes on inactive tabs are not in the AX tree. Additionally, `AXValue` set on a terminal's `AXTextArea` replaces the buffer contents rather than injecting input keystrokes.

**Root cause**: AX operates on the rendered UI, not the application's session model. Inactive tabs have no rendered UI elements to target.

### 4. System Events AppleScript (AX via AppleScript)

```applescript
tell application "System Events"
    tell process "iTerm2"
        get entire contents
    end tell
end tell
```

Failed with type coercion errors (`-1700`). iTerm's AX implementation has known gaps (GitLab issue #5527: `AXTabGroup` lacks standard `AXTabs` and `AXChildren` attributes).

### 5. CGEvent Keyboard Input

`CGEvent.postToPid()` can send keystrokes to a specific process, but:
- Requires the target window/field to have focus
- Cannot target a specific pane within a process
- Violates the no-focus-steal principle

---

## Known Limitations and Edge Cases

### 1. Text That Contains Special Characters

The JXA string escaping must handle:
- Backslashes (`\\`)
- Quotes (`"` and `'`)
- Unicode characters
- Escape sequences that iTerm might interpret (e.g., `\u001b[...` ANSI codes)

Mitigation: Proper escaping in the JXA bridge layer. Consider Base64 encoding for complex payloads.

### 2. Timing Sensitivity

The delays between escape, text, and carriage return are empirically chosen:
- 300ms after Escape (to allow TUI to process dismiss/clear)
- 200ms after text before CR (to allow TUI to process all characters)

These may need tuning for slower machines or complex TUI states. Consider making delays configurable.

### 3. iTerm Not Running or No Sessions

If iTerm2 is not running, `osascript` calls will fail. The MCP server should handle this gracefully.

### 4. Session State Ambiguity

`isProcessing` is a 2-second trailing indicator. A session that just finished processing will still report `true` for up to 2 seconds. `isAtShellPrompt` requires shell integration and does not work for TUI apps.

For Claude Code specifically, reading the buffer contents and looking for the prompt indicator (`>`) may be more reliable.

### 5. Race Conditions with `write text`

iTerm2 GitLab issue #7753 documents that `write text` can fail if a session is not yet fully initialized. Issue #5462 notes that if a session is killed, `write text` may target the wrong session.

Mitigation: Always verify session existence by TTY before writing.

### 6. `tell application "iTerm2"` May Activate iTerm

Standard AppleScript `tell application` can activate (bring to front) the target application. In our testing, the JXA approach (`Application("iTerm2")`) did NOT activate iTerm or steal focus. However, this should be monitored. If it becomes an issue, investigate:
- `Application("iTerm2")` with `includeStandardAdditions = false`
- Using `System Events` to send the osascript rather than addressing iTerm directly
- Running osascript with `without activating` qualifier

### 7. Multiple iTerm Windows

The current implementation iterates all windows. If the user has multiple iTerm windows, session lookup by TTY will still work correctly since TTYs are globally unique. Label-based targeting should also work across windows.

---

## Anti-Patterns: Do Not Attempt

These approaches have been tested or researched and confirmed to not work for this use case. If debugging issues, do NOT go down these paths:

| Anti-Pattern | Why It Fails |
|-------------|--------------|
| `printf > /dev/ttys*` | Writes to terminal output, not input. TUI never sees it as keystrokes. |
| `write text` with default newline | Sends `\n` which raw-mode TUIs don't interpret as Enter. |
| AX `setValue` on terminal TextArea | Replaces buffer contents, doesn't inject input. Also can't reach inactive tabs. |
| `CGEvent.postToPid()` for keystrokes | Requires focus. Cannot target specific panes. |
| AX tree traversal for inactive tabs | Inactive tab elements don't exist in the AX tree. |
| `xdotool` / `cliclick` style tools | macOS doesn't have xdotool. Click-based tools require coordinates and focus. |
| Setting `AXFocused` on a pane | Steals focus, violating core principle. |
| iTerm Python API from MCP server | Requires running inside iTerm's Python runtime. The JXA/osascript approach is simpler and works from any process. |
| `tmux send-keys` without tmux | Only works if sessions are running inside tmux. Not a general solution. |

---

## Appendix: Key References

- **iTerm2 Scripting Docs**: https://iterm2.com/documentation-scripting.html
- **iTerm2 Python API (Session)**: https://iterm2.com/python-api/session.html
- **iTerm2 Variables**: https://iterm2.com/documentation-variables.html
- **iTerm2 GitLab Issues**: #7753 (write text race), #5462 (stale session targeting), #5527 (AX tab gaps)
- **macOS Accessibility**: AXUIElement API, `AXTextArea` role, `-icrnl` terminal flag
- **Related Harness Scenario**: `scenarios/2026-03-02-ax-focus-steal-empirical-research.md`

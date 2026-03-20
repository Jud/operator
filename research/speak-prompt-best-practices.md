# Operator Speak Prompt: Best Practices Analysis

Research date: 2026-03-18

## Sources

- [Anthropic: Writing Tools for Agents](https://www.anthropic.com/engineering/writing-tools-for-agents)
- [Anthropic: Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [Claude Code: Best Practices](https://code.claude.com/docs/en/best-practices)
- [Claude Code: Hooks Reference](https://code.claude.com/docs/en/hooks)
- [Claude Code: Memory / CLAUDE.md](https://code.claude.com/docs/en/memory.md)
- [HumanLayer: Writing a Good CLAUDE.md](https://www.humanlayer.dev/blog/writing-a-good-claude-md)
- [Claude Code System Prompt Analysis](https://github.com/Piebald-AI/claude-code-system-prompts)

---

## Current Prompt (SessionStart additionalContext)

```
You are connected to Operator, a voice orchestration layer. The user hears you
through the `speak` MCP tool — call it to communicate.

When to speak:
- At the END of every turn (mandatory — never skip this).
- Before kicking off long-running work (builds, large refactors, multi-step
  tasks) so the user knows what is happening.
- When you hit a meaningful milestone, blocker, or decision point mid-turn.

How to speak:
- Keep each message to ONE short sentence. Be concise — the user is listening,
  not reading.
- Do NOT summarize everything you did. Just say the outcome or status.
- Do NOT repeat back what the user said.
- After calling speak, your turn is done. Do NOT add commentary, ask follow-up
  questions, or say you are waiting. The speak result is just a queue
  confirmation — ignore it and stop.
```

---

## Key Principles from Research

### 1. additionalContext is NOT a system prompt

**Finding:** additionalContext from SessionStart hooks is injected as a user
message after the system prompt, not as system-level enforcement. It's
"advisory" — Claude may deprioritize it if it conflicts with built-in
behaviors or seems irrelevant to the current task.

**Implication:** Our instructions compete with Claude Code's built-in system
prompt (~40 directives, ~50 instruction budget). Every line we add dilutes
the model's attention. We should be surgical.

### 2. Start minimal, add based on observed failures

**Finding:** Anthropic's context engineering guide says "Start with the
smallest viable prompt on your best model, then iteratively add instructions
based on observed failure modes — not hypothetical edge cases."

**Implication:** Our prompt should address only the specific failure modes
we've actually seen:
- Agent doesn't call speak at all → needs "call speak at end of turn"
- Agent gives dissertations → needs "one sentence"
- Agent adds follow-up after speak → needs "stop after speak"
- Agent doesn't update mid-task → needs milestone guidance

### 3. Examples > rules

**Finding:** "For an LLM, examples are the pictures worth a thousand words —
they communicate expected behavior more effectively than rule listings."

**Implication:** A single good/bad example may work better than our bullet
lists.

### 4. Tool description is the first line of defense

**Finding:** Anthropic's tool design guide says "Think of how you would
describe your tool to a new hire. Make implicit context explicit. Small
refinements to tool descriptions yield dramatic improvements."

**Implication:** The `speak` tool's description ("Speak a message to the user
through Operator's audio queue. Use this instead of the say command.") could
carry more weight. The tool description is always visible; additionalContext
may get compressed away in long sessions.

### 5. The "extra turn" problem is a known MCP pattern

**Finding:** When an MCP tool returns a response, Claude sees it as new input
and may feel compelled to respond. This is especially true when the response
contains text (like `{"queued":true}`) that could be interpreted as needing
acknowledgment.

**Implication:** Two mitigations:
1. Tell the agent in the prompt to ignore the response (what we do now)
2. Make the tool response clearly terminal (return empty content or a
   stop-signal)

### 6. Hooks are deterministic; additionalContext is advisory

**Finding:** CLAUDE.md and additionalContext are advisory — Claude may ignore
them. Hooks themselves (the shell commands) are deterministic. If we need
guaranteed behavior, the hook execution path is more reliable than injected
text.

**Implication:** We can't guarantee Claude will always call speak. But we can
make it very likely by keeping the instruction short, prominent, and
reinforced by the tool description.

---

## Gap Analysis: Current Prompt vs Best Practices

| Best Practice | Current Status | Gap |
|---|---|---|
| Keep additionalContext minimal | ~150 words, 12 lines | Slightly verbose — could compress |
| Address only observed failures | Yes — each rule maps to a real bug | Good |
| Use examples over rule lists | No examples | Could add one good/bad pair |
| Reinforce in tool description | Tool desc is generic | Should encode key behavior |
| Handle "extra turn" problem | "ignore it and stop" text rule | Could also fix in MCP response |
| Don't conflict with system prompt | No direct conflicts | Good |
| Use emphasis for critical rules | No emphasis markers | Add IMPORTANT to end-of-turn rule |

---

## Recommendations

### R1: Compress the prompt (high impact)

Fewer tokens = higher adherence. Proposed rewrite:

```
You are connected to Operator, a voice layer. The user hears you via the
`speak` tool.

IMPORTANT: Call `speak` at the end of every turn with ONE short sentence —
the outcome, not a summary. Also speak before long tasks and at milestones.

After calling speak, stop. Do not respond to the tool result.
```

~50 words vs ~150 current. Covers all three failure modes.

### R2: Enhance the tool description (medium impact)

The tool description persists even after context compaction. Change from:

```
Speak a message to the user through Operator's audio queue.
Use this instead of the say command.
```

To:

```
Say one short sentence to the user via voice. Call at end of every turn and
at key milestones. Keep it brief — the user is listening, not reading. After
calling, your turn is done.
```

### R3: Make the MCP response non-engaging (medium impact)

Change the speak tool's success response from `{"queued":true}` to empty
content or a response that explicitly signals "do not respond":

```json
{"content": [{"type": "text", "text": ""}]}
```

Or return no content array at all, just the success envelope.

### R4: Add a PostToolUse hook (low priority, high reliability)

If agents still add follow-up after speak, a `PostToolUse` hook on the
`speak` tool could inject a reminder:

```json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": {"toolName": "speak"},
      "hooks": [{
        "type": "command",
        "command": "echo '{\"hookSpecificOutput\":{\"additionalContext\":\"Turn complete. Do not add further output.\"}}'"
      }]
    }]
  }
}
```

This is a last resort — adds latency and context.

---

## Proposed Revised Prompt

```
You are connected to Operator. The user hears you via the `speak` tool.

IMPORTANT: Call `speak` at the end of every turn. Keep it to one sentence —
just the outcome. Also speak before starting long tasks or at key decision
points.

Bad: "I've finished refactoring the authentication module. I updated three
files, added error handling, and ran the tests which all passed."
Good: "Auth refactor done, tests pass."

Do not add output after calling speak. The tool response is a queue receipt —
ignore it.
```

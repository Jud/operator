---
description: Debug Operator transcription and voice pipeline issues
---

# Debug Operator

Debug the Operator voice pipeline by pulling system logs and analyzing them for known failure patterns.

## Mode

If `$ARGUMENTS` contains "last" or is empty, analyze the most recent transcription attempt.
If `$ARGUMENTS` contains "crash", look for recent crashes and SIGTRAP events.
If `$ARGUMENTS` contains a number N, show the last N minutes of logs (default: 2).

## Step 1: Pull Logs

Run this command to get the relevant Operator logs, filtering out discovery/hook noise:

```
/usr/bin/log show \
  --predicate 'subsystem CONTAINS "operator" AND NOT category == "HTTPServer" AND NOT category == "SessionRegistry" AND NOT category == "ITermBridge" AND NOT category == "MultiTerminalBridge"' \
  --last <N>m --debug --info
```

Where N is the time window (default 2 minutes, or as specified in arguments).

## Step 2: Analyze for Known Failure Patterns

Check the logs for each of these patterns in order:

### Critical Issues

| Pattern | Log Signature | Root Cause |
|---------|--------------|------------|
| No mic permissions | `Microphone permission not granted` | App launched from CLI instead of .app bundle, or TCC permissions revoked. Audio engine captures silence. |
| App crash | `exited due to SIGTRAP` or `exited due to signal` | Pipeline corruption, usually from concurrent `pipe.transcribe()` calls or force-unwrap. |
| Audio engine failed | No `Audio engine started` after `IDLE -> LISTENING` | Mic device unavailable or format mismatch. |

### Transcription Failures

| Pattern | Log Signature | Root Cause |
|---------|--------------|------------|
| Empty transcription | `Transcription returned empty` | Model produced no segments. Check for fallback/temperature escalation in preceding Argmax logs. |
| Decoder fallbacks | `Fallback #1 (compressionRatioThreshold)` or `Fallback #2 (firstTokenLogProbThreshold)` | Model not confident in output. Usually means silent/noisy audio, or audio too short. Multiple fallbacks raising temperature to 0.4+ almost always means garbage input. |
| No audio samples | `No audio samples captured` | Session was stopped before any audio buffers arrived. |
| Concurrent inference | Two `Decoding X.XXs` lines overlapping in time on different threads | Race condition: background loop and final pass running `pipe.transcribe()` simultaneously. |

### Timing Issues

| Pattern | Log Signature | Root Cause |
|---------|--------------|------------|
| Slow final pass | Long gap between `Audio engine stopped` and `Transcription` result | Either awaiting a background pass (expected) or re-transcribing too much audio (check clip= value). |
| No confirmation | Only `Background pass: 1 seg (need 2+ to confirm)` messages | Normal for utterances under ~5s. Check if clip= stays at 0 on long utterances (would indicate confirmation never kicks in). |
| Short recording | Less than 1s between LISTENING and TRANSCRIBING | User tapped instead of held the push-to-talk key. |

### State Machine Issues

| Pattern | Log Signature | Root Cause |
|---------|--------------|------------|
| Trigger ignored | `Trigger STOP ignored: not in LISTENING state` | Double key-release event. Usually harmless. |
| Stuck state | No transition out of TRANSCRIBING or ROUTING for 10+ seconds | Pipeline hung. Check for concurrent inference or model loading issues. |

## Step 3: Report

Summarize findings as:

1. **What happened**: Trace the state transitions and key events in chronological order
2. **Root cause**: Which failure pattern(s) matched
3. **Evidence**: The specific log lines that confirm the diagnosis
4. **Fix**: What to do about it (relaunch app, code change needed, user action, etc.)

If the transcription succeeded, report the timing breakdown:
- Recording duration (LISTENING to TRANSCRIBING)
- Background passes run and segments confirmed
- Final pass duration and clip point
- Total latency (key release to result)

## Step 4: Speak

Use the `speak` MCP tool to give a brief verbal summary of the diagnosis.

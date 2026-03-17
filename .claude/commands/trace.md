---
description: Performance trace of the last Operator push-to-talk transcription
---

# Trace Last Transcription

Pull Operator system logs and produce a performance trace of the most recent push-to-talk session.

## Step 1: Pull Logs

Run this command to get the last 2 minutes of WhisperKitEngine and StateMachine logs from the live Operator app (excluding test runners and benchmarks):

```
/usr/bin/log show \
  --predicate 'subsystem CONTAINS "operator" AND (category == "WhisperKitEngine" OR category == "StateMachine" OR category == "SpeechTranscriber" OR category == "Argmax") AND processImagePath ENDSWITH "Operator"' \
  --last 2m --debug --info
```

If `$ARGUMENTS` contains a number, use that as the minutes window instead of 2.

If the window returns no results, widen automatically to 10m, then 30m.

## Step 2: Find the Last Session

Scan backwards for the most recent `Session started` log from WhisperKitEngine. Everything from that line through the next `Result:` or `Transcription empty` line is one session.

If there's no session in the window, tell the user and suggest a longer window.

## Step 3: Build the Timeline

Extract these events into a timeline table:

| Time | Event | Details |
|------|-------|---------|
| T+0ms | **Session started** | (prepare called) |
| T+Xms | IDLE -> LISTENING | Audio engine starting |
| T+Xms | Audio engine started | Recording begins |
| T+Xms | **Background pass N** | N seg in Xms (clip=Xs, audio=Xs) |
| T+Xms | **Confirmed** | N seg to Xs: "text..." |
| T+Xms | LISTENING -> TRANSCRIBING | Key released |
| T+Xms | **Stop** | Xs audio, confirmed=Xs, await=Xms |
| T+Xms | **Result** | Xms total (await=Xms, final=Xms) clip=Xs |

Use millisecond offsets from session start (T+0ms).

## Step 4: Compute Metrics

Calculate and display:

- **Recording duration**: Time from LISTENING to TRANSCRIBING
- **Background passes**: Count, average inference time per pass
- **Confirmation**: How many seconds confirmed vs total audio (percentage)
- **Post-release latency**: Time from TRANSCRIBING to Result (this is what the user feels)
  - Await time (waiting for background task to finish)
  - Final pass time (decoding unconfirmed tail)
- **Decoder fallbacks**: Any temperature escalation from Argmax logs (compressionRatioThreshold, firstTokenLogProbThreshold)

## Step 5: Assess

Rate the session:

- **Fast** (< 500ms post-release): Confirmation worked well, short tail to decode
- **Normal** (500ms - 1s post-release): Acceptable for push-to-talk
- **Slow** (1s - 2s post-release): Confirmation may not have kicked in, or long unconfirmed tail
- **Problem** (> 2s post-release): Investigate — likely race condition, decoder fallbacks, or no confirmation on long audio

Flag any issues:
- Decoder fallbacks (temperature > 0) indicate the model struggled with the audio
- `await` time > 100ms means a background inference was in-flight at key release
- `clip=0.0s` on audio > 5s means confirmation never worked
- `Transcription empty` means the model produced no output

## Step 6: Speak

Use the `speak` MCP tool to give a brief verbal summary: the post-release latency, whether confirmation worked, and any issues found.

---
description: Performance trace of the last Operator push-to-talk transcription
---

# Trace Last Transcription

Pull Operator system logs and produce a performance trace of the most recent push-to-talk session.

## Background

Operator uses WhisperKit for on-device speech-to-text with a word-level streaming confirmation algorithm. While the user holds the push-to-talk key (FN), a background loop repeatedly decodes the growing audio. Each pass compares its output to the previous pass — words that agree across two consecutive passes are "confirmed" and locked in. When the user releases the key, a final decode handles the unconfirmed tail.

Key state tracked during a session:
- `lastAgreedSeconds` — how far into the audio confirmation has reached
- `confirmedText` — locked-in words from background passes
- `consecutiveZeroProgress` — passes where agreement was found but no NEW words were confirmed (the algorithm is stuck but doesn't know it)
- `prefixTokens` — decoder bias tokens from the last agreed words, used in the final decode

The main failure mode is **confirmation stall**: the algorithm finds agreement on the same 2 words each pass but never advances, leaving a large unconfirmed tail. If the final decode also fails to recover the tail, the transcription is truncated.

## Step 1: Pull Logs

Run this command to get recent WhisperKitEngine, StateMachine, SpeechTranscriber, and Argmax logs from the live Operator app (excluding test runners and benchmarks):

```
/usr/bin/log show \
  --predicate 'subsystem CONTAINS "operator" AND (category == "WhisperKitEngine" OR category == "StateMachine" OR category == "SpeechTranscriber" OR category == "Argmax" OR category == "AudioFeedback") AND processImagePath ENDSWITH "Operator"' \
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
| T+0ms | **Session started** | prepare called |
| T+Xms | IDLE → LISTENING | State transition |
| T+Xms | Audio engine started | Recording begins |
| T+Xms | **Background pass** | Argmax decode X.Xs–Y.Ys |
| T+Xms | **Confirmed N words** | to X.Xs (zeroProgress=N) |
| T+Xms | LISTENING → TRANSCRIBING | Key released |
| T+Xms | **Stop** | Xs audio, agreed=Xs, tail=Xs, zeroProgress=N |
| T+Xms | **Final tail** | N words from X.Xs: "text..." |
| T+Xms | **Final combined** | confirmed=N chars + tail=N chars |
| T+Xms | **Result** | Xms total (await=Xms, final=Xms) |
| T+Xms | 🔊 **Tone: processing** | (or dismissed, dictation, delivered, error) |

Use millisecond offsets from session start (T+0ms).

Include AudioFeedback "Playing cue:" events in the timeline with a 🔊 prefix. These show the exact timing of feedback tones — verify that processing and dictation tones have adequate spacing (≥300ms).

For Confirmed lines, include the `zeroProgress` counter when it's > 0. This counter tracks consecutive passes where agreement was found but no new words were confirmed — it indicates confirmation is stalling.

## Step 4: Compute Metrics

Calculate and display:

- **Recording duration**: Time from LISTENING to TRANSCRIBING
- **Background passes**: Count, average inference time per pass
- **Confirmation coverage**: `agreed / audioDuration` as a percentage — how much audio was confirmed before key release
- **Unconfirmed tail**: `audioDuration - lastAgreedSeconds` (from the Stop log's `tail=` field). This is the audio the final decode must handle alone.
- **Zero-progress stalls**: The `zeroProgress` value from the Stop log. Non-zero means confirmation was stuck.
- **Post-release latency**: Time from TRANSCRIBING to Result (this is what the user feels)
  - Await time (waiting for background task to finish)
  - Final pass time (decoding unconfirmed tail)
- **Decoder fallbacks**: Any temperature escalation > 0 from Argmax logs
- **Final decode assembly**: From the `Final combined` log — how many chars came from confirmed text vs the final tail decode

## Step 5: Assess

Rate the session:

- **Fast** (< 500ms post-release): Confirmation worked well, short tail to decode
- **Normal** (500ms–1s post-release): Acceptable for push-to-talk
- **Slow** (1s–2s post-release): Confirmation may not have kicked in, or long unconfirmed tail
- **Problem** (> 2s post-release): Investigate — likely race condition, decoder fallbacks, or no confirmation on long audio

Flag any issues:

- **Confirmation stall** (`zeroProgress > 0` in Stop log): The confirmation algorithm found agreement but stopped making progress. The same 2 words kept matching without new words being confirmed. Severity depends on the resulting tail size.
- **Large unconfirmed tail** (`tail > 5s`): A big chunk of audio had to be decoded in the final pass. Risk of truncation if the final decode fails.
- **Truncation** (Result text is significantly shorter than expected for the audio duration): Compare word count to audio duration. Normal speech is 2–3 words/sec. Below 1 word/sec on audio > 3s suggests content was lost.
- **Filtered words** (`Final decode: N words, M filtered`): Words from the final decode were dropped because their timestamps were before `lastAgreedSeconds`. This can cause silent content loss.
- **Decoder fallbacks** (temperature > 0): The model struggled with the audio segment and retried at higher temperature.
- **Await > 100ms**: A background inference was still running when the key was released, adding latency.
- **CancellationError in logs**: The background task was cancelled mid-inference at key release. Normal if `await` is small, but if the cancelled pass was about to confirm words, those words are lost.
- **Transcription empty**: The model produced no output at all.

## Step 6: Speak

Use the `speak` MCP tool to give a brief verbal summary: the post-release latency, confirmation coverage, tail size, and any issues found.

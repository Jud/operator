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

## Step 5: Ground Truth Verification

Find the audio trace WAV file from the `Saved audio trace:` log line. Extract the filename (e.g., `2026-03-20T09-17-23-05-00.wav`).

Run a full batch decode of the audio trace to get the ground truth transcription:

```bash
OPERATOR_STT_WAV="$HOME/Library/Application Support/Operator/audio-traces/<filename>" \
  ./scripts/run-benchmarks.sh --no-build stt-replay-batch
```

Compare the ground truth text against the session's Result text. Flag any differences:
- **Missing content**: words in ground truth but not in the result (content was lost)
- **Extra content**: words in result but not in ground truth (hallucination)
- **Different words**: substitutions between the two (may indicate confirmation or decoder issues)

Display both texts side-by-side for easy comparison.

## Step 6: Assess

Rate the session:

- **Fast** (< 500ms post-release): Confirmation worked well, short tail to decode
- **Normal** (500ms–1s post-release): Acceptable for push-to-talk
- **Slow** (1s–2s post-release): Confirmation may not have kicked in, or long unconfirmed tail
- **Problem** (> 2s post-release): Investigate — likely race condition, decoder fallbacks, or no confirmation on long audio

Flag any issues:

- **Confirmation stall** (`zeroProgress > 0` in Stop log): The confirmation algorithm found agreement but stopped making progress. The same 2 words kept matching without new words being confirmed. Severity depends on the resulting tail size.
- **Large unconfirmed tail** (`tail > 5s`): A big chunk of audio had to be decoded in the final pass. Risk of truncation if the final decode fails.
- **Truncation** (Result text is significantly shorter than ground truth): Content was lost. Check whether it was lost in the tail decode or the confirmation phase.
- **Filtered words** (`Final decode: N words, M filtered`): Words from the final decode were dropped because their timestamps were before `lastAgreedSeconds`. This can cause silent content loss.
- **Decoder fallbacks** (temperature > 0): The model struggled with the audio segment and retried at higher temperature.
- **Await > 100ms**: A background inference was still running when the key was released, adding latency.
- **CancellationError in logs**: The background task was cancelled mid-inference at key release. Normal if `await` is small, but if the cancelled pass was about to confirm words, those words are lost.
- **Transcription empty**: The model produced no output at all.

## Step 7: Speak

Use the `speak` MCP tool to give a brief verbal summary: the post-release latency, confirmation coverage, tail size, ground truth comparison, and any issues found.

## Step 8: Capture Fixture

Every trace should produce a fixture. The default is to always capture.

### For failing sessions (similarity < 85%):
1. Generate a descriptive name from the issue (e.g., `content-loss-short-utterance`, `stall-at-30s`, `empty-tail-decode`)
2. Run: `./scripts/capture-fixture.sh <name> <wav-path>`
3. Tell the user the fixture was captured and they can run `make test-transcription` to verify

### For passing sessions (similarity >= 85%):
Before capturing, check if the session is meaningfully different from existing fixtures. Consider it unique if it has a characteristic not already covered:
- **Duration range**: short (<3s), medium (3-15s), long (15-60s), very long (>60s)
- **Confirmation behavior**: high coverage (>80%), low coverage (<30%), zero-progress stalls, strategy transition to segment-level
- **Content type**: single sentence, multiple sentences, technical terms, numbers/dates
- **Edge cases**: rapid PTT, toggle mode, silence after speech

If unique, capture with a descriptive name (e.g., `passing-long-high-coverage`, `passing-short-single-sentence`).

List existing fixtures first:
```bash
ls ~/Library/Application\ Support/Operator/test-fixtures/*.wav 2>/dev/null | while read f; do basename "$f" .wav; done
```

If the session is not meaningfully different from an existing fixture, skip capture and note why.

This builds a growing regression suite that covers both failure modes and successful transcription patterns.

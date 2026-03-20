# Speech-to-Text Architecture

> Living document. Update this when the transcription strategy changes.
> Last updated: 2026-03-20

## Overview

Operator uses a **two-engine, three-tier** transcription pipeline optimized for push-to-talk voice input. The system starts with Apple Speech (instant availability) and hot-swaps to WhisperKit (higher accuracy, on-device ML) once the model loads asynchronously.

```
                     ┌─────────────────────────────────────────────────────────┐
                     │                    Push-to-Talk Cycle                   │
                     │                                                         │
 FN Press ──► IDLE ──► LISTENING ──► TRANSCRIBING ──► ROUTING ──► DELIVERING  │
                     │      ▲              │                                   │
                     │      │         [silence?] ──► IDLE (dismissed)          │
                     │      │         [too short?] ──► IDLE (dismissed)        │
                     └──────┼──────────────────────────────────────────────────┘
                            │
                    Audio flows through:
                    Mic (48kHz) → Format Convert (16kHz) → Engine → Background Confirm
```

## Engine Lifecycle

### Startup Sequence

```
App launches
  ├─ AppleSpeechEngine created (instant, no download)
  ├─ StateMachine bootstrapped with AppleSpeechEngine
  ├─ User can start talking immediately
  │
  └─ Background task:
       ├─ Check for WhisperKit model on disk
       ├─ Download if missing (~39-461 MB depending on variant)
       ├─ Load WhisperKit pipeline (CoreML compilation)
       └─ Hot-swap: transcriber.replaceEngine(whisperKitEngine)
           (seamless, no user-visible transition)
```

### AppleSpeechEngine (Fallback)

**File:** `Sources/Voice/AppleSpeechEngine.swift`

- Uses `SFSpeechRecognizer` with on-device recognition (macOS 15+)
- Streams partial results during listening
- No word-level timestamps — just final text
- Timeout: 3s after `endAudio()` if partials were received, 1s if no partials
- `taskHint = .dictation` to preserve speech across pauses
- Supports `contextualStrings` for vocabulary biasing

### WhisperKitEngine (Primary)

**File:** `Sources/Voice/WhisperKitEngine.swift`

- On-device Whisper model via CoreML
- Word-level streaming confirmation during recording
- Three-tier decode strategy (see below)
- Supports `contextualStrings` via `VocabularyCorrector`

## Audio Capture Pipeline

**File:** `Sources/Voice/SpeechTranscriber.swift`

### Microphone → Engine

```
AVAudioEngine.inputNode
  │
  ├─ installTap(bufferSize: 1024, format: native)
  │
  ├─ [If multi-channel] Extract channel 0 (mono)
  │
  ├─► capturedBuffers (original format, for WAV trace)
  │
  └─► engine.append(buffer)
        └─ AudioFormatConverter: 48kHz → 16kHz mono Float32
             └─ Appends to SessionState.audioSamples
```

### Voice Processing: Intentionally Disabled

We do **not** use `setVoiceProcessingEnabled(true)`. Reasons:

1. **Push-to-talk = half-duplex.** The speaker is muted (`AudioQueue.userStartedSpeaking()` interrupts TTS) before recording starts. No echo to cancel.
2. **VP creates a system-wide aggregate audio device** that degrades other apps' audio quality and forces Bluetooth into low-quality HFP mode.
3. **The aggregate device persists** after `audioEngine.stop()` — only cleaned up when the process terminates.
4. **The feedback tone** (which does leak into the mic) is handled by the 0.4s RMS skip window in silence detection.

The mic mode picker (Standard/Voice Isolation/Wide Spectrum) only appears when VP is enabled. This is a known tradeoff — voice isolation is not available to users without VP.

### Audio Traces

Saved to `~/Library/Application Support/Operator/audio-traces/` as timestamped WAV files. Pruned to 20 most recent. Used for offline debugging via `stt-replay` benchmarks.

## WhisperKit Three-Tier Decode Strategy

### Tier 1: Word-Level Streaming Confirmation (during recording)

**Goal:** Lock in transcribed words while the user is still speaking, so the final decode only needs to handle the trailing unconfirmed tail.

**How it works:**

1. A background retranscription loop runs during recording:
   - Initial delay: 1000ms (battery friendly during silence)
   - After first pass: 500ms intervals (rapid early confirmation)
   - Subsequent: 1000ms intervals (reduced chatter)
   - Only triggers when ≥16,000 new samples (1s at 16kHz) available

2. Each background pass:
   - Decodes from `lastAgreedSeconds` to current end of audio
   - Passes `prefixTokens` from last agreed words (decoder continuity hint)
   - Uses `chunkingStrategy: .none` (short clips, no chunking needed)
   - Compares result words to previous pass's words

3. Word agreement algorithm:
   ```
   commonPrefix = longestCommonPrefix(prevWords, hypothesisWords)

   if commonPrefix.count >= 2:     // 2 consecutive words agree
       confirmedText += words before last 2
       lastAgreedWords = last 2 words
       lastAgreedSeconds = first agreed word's start time
       consecutiveMisses = 0
   else:
       consecutiveMisses += 1
   ```

4. **Why 2 words?** Single words flicker between passes ("the" ↔ "a"). Requiring 2 consecutive matching words ensures stability.

**State tracked:**
| Variable | Purpose |
|----------|---------|
| `confirmedText` | Locked-in text, never re-decoded |
| `lastAgreedWords` | Trailing 2 words waiting for next agreement |
| `lastAgreedSeconds` | Timestamp of confirmation frontier |
| `prevWords` | Previous pass's hypothesis for comparison |
| `consecutiveMisses` | Passes without agreement (triggers Tier 2 at 3) |
| `consecutiveZeroProgress` | Passes with agreement but no NEW words confirmed |

### Tier 2: Segment-Level Fallback (during recording)

**Trigger:** 3 consecutive background passes with no word agreement (`consecutiveMisses >= 3`).

**Why it exists:** Word-level timestamps can be unreliable on certain audio (background noise, music, overlapping speech). Segment-level timestamps are coarser but more stable.

**How it works:**
- Confirms all segments except the last (still forming)
- Clips subsequent passes from `segmentConfirmedEndSeconds`
- Set `useSegmentFallback = true` — changes the final decode path too

### Tier 3: Final Decode (after key release)

When the user releases the push-to-talk key, `finishAndTranscribe()` runs:

1. **Cancel background loop** and await its completion
2. **Snapshot** the stream state and audio samples
3. **Decode the unconfirmed tail:**

   **Word path** (normal):
   ```
   transcribeResult(
       samples: allSamples,
       clipStart: lastAgreedSeconds,
       prefixTokens: lastAgreedWords.tokens,  // unless zeroProgress > 0
       chunked: true                           // VAD segmentation
   )

   finalText = confirmedText + filteredTailWords
   ```

   **Segment path** (fallback):
   ```
   transcribeResult(
       samples: allSamples,
       clipStart: segmentConfirmedEndSeconds,
       chunked: true
   )

   finalText = segmentConfirmedText + tailText
   ```

4. **Safety net rescue** — if result has < 0.5 words/sec for audio > 3s:
   ```
   rescueResult = transcribeResult(samples: allSamples)  // full decode from 0.0s
   if rescueResult longer → use it
   ```

5. **Empty rescue** — if result is empty on audio > 1s:
   ```
   rescueResult = transcribeResult(samples: allSamples)  // full decode from 0.0s
   ```

6. **VocabularyCorrector** — edit-distance correction against session names and contextual strings

### Why VAD Chunking on Final Decode

The final decode uses `chunkingStrategy: .vad` while background passes use `.none`.

- Background passes decode short clips (typically 3-8s from the confirmation frontier) — always within Whisper's 30s window, no chunking needed
- The final tail can be much longer if confirmation stalled (28s+ observed)
- Without chunking, WhisperKit silently returns empty results on long audio
- VAD chunking splits at voice activity boundaries and decodes each chunk independently

Background passes must NOT use VAD chunking — the confirmation algorithm relies on consistent word timestamps relative to a single decode window.

## Silence Detection

**Two independent guards** prevent accidental transcription:

### 1. Minimum Recording Duration (StateMachine)

```swift
if elapsed < minimumRecordingDuration {  // 500ms
    feedback.play(.dismissed)
    enterIdle()
}
```

Catches accidental FN taps. Applied before any audio processing.

### 2. RMS Energy Check (SpeechTranscriber)

```swift
let rms = computeRMS(pendingBuffers, skippingSeconds: 0.4)
if rms < 0.008 {
    // Silence — skip transcription
}
```

- Skips first 0.4s to exclude the feedback tone from the calculation
- Threshold 0.008 — typical speech RMS is 0.02-0.2; background noise is well below 0.01
- Only affects the silence decision — full audio (including first 0.4s) still goes to WhisperKit

## Decoding Options

```swift
var options = DecodingOptions()
options.language = "en"
options.skipSpecialTokens = true
options.wordTimestamps = true
options.chunkingStrategy = chunked ? .vad : .none

if audioDuration < 3 {
    options.windowClipTime = 0      // Prevent dropping short audio
}
if clipStart > 0 {
    options.clipTimestamps = [clipStart]  // Skip confirmed audio
}
if !prefixTokens.isEmpty {
    options.prefixTokens = prefixTokens  // Decoder continuity hint
}
```

### Key Options Explained

| Option | Value | Why |
|--------|-------|-----|
| `windowClipTime = 0` | Only when audio < 3s | Default 1.0s padding prevents the decoder from processing the last 1s of the 30s window. For short audio, this causes the decode loop to never enter. |
| `clipTimestamps` | `[lastAgreedSeconds]` | Tells WhisperKit to start decoding from this timestamp, skipping already-confirmed audio. |
| `prefixTokens` | Last agreed words' tokens | Biases the decoder to continue from where confirmation left off. Dropped when `zeroProgress > 0` to avoid poisoning a stalled decode. |
| `chunkingStrategy: .vad` | Final decode only | VAD splits long audio at voice boundaries. Prevents silent failure on audio > 30s from clip point. |

## Whisper Hallucination Filtering

```swift
private static let blankPatterns: Set<String> = [
    "[BLANK_AUDIO]", "(BLANK_AUDIO)", "[silence]", "(silence)",
    "[no speech]", "(no speech)", "[inaudible]", "(inaudible)"
]

// Also filtered: anything matching /^\s*[\(\[].+[\)\]]\s*$/
// e.g., "(upbeat music)", "[electronic sound]"
```

WhisperKit hallucinates these pseudo-annotations on silence or very short audio. `cleanText()` strips them.

## Audio Format Conversion

**File:** `Sources/Voice/AudioFormatConverter.swift`

```
Input: AVAudioPCMBuffer at mic native rate (48kHz stereo)
Output: [Float] at 16kHz mono (WhisperKit's required format)
```

- Uses `AVAudioConverter` with closure-based input
- Reusable output buffer (allocated once, reused for stable tap buffer sizes)
- `converter.reset()` called before each conversion to clear internal state
- Ratio: `16000 / 48000 = 0.333...`
- Memory: ~480KB/second at 16kHz (30s = ~14.4MB)

## Feedback Tones

**File:** `Sources/Voice/AudioFeedback.swift`

| Cue | When | Purpose |
|-----|------|---------|
| `listening` | FN pressed → LISTENING | "I'm recording" |
| `processing` | Key released → TRANSCRIBING | "Working on it" |
| `dictation` | Text pasted | "Done, text inserted" |
| `delivered` | Message sent to agent | "Sent to agent" |
| `dismissed` | Too short / silence | "Nevermind" |
| `error` | Any error state | "Something went wrong" |

Minimum 300ms spacing between tones to prevent spam.

## Mute-on-Talk

When the user presses FN to start recording:

1. `AudioQueue.userStartedSpeaking()` interrupts any playing TTS
2. The speaker is silent during recording
3. On key release and transcription complete, `AudioQueue.userStoppedSpeaking()` resumes playback
4. If messages were queued during recording, a "pending" tone plays before resuming

This is why AEC/VP is unnecessary — the half-duplex design eliminates echo by construction.

## Known Limitations

1. **Long recordings (>30s from confirmation frontier):** If word confirmation stalls early and the user keeps talking, the unconfirmed tail grows. VAD chunking on the final decode handles this, but background passes still return "no words" on long tails because they use unchunked decoding.

2. **Apple Speech → WhisperKit transition:** If the user speaks before WhisperKit loads, Apple Speech handles it. The transition is seamless but Apple Speech has lower accuracy and no word timestamps.

3. **No Voice Isolation without VP:** macOS only shows the mic mode picker when Voice Processing is enabled. Since we disable VP to avoid the aggregate device problem, users can't select Voice Isolation from the system menu.

## Thresholds Reference

| Parameter | Value | Location |
|-----------|-------|----------|
| Minimum recording duration | 500ms | StateMachine |
| Silence RMS threshold | 0.008 | SpeechTranscriber |
| Feedback tone skip window | 0.4s | SpeechTranscriber |
| Background pass trigger | 16,000 samples (1s) | WhisperKitEngine |
| Word agreement count | 2 consecutive words | WhisperKitEngine |
| Word miss → segment fallback | 3 consecutive misses | WhisperKitEngine |
| Initial background delay | 1000ms | WhisperKitEngine |
| Post-first-pass interval | 500ms | WhisperKitEngine |
| Standard background interval | 1000ms | WhisperKitEngine |
| Short audio window clip | 0 (for audio < 3s) | WhisperKitEngine |
| Safety net words/sec | < 0.5 words/sec | WhisperKitEngine |
| Safety net min audio | > 3s | WhisperKitEngine |
| Tone spacing | 300ms minimum | AudioFeedback |
| Toggle tap threshold | 200ms | FNKeyTrigger |

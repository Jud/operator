# WhisperKit Reference for Operator

Research from source code at `Operator/.build/checkouts/WhisperKit/` (Argmax).

## Pipeline Safety

- **Reuse is safe**: Sequential `pipe.transcribe()` calls are safe. KV cache is explicitly reset between windows via `decoderInputs.reset()`. No state accumulates.
- **Concurrent calls corrupt**: Two concurrent `pipe.transcribe()` calls corrupt the shared KV cache. Must serialize all calls.

## DecodingOptions for Push-to-Talk

```swift
var options = DecodingOptions()
options.language = "en"              // Skip language detection
options.skipSpecialTokens = true     // Only vocabulary tokens in segment.text
options.chunkingStrategy = .none     // No VAD chunking for <30s audio
options.temperature = 0.0            // Greedy decoding (default)

// Short audio: disable end-of-window clip to allow <1s decoding
if audioDuration < 3 {
    options.windowClipTime = 0       // Default 1.0s prevents decoding audio <1s
}

// Streaming re-transcription
options.clipTimestamps = [lastConfirmedEndSeconds]
```

### Key Options

| Option | Default | Purpose |
|--------|---------|---------|
| `windowClipTime` | 1.0s | Clips end of 30s window to prevent hallucination. **Audio < 1s produces zero segments with default.** Set to 0 for short audio. |
| `compressionRatioThreshold` | 2.4 | Catches repetitive hallucination. Triggers temp fallback. |
| `logProbThreshold` | -1.0 | Catches low-confidence output. Triggers temp fallback. |
| `firstTokenLogProbThreshold` | -1.5 | Catches bad first token. **Triggers falsely with promptTokens (Issue #372).** |
| `noSpeechThreshold` | 0.6 | Should detect silence, but **noSpeechProb is hardcoded to 0** (TODO in TextDecoder.swift:993). Never triggers. |
| `suppressBlank` | false | Only suppresses whitespace + EOT at decode start. Not related to BLANK_AUDIO. |
| `temperatureIncrementOnFallback` | 0.2 | Each fallback adds 0.2 to temperature. |
| `temperatureFallbackCount` | 5 | Max fallback attempts. |
| `sampleLength` | 224 | Max tokens per window. Shared budget with prompt/prefix tokens. |

## Token Handling

### skipSpecialTokens

Filters tokens with ID >= `specialTokenBegin` (50257) from **segment text only**. The segment's `tokens` array always contains all tokens.

Special tokens filtered: `<|endoftext|>`, `<|startoftranscript|>`, language tags, `<|transcribe|>`, `<|translate|>`, `<|startofprev|>`, `<|nospeech|>`, `<|notimestamps|>`, all timestamp tokens `<|0.00|>` through `<|30.00|>`.

**Source**: SegmentSeeker.swift:120-122

### BLANK_AUDIO

**Not a WhisperKit token.** Regular vocabulary text hallucinated by Whisper for silence/short audio. WhisperKit has no built-in filtering. Known patterns: `[BLANK_AUDIO]`, `(silence)`, `[inaudible]`, etc. Must be filtered manually.

### Prompt Tokens

Placed as `[startOfPreviousToken] + tokens` before `[SOT, lang, task, timestamp]`. Simulates context from a "previous segment."

**Correct usage** (from CLI):
```swift
tokenizer.encode(text: " " + text.trimmingCharacters(in: .whitespaces))
    .filter { $0 < tokenizer.specialTokens.specialTokenBegin }
```

**Known issues**:
- Prefill cache is disabled when promptTokens are set (adds latency)
- firstTokenLogProbThreshold triggers falsely (Issue #372)
- File paths / structured data contaminate decoder output under fallback
- Budget: max 111 tokens (maxTokenContext/2 - 1)

**Best practice**: Short, natural-language vocabulary hints only. No paths, URLs, or structured data.

### Prefix Tokens

Different from promptTokens. Placed AFTER `[SOT, lang, task, timestamp]`. Forces the decoder to output these tokens first. Used in simulated streaming to force last-agreed words.

## Segment Quality Metrics

Each `TranscriptionSegment` has:

| Metric | What it means | Threshold |
|--------|--------------|-----------|
| `avgLogprob` | Average log probability of word tokens. Lower = less confident. | < -1.0 triggers fallback |
| `compressionRatio` | Repetitiveness (zlib ratio of token IDs). Higher = more repetitive. | > 2.4 triggers fallback |
| `noSpeechProb` | Probability of no speech. **Always 0 — not implemented.** | 0.6 (never triggers) |
| `temperature` | Decoding temperature used. 0.0 = greedy, >0 = fallback occurred. | Informational |
| `tokenLogProbs` | Per-token `[tokenId: logProb]` maps. | Per-token confidence |

**Fallback order** (checked after each window decode):
1. `firstTokenLogProbThreshold` — first token too bad → immediate fallback
2. `noSpeechThreshold` — silence detected → skip (never triggers, noSpeechProb=0)
3. `compressionRatioThreshold` — too repetitive → fallback with temp+0.2
4. `logProbThreshold` — too uncertain → fallback with temp+0.2

After all fallbacks exhausted, best result is used regardless of quality.

## Streaming Approaches

### 1. AudioStreamTranscriber (built-in actor)

- Re-transcribes **entire** accumulated buffer each pass
- Uses `clipTimestamps=[lastConfirmedEndSeconds]` to skip confirmed audio
- Confirms when `segments.count > requiredSegmentsForConfirmation` (default 2)
- Has VAD (energy-based, silenceThreshold=0.3)
- Has early-stop on compressionRatio (window=60 tokens) and avgLogprob

### 2. Simulated Streaming (CLI)

- Processes 1-second chunks incrementally
- Uses **word-level agreement** via `wordTimestamps` for confirmation
- Compares words between consecutive passes using `findLongestCommonPrefix()`
- Confirms when `agreementCountNeeded` (2) consecutive words match
- Uses both `clipTimestamps` and `prefixTokens` for context

### 3. Operator's Approach

- Custom background loop with adaptive timing (1s initial, 500ms, 1s)
- Segment-count-based confirmation (2+ segments, confirm all but last)
- clipTimestamps for skipping confirmed audio
- Pipeline drain between sessions

## Audio Constraints

- **Sample rate**: 16kHz mono Float32
- **Window**: 30s (480000 samples), zero-padded
- **Mel spectrogram**: 80 channels, hop_length=160, 3000 time steps per 30s
- **Timestamp resolution**: 20ms per token
- **Minimum audio**: ~1s with default windowClipTime=1.0. Set windowClipTime=0 for shorter.
- **Max tokens per window**: 224 (shared with prompt/prefix tokens)

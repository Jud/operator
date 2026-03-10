# Plan: speech-swift Integration

Replace AVSpeechSynthesizer (TTS) and SFSpeechRecognizer (STT) with speech-swift's Qwen3-TTS and Parakeet CoreML engines for dramatically better voice quality, all running locally on Apple Silicon.

## Why speech-swift

- Single SPM dependency for both STT and TTS
- ~120ms streaming TTS latency (vs AVSpeechSynthesizer's ~500ms+)
- Parakeet CoreML STT runs on Neural Engine (~400MB, fast)
- ~2.4GB combined peak memory — fine for 8GB Macs
- No API keys, no Python, no subprocess
- Bonus: Silero VAD + noise suppression included

## Blocker: Hummingbird Version Conflict

speech-swift pins Hummingbird to `"2.5.0"..<"2.17.0"`. Operator resolves to 2.20.1.
The library targets (Qwen3ASR, Qwen3TTS, ParakeetASR) don't use Hummingbird — only AudioServer/AudioServerCLI do.

**Fix:** Fork speech-swift, remove AudioServer + AudioServerCLI targets and Hummingbird dependency.

## Phase 1: Fork & SPM Integration (Days 1-2)

- [ ] Fork `soniqo/speech-swift`
- [ ] Remove `AudioServer`, `AudioServerCLI` targets from Package.swift
- [ ] Remove Hummingbird + HummingbirdWebSocket package dependencies
- [ ] Add fork to Operator's Package.swift: `import Qwen3TTS, ParakeetASR`
- [ ] Verify clean SPM resolution with existing Hummingbird 2.20.1

## Phase 2: STT Integration (Days 3-4)

- [ ] Create `ParakeetTranscriptionEngine` conforming to `TranscriptionEngine`
  - `prepare()` → load/warm Parakeet CoreML model
  - `append(_:)` → accumulate audio buffers
  - `finishAndTranscribe()` → run inference, return text
  - `cancel()` → abort recognition
- [ ] Wire into `SpeechTranscriber` as a drop-in replacement for `AppleSpeechEngine`
- [ ] Test transcription quality with push-to-talk flow
- [ ] Keep `AppleSpeechEngine` as fallback (selectable in settings)

## Phase 3: TTS Integration (Days 5-8)

### Protocol Changes

- [ ] Define `VoiceDescriptor` to replace `AVSpeechSynthesisVoice`:
  ```swift
  public enum VoiceDescriptor: Sendable {
      case qwen3(speakerID: String)
      case system(AVSpeechSynthesisVoice) // fallback
  }
  ```
- [ ] Update `SpeechManaging` protocol: replace `AVSpeechSynthesisVoice` param with `VoiceDescriptor`
- [ ] Update `AudioQueue.QueuedMessage` voice field
- [ ] Update `VoiceManager` to return `VoiceDescriptor` instead of `AVSpeechSynthesisVoice`

### New Speech Manager

- [ ] Create `Qwen3SpeechManager` conforming to `SpeechManaging`
  - Uses `AVAudioEngine` + `AVAudioPlayerNode` for playback
  - Receives streaming PCM chunks from Qwen3-TTS
  - Schedules buffers for near-instant playback start
- [ ] Implement playback-position-based interruption tracking:
  - Track elapsed playback time via `AVAudioPlayerNode.lastRenderTime`
  - On interrupt: compute heard fraction, split text at nearest word boundary
  - Return `InterruptInfo` with estimated heard/unheard split

### Voice Management

- [ ] Map session voices to Qwen3-TTS speaker IDs (9 built-in speakers)
- [ ] Keep pitch multiplier for per-agent differentiation
- [ ] "Operator" system voice gets a distinct speaker ID

## Phase 4: Model Lifecycle (Days 9-10)

- [ ] First-run model download with progress indicator in Settings panel
- [ ] Cache models in `~/Library/Caches/com.operator.app/models/`
- [ ] Lazy loading: STT model on first PTT press, TTS model on first speak
- [ ] Memory management: unload STT after transcription completes, keep TTS warm
- [ ] Graceful fallback to Apple engines if models unavailable

## Phase 5: Polish (Days 11-12)

- [ ] A/B test Qwen3-TTS vs CosyVoice3 for short utterance quality
- [ ] Add engine selection toggle in Settings (Apple / Qwen3)
- [ ] Update E2E tests with mock for new engine
- [ ] Performance profiling on M1 8GB baseline

## Files to Modify

| File | Change |
|------|--------|
| `Package.swift` | Add speech-swift fork dependency |
| `Voice/TranscriptionEngine.swift` | No change (protocol fits) |
| `Voice/SpeechManaging.swift` | Replace `AVSpeechSynthesisVoice` with `VoiceDescriptor` |
| `Voice/SpeechManager.swift` | Keep as `AppleSpeechManager` fallback |
| `Voice/AudioQueue.swift` | Update `QueuedMessage.voice` type |
| `Voice/VoiceManager.swift` | Return `VoiceDescriptor`, map to speaker IDs |
| `Voice/SpeechTranscriber.swift` | Accept generic `TranscriptionEngine` (already does) |
| `App/OperatorApp.swift` | Wire new engines based on settings |
| `UI/SettingsView.swift` | Add engine selection toggle |

## New Files

| File | Purpose |
|------|---------|
| `Voice/ParakeetTranscriptionEngine.swift` | Parakeet CoreML STT wrapper |
| `Voice/Qwen3SpeechManager.swift` | Qwen3-TTS playback via AVAudioEngine |
| `Voice/VoiceDescriptor.swift` | Generic voice type enum |

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Hummingbird fork conflict | Low | Simple Package.swift edit — remove 2 targets + 2 deps |
| Qwen3-TTS quality for short utterances | Low-Med | CosyVoice3 is alternative in same package |
| First-run model download UX | Medium | Progress UI in Settings; could pre-download via Homebrew |
| Memory pressure on 8GB Macs | Medium | Use Parakeet CoreML (~400MB) not Qwen3-ASR (~2.2GB) |
| Interruption tracking less precise | Low | Audio-position tracking within ±2 words is sufficient |

## Success Criteria

- TTS subjectively "natural sounding" (target: 8/10 preference vs AVSpeech)
- Time-to-first-audio < 200ms for 1-sentence utterances
- STT accuracy ≥ 95% for clear English at normal pace
- Total memory < 3GB with both models loaded
- No latency regression in PTT → transcription → routing pipeline

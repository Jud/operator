# Transcription Engine Design

> Target architecture for the refactored transcription pipeline.
> This document describes where we want to end up, not the current state.
> See `docs/stt-architecture.md` for the current implementation.

## Goals

1. **Testable confirmation logic** — the confirmation algorithm is a pure state machine with explicit modes, testable with synthetic inputs, no WhisperKit or audio dependency
2. **Deterministic replay** — tests replay exact transcription results from production session logs through the state machine, no inference variance
3. **Continuous quality measurement** — debug builds automatically capture ground truth for every session, building a growing fixture library and tracking accuracy over time
4. **Extensible strategies** — adding a new confirmation or post-decode strategy requires changes only in the state machine, never in the coordinator
5. **Same production behavior** — the refactor changes structure, not behavior

## Components

### ConfirmationState

A pure value type. No async, no locks, no framework dependencies. All policy decisions about what text is confirmed, where to clip, and how to assemble the final result live here — not in the coordinator.

```swift
struct ConfirmationState {
    /// The current confirmation mode. Explicit enum prevents invalid
    /// combinations of boolean flags.
    enum Mode: Equatable {
        /// Word-level agreement (default). Tracks previous hypothesis
        /// for longest-common-prefix comparison.
        case wordLevel(consecutiveMisses: Int)

        /// Segment-level fallback. Entered after 3 consecutive word
        /// misses. One-way transition — does not return to word-level.
        case segmentLevel
    }

    // Configuration (injectable for testing)
    let agreementCountNeeded: Int  // default: 2
    let maxWordMisses: Int         // default: 3

    // Confirmed output
    private(set) var confirmedText: String = ""
    private(set) var mode: Mode = .wordLevel(consecutiveMisses: 0)

    // Word-level state (active in .wordLevel mode)
    private(set) var lastAgreedWords: [WordTiming] = []
    private(set) var lastAgreedSeconds: Float = 0
    private(set) var prevWords: [WordTiming] = []
    private(set) var hasZeroProgress: Bool = false

    // Segment-level state (active in .segmentLevel mode)
    private(set) var segmentConfirmedText: String = ""
    private(set) var segmentConfirmedEndSeconds: Float = 0

    /// Process a background pass result. Returns what happened.
    mutating func processBackgroundResult(
        _ result: TranscriptionResult,
        audioDuration: Float
    ) -> PassOutcome

    /// Compute parameters for the next background pass.
    var nextPassParameters: PassParameters { get }

    /// Compute parameters for the final tail decode.
    func finalDecodeParameters() -> FinalDecodeParams

    /// Assemble the final text from confirmed content and tail decode result.
    /// All policy about word filtering, segment fallback text, and
    /// empty-result handling lives here — not in the coordinator.
    func assembleResult(tailResult: TranscriptionResult?) -> String
}
```

**Responsibilities:**
- Word agreement (longest common prefix, configurable threshold)
- Mode transitions (word-level → segment fallback, one-way by design)
- Zero-progress tracking (agreement without advancement)
- Timestamp clamping (never exceed audio duration)
- Computing clip point and prefix tokens for both background and final passes
- Assembling the final result from confirmed text + tail decode output
- Filtering hallucinated words (timestamps past audio end)
- Cleaning blank/annotation text (`[BLANK_AUDIO]`, `(upbeat music)`, etc.)

**Does NOT:**
- Call WhisperKit
- Manage audio buffers
- Schedule anything
- Hold locks or run async

### PassParameters

What the coordinator needs to run any decode pass (background or final).

```swift
struct PassParameters {
    let clipStart: Float
    let prefixTokens: [Int]
    let chunked: Bool
}
```

The coordinator never computes these itself — it always asks the state machine.

### PassOutcome

Returned by `processBackgroundResult`. Must be `Equatable` for test assertions.

```swift
enum PassOutcome: Equatable {
    case confirmed(newWords: Int, agreedSeconds: Float)
    case zeroProgress
    case miss(count: Int)
    case segmentConfirmed(newSegments: Int, endSeconds: Float)
    case transitionedToSegmentFallback
    case noWords          // result had segments but no word timestamps
    case noResult         // WhisperKit returned nil
}
```

### FinalDecodeParams

What the coordinator needs to run the tail decode. Superset of `PassParameters` with assembly context.

```swift
struct FinalDecodeParams {
    let clipStart: Float
    let prefixTokens: [Int]
    let chunked: Bool         // true → VAD chunking on final decode
    let confirmedText: String
}
```

### TranscriptionScheduler

Protocol that controls when background passes run. The engine doesn't own the timing — the scheduler does.

**Correctness invariant:** `stop()` must not return until the last in-flight pass has completed. After `stop()` returns, no further calls to `runScheduledPass()` will occur. This guarantees safe, unsynchronized access to `ConfirmationState` from `finishAndTranscribe()`.

```swift
protocol TranscriptionScheduler: Sendable {
    /// Start scheduling background passes.
    func start(engine: TranscriptionEngine) async

    /// Stop scheduling. Must not return until the last pass completes.
    func stop() async
}
```

#### BackgroundLoopScheduler (production)

Replaces the current `startRetranscriptionLoop` method. Same behavior:

- Initial 1-second delay
- Check if ≥16,000 new samples since last pass
- Run pass, then sleep 500ms (first pass) or 1000ms (subsequent)
- Interruptible sleep so `stop()` returns immediately after the current pass

```swift
final class BackgroundLoopScheduler: TranscriptionScheduler {
    static let retranscribeThreshold = 16_000  // 1s at 16kHz
    static let initialDelay: UInt64 = 1_000_000_000
    static let firstPassInterval: UInt64 = 500_000_000
    static let standardInterval: UInt64 = 1_000_000_000

    func start(engine: TranscriptionEngine) async { ... }
    func stop() async { ... }
}
```

#### ManifestReplayScheduler (tests)

Drives the engine from a JSON manifest captured from real session logs. Each entry specifies the exact sample count at which a pass should run.

`runScheduledPass` accepts a `maxSampleCount` parameter so the snapshot is clamped to exactly what the production session had — prevents the test from seeing more audio than the live pass did.

```swift
final class ManifestReplayScheduler: TranscriptionScheduler {
    struct PassEntry: Codable {
        let sampleCount: Int
        let offsetMs: Int
    }

    let entries: [PassEntry]

    func start(engine: TranscriptionEngine) async {
        for entry in entries {
            await engine.waitForSamples(entry.sampleCount)
            await engine.runScheduledPass(maxSampleCount: entry.sampleCount)
        }
    }

    func stop() async { }
}
```

### WhisperKitEngine (coordinator)

Thin glue. Owns resources, wires components together. Contains **no policy logic** — all decisions about clip points, prefix tokens, word filtering, and result assembly are delegated to `ConfirmationState`.

```swift
final class WhisperKitEngine: TranscriptionEngine {
    private let pipe: WhisperKit
    private let samplesLock = OSAllocatedUnfairLock<[Float]>([])
    private var converter: AudioFormatConverter?
    private var confirmation = ConfirmationState()
    private var scheduler: TranscriptionScheduler

    // Called by SpeechTranscriber on each mic buffer (audio callback thread)
    func append(_ buffer: AVAudioPCMBuffer)

    // Called by the scheduler when it's time for a background pass
    func runScheduledPass(maxSampleCount: Int? = nil) async {
        let samples = samplesLock.withLock { s in
            if let max = maxSampleCount {
                return Array(s.prefix(max))
            }
            return s
        }

        let params = confirmation.nextPassParameters
        let result = await transcribe(
            samples: samples,
            clipStart: params.clipStart,
            prefixTokens: params.prefixTokens,
            chunked: params.chunked
        )

        let audioDuration = Float(samples.count) / Float(WhisperKit.sampleRate)
        let outcome = confirmation.processBackgroundResult(
            result, audioDuration: audioDuration
        )
        log(outcome)
    }

    // Called when user releases push-to-talk key
    func finishAndTranscribe() async -> String? {
        await scheduler.stop()

        let samples = samplesLock.withLock { $0 }
        let audioDuration = Float(samples.count) / Float(WhisperKit.sampleRate)
        let params = confirmation.finalDecodeParameters()

        let tailResult = await transcribe(
            samples: samples,
            clipStart: params.clipStart,
            prefixTokens: params.prefixTokens,
            chunked: params.chunked
        )

        var fullText = confirmation.assembleResult(tailResult: tailResult)

        // Safety net: rescue if too short
        if audioDuration > 3 && fullText.wordsPerSecond(audioDuration) < 0.5 {
            if let rescue = await transcribe(samples: samples) {
                let rescueText = cleanText(rescue.text)
                if rescueText.count > fullText.count {
                    fullText = rescueText
                }
            }
        }

        // Empty rescue
        if fullText.isEmpty && audioDuration > 1 {
            if let rescue = await transcribe(samples: samples) {
                fullText = cleanText(rescue.text)
            }
        }

        guard !fullText.isEmpty else { return nil }
        return VocabularyCorrector.correct(text: fullText, vocabulary: vocabulary)
    }
}
```

### SpeechTranscriber (unchanged)

Owns `AVAudioEngine`, captures mic audio, handles silence detection. Calls `engine.prepare()`, `engine.append()`, and `engine.finishAndTranscribe()`. No changes needed.

### StateMachine (unchanged)

Orchestrates the overall flow (IDLE → LISTENING → TRANSCRIBING → ROUTING → DELIVERING). Doesn't know about confirmation strategies or scheduling.

## Concurrency Model

Two synchronization boundaries:

1. **`samplesLock`** — protects the `[Float]` audio buffer. `append()` writes from the audio callback thread; `runScheduledPass()` and `finishAndTranscribe()` read snapshots.

2. **`ConfirmationState` has no lock** — synchronization is structural. The scheduler guarantees serial access: only one pass runs at a time, and `stop()` completes before `finishAndTranscribe()` touches the state. This is a correctness invariant of the `TranscriptionScheduler` protocol.

`append()` and `runScheduledPass()` access different state — `samplesLock` for audio, `ConfirmationState` for confirmation. No cross-lock dependencies.

## Continuous Quality Measurement

### Debug Session Recording

On debug builds, every push-to-talk session automatically captures a complete record:

```swift
// In finishAndTranscribe(), after the normal pipeline:
#if DEBUG
Task.detached { [samples, confirmation, fullText] in
    // Run a clean full-audio batch decode in the background
    let groundTruth = await transcribe(samples: samples)

    // Record the session
    SessionRecorder.record(SessionRecord(
        streamingResult: fullText,
        groundTruth: cleanText(groundTruth?.text ?? ""),
        confirmationLog: confirmation.passLog,
        audioDuration: audioDuration,
        commit: BuildInfo.gitCommit,
        modelVariant: modelVariant
    ))
}
#endif
```

This runs the full decode in a detached task — no latency impact on the user. The ground truth is always available, even for sessions that transcribed correctly.

### Session Records

Stored in `~/Library/Application Support/Operator/session-records/`:

```json
{
    "timestamp": "2026-03-20T10:21:54-05:00",
    "commit": "172bc8a",
    "branch": "feat/stt-word-level-streaming",
    "modelVariant": "openai_whisper-large-v3_turbo",
    "audioDuration": 78.8,
    "streamingResult": "I guess the question...",
    "groundTruth": "I guess the question that I have is like this still feels...",
    "similarity": 0.62,
    "confirmationLog": [
        {
            "offsetMs": 1694,
            "sampleCount": 22400,
            "outcome": "miss",
            "agreedSeconds": 0.0,
            "words": [["Hello", 0.0, 0.5], ["world", 0.5, 1.0]]
        }
    ]
}
```

### Automatic Fixture Promotion

When a session's similarity drops below a threshold (e.g., 80%), it's automatically flagged:

```
[TranscriptionQuality] Session diverged: 62% similarity
  Streaming: "I guess the question..."
  Ground truth: "I guess the question that I have is like..."
  Saved to: ~/Library/.../session-records/2026-03-20T10-21-54.json
  Promote to fixture: ./scripts/promote-fixture.sh 2026-03-20T10-21-54 timestamp-drift
```

The `promote-fixture.sh` script copies the WAV and ground truth into the test-fixtures directory, ready to become a regression test.

### Quality Dashboard

Session records accumulate over time. A simple report shows trends:

```bash
./scripts/quality-report.sh

Transcription Quality Report (last 7 days)
  Sessions:    247
  Mean similarity: 94.2%
  Sessions < 85%:  12 (4.9%)
  Sessions < 70%:  3 (1.2%)

  Worst sessions:
    2026-03-20T10:21:54  62%  timestamp-drift (promoted to fixture)
    2026-03-20T09:17:23  47%  short-utterance-loss (promoted to fixture)
    2026-03-19T15:28:28  44%  long-recording-stall (promoted to fixture)
```

## Testing Architecture

### Level 1: Algorithmic Tests (ConfirmationState)

Pure unit tests. No audio, no WhisperKit, no timing. Fast and deterministic. These are Swift `XCTestCase` tests in the `OperatorTests` target.

```swift
func testWordConfirmation() {
    var state = ConfirmationState()

    let pass1 = makeResult(words: [("Hello", 0.0, 0.5), ("world", 0.5, 1.0)])
    let outcome1 = state.processBackgroundResult(pass1, audioDuration: 2.0)
    XCTAssertEqual(outcome1, .miss(count: 1))  // first pass, no prev to compare

    let pass2 = makeResult(words: [("Hello", 0.0, 0.5), ("world", 0.5, 1.0)])
    let outcome2 = state.processBackgroundResult(pass2, audioDuration: 3.0)
    XCTAssertEqual(outcome2, .zeroProgress)
    // 2 words agree but nothing beyond them to confirm
}

func testTimestampDriftPrevention() {
    var state = ConfirmationState()
    // Set up: confirmed to 6.0s with agreed words ["We", "need..."]
    // ...

    // Feed hallucinated words with timestamps past audio end
    let hallucinated = makeResult(words: [
        ("We", 93.9, 94.5),
        ("need...", 94.5, 95.0)
    ])
    let outcome = state.processBackgroundResult(hallucinated, audioDuration: 78.8)
    XCTAssertEqual(state.lastAgreedSeconds, 6.0)  // must NOT drift
}

func testSegmentFallbackTransition() {
    var state = ConfirmationState(maxWordMisses: 3)
    // Feed 3 non-matching passes
    state.processBackgroundResult(makeResult(words: [("a", 0, 1)]), audioDuration: 5)
    state.processBackgroundResult(makeResult(words: [("b", 0, 1)]), audioDuration: 6)
    state.processBackgroundResult(makeResult(words: [("c", 0, 1)]), audioDuration: 7)
    XCTAssertEqual(state.mode, .segmentLevel)
}

func testZeroProgressDropsPrefixTokens() {
    var state = ConfirmationState()
    // Set up zero-progress state
    // ...
    let params = state.finalDecodeParameters()
    XCTAssertTrue(params.prefixTokens.isEmpty)
}

func testCleanTextFiltersHallucinations() {
    XCTAssertEqual(ConfirmationState.cleanText("[BLANK_AUDIO]"), "")
    XCTAssertEqual(ConfirmationState.cleanText("(upbeat music)"), "")
    XCTAssertEqual(ConfirmationState.cleanText("Hello world"), "Hello world")
}

func testAssembleResultFiltersWordsPastClipPoint() {
    // ...
}
```

**What these catch:** Timestamp drift, mode transition bugs, zero-progress handling, hallucination filtering, result assembly edge cases.

**Replay from session records:** Level 1 tests can also replay the `confirmationLog` from session records. Each entry contains the exact words and timestamps from a production pass. This is fully deterministic — no WhisperKit inference.

```swift
func testReplayTimestampDriftSession() {
    let record = loadSessionRecord("timestamp-drift")
    var state = ConfirmationState()

    for entry in record.confirmationLog {
        let result = makeResult(from: entry)
        state.processBackgroundResult(result, audioDuration: entry.audioDuration)
    }

    // Verify the state machine didn't drift
    XCTAssertLessThanOrEqual(state.lastAgreedSeconds, Float(record.audioDuration))

    // Verify the result assembly produces good output
    let params = state.finalDecodeParameters()
    XCTAssertLessThanOrEqual(params.clipStart, Float(record.audioDuration))
}
```

### Level 2: Integration Tests (Streaming Pipeline)

Full engine with real WhisperKit, audio replay from WAV fixtures. Compares output against ground truth. These are probabilistic — WhisperKit inference is non-deterministic across runs.

```
Fixtures: ~/Library/Application Support/Operator/test-fixtures/
  <name>.wav              # Audio recording
  <name>.expected.txt     # Ground truth (from batch decode)
  <name>.meta.json        # Commit, branch, timestamp, model variant
```

Threshold: ≥85% Levenshtein similarity = pass.

**What these catch:** End-to-end pipeline issues, format conversion problems, WhisperKit version regressions, safety net effectiveness, VAD chunking behavior.

### Level 3: Manifest Replay Tests

Uses `ManifestReplayScheduler` to feed audio at exact sample counts from production logs. Closer to reality than Level 2 (controlled timing) but still uses live WhisperKit inference (non-deterministic).

For **fully deterministic** replay, use the Level 1 session record replay instead — it feeds recorded `TranscriptionResult` data through `ConfirmationState` without running inference.

**What these catch:** Timing-dependent bugs where the background loop fires at specific moments. Races between audio accumulation and pass scheduling.

## Confirmation Strategies

The `ConfirmationState` handles strategies through its `processBackgroundResult` method. The mode enum makes the active strategy explicit.

### Adding a New Strategy

1. Add a case to `ConfirmationState.Mode`
2. Add a transition condition in `processBackgroundResult`
3. Add the processing logic
4. Update `nextPassParameters` and `finalDecodeParameters` for the new mode
5. Update `assembleResult` if the new mode changes how text is combined
6. Add algorithmic test fixtures

The coordinator never branches on mode — it always asks the state machine for parameters. This is the key extensibility property.

### Current Strategies

| Strategy | Mode | Trigger | Confirmation Unit | Clip Point |
|----------|------|---------|-------------------|------------|
| **Word-level** | `.wordLevel` | Default | 2 consecutive agreed words | `lastAgreedSeconds` |
| **Segment-level** | `.segmentLevel` | 3 word misses | All segments except last | `segmentConfirmedEndSeconds` |

Transition: `.wordLevel` → `.segmentLevel` is **one-way by design**. Once word timestamps prove unreliable (3 misses), we don't trust them for the remainder of the session. A future strategy could add a recovery path if there's evidence word timestamps have stabilized.

### Post-Decode Steps (coordinator)

These run after the state machine produces its result. They're in the coordinator because they involve calling WhisperKit again.

| Step | Trigger | Action |
|------|---------|--------|
| **VAD chunking** | Always on final decode | `chunkingStrategy: .vad` |
| **Safety net** | < 0.5 words/sec on audio > 3s | Full decode from 0.0s |
| **Empty rescue** | Empty result on audio > 1s | Full decode from 0.0s |
| **Vocabulary correction** | Always | Edit-distance matching |

### Planned Post-Decode Steps

| Step | Trigger | Action |
|------|---------|--------|
| **LLM correction** | Configurable | Local CoreML model text refinement |
| **Context-aware formatting** | Active app detection | Adapt tone/format per target app |

## Data Flow

```
┌────────────────────────────────────────────────────────────────┐
│ SpeechTranscriber                                              │
│                                                                │
│  AVAudioEngine ──► mic buffer ──► engine.append(buffer)        │
│                          │                                     │
│                          └──► capturedBuffers (WAV trace)       │
└────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────┐
│ WhisperKitEngine (coordinator)                                 │
│                                                                │
│  ┌──────────────┐   ┌────────────────┐   ┌──────────────────┐ │
│  │ Audio Buffer  │   │ WhisperKit     │   │ Confirmation     │ │
│  │ (samplesLock) │   │ Pipeline       │   │ State            │ │
│  │               │──►│ (CoreML)       │──►│ (pure struct)    │ │
│  │ [Float] 16kHz │   │ .transcribe()  │   │ .processResult() │ │
│  └──────────────┘   └────────────────┘   │ .assembleResult() │ │
│         ▲                                 │ .nextPassParams   │ │
│         │                                 │ .finalDecodeParams│ │
│         │                                 └──────────────────┘ │
│  ┌──────┴──────────────────────────────────────────┐           │
│  │  Scheduler (protocol)                           │           │
│  │  ├── BackgroundLoopScheduler (production)       │           │
│  │  └── ManifestReplayScheduler (tests)            │           │
│  │                                                 │           │
│  │  "Run a pass now" ──► runScheduledPass()        │           │
│  └─────────────────────────────────────────────────┘           │
│                                                                │
│  finishAndTranscribe():                                        │
│    1. scheduler.stop()                                         │
│    2. confirmation.finalDecodeParameters()                     │
│    3. WhisperKit decode (params from state machine)            │
│    4. confirmation.assembleResult(tailResult)                  │
│    5. Safety net rescue (if too short)                          │
│    6. VocabularyCorrector                                      │
│    7. [DEBUG] Background ground truth decode                   │
│    8. Return text                                              │
└────────────────────────────────────────────────────────────────┘
```

## Migration Path

The refactor can be done incrementally. Each step is independently shippable and testable.

### Step 1: Extract ConfirmationState

Move `confirmWords`, `confirmSegments`, `StreamState` fields, and the `assembleResult` logic (currently in `decodeFinalText`) into a separate `ConfirmationState.swift` file. Replace `useSegmentFallback` boolean with `Mode` enum. Add `nextPassParameters` and `finalDecodeParameters` computed properties. Move threshold constants (`agreementCountNeeded`, `maxWordMisses`) into the struct as injectable configuration.

Write Level 1 algorithmic tests including the timestamp drift regression test.

### Step 2: Extract TranscriptionScheduler

Pull the background loop into `BackgroundLoopScheduler`. Add `runScheduledPass(maxSampleCount:)` to the engine. The coordinator stops branching on mode for clip/prefix computation — it uses `confirmation.nextPassParameters` instead.

Production behavior unchanged.

### Step 3: Add Session Recording (debug builds)

Add the background ground truth decode in `finishAndTranscribe`. Write session records with confirmation logs. Add the `promote-fixture.sh` script and quality report.

### Step 4: Add ManifestReplayScheduler

Build manifest capture from session records. Add Level 3 tests. Add Level 1 session record replay tests (deterministic, no inference).

### Step 5: Clean Up Coordinator

Remove all mode-branching from the coordinator. `finishAndTranscribe` becomes a simple pipeline: get params → decode → assemble → safety net → correct. All policy lives in `ConfirmationState`.

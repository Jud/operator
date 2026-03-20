# Transcription Engine Interface Contract

> Approved. These signatures are the contract that parallel
> implementation agents build against.

## Glossary

| Term | Meaning |
|------|---------|
| **Transcript** | The locked-in text. Grows as words are confirmed. |
| **Frontier** | Where confirmed audio ends (seconds). Audio before the frontier is settled. |
| **Working window** | The audio region from the frontier to the live edge — what's being transcribed. |
| **Window transcription** | What WhisperKit produces from the working window. Compared for boundary matching. |
| **Boundary matching** | Comparing two consecutive window transcriptions to find overlapping words. When they match, the transcript is extended. |
| **Match failure** | Two window transcriptions didn't agree at the boundary. The frontier doesn't move. |
| **Overlap tokens** | Decoder tokens from the last matched boundary words. Used by the coordinator to hint WhisperKit for continuity. |
| **Filter pipeline** | A chain of filters that clean transcription output before it reaches the session or the final result. Runs in the coordinator. |
| **Session** | One push-to-talk activation — from FN press to text delivery. |

```
Audio stream:

[========= transcript =========][=== working window ===][incoming...]
0s                             6.0s                    8.5s
                                ^                        ^
                             frontier                 live edge
```

## Types

### ExtensionOutcome

What happened when the session tried to extend the transcript.

```swift
public enum ExtensionOutcome: Equatable, Sendable {
    /// Transcript was extended. newWords is the number of words
    /// locked into the transcript. May be 0 if boundaries matched
    /// but there was nothing new beyond the match point.
    case extended(newWords: Int)

    /// The window transcription didn't match the previous one
    /// at the boundary. The frontier doesn't move.
    case matchFailed
}
```

Usage:

```swift
let outcome = session.extendTranscript(with: filtered)
switch outcome {
case .extended(let n): log("Extended by \(n) words, frontier at \(session.frontier)")
case .matchFailed:     log("No match, \(session.consecutiveMatchFailures) failures")
}
```

### MatchingStrategy

```swift
public enum MatchingStrategy: Equatable, Sendable {
    /// Word-level boundary matching. Default.
    /// Uses longest-common-prefix on word arrays. Requires
    /// matchThreshold consecutive matching words for agreement.
    case wordLevel

    /// Segment-level boundary matching. Entered after maxMatchFailures
    /// consecutive failures. One-way — does not return to wordLevel.
    ///
    /// Confirms all-but-last segments (the last segment is still forming).
    /// Advances the frontier to the end of the last confirmed segment.
    /// Does not require word-level timestamps — operates on segment
    /// boundaries and segment text only.
    case segmentLevel
}
```

### TranscriptionSessionConfig

```swift
public struct TranscriptionSessionConfig: Equatable, Sendable {
    /// Number of consecutive matching words required at the boundary.
    public var matchThreshold: Int = 2

    /// Consecutive match failures before switching to segment-level matching.
    public var maxMatchFailures: Int = 3

    public init(matchThreshold: Int = 2, maxMatchFailures: Int = 3)
}
```

## Window Transcription Filter Pipeline

The coordinator runs a chain of filters on transcription output before
it reaches the session (background transcriptions) or the final result
assembly (tail transcription). This keeps audio-domain concerns out of
the session and ensures consistent filtering across all decode paths.

```swift
/// Context available to filters. Provides access to the full
/// audio state so filters can derive whatever they need.
public struct FilterContext: Sendable {
    /// Full accumulated audio samples (16kHz mono Float32).
    public let samples: [Float]

    /// Sample rate (16,000).
    public let sampleRate: Int

    /// Where the working window starts.
    public let frontier: Float

    /// Computed: total audio duration in seconds.
    public var audioDuration: Float {
        Float(samples.count) / Float(sampleRate)
    }
}

/// A filter that transforms a transcription result before it
/// reaches the session or final assembly. Each filter is
/// independently testable.
public protocol TranscriptionFilter: Sendable {
    func filter(
        _ transcription: TranscriptionResult,
        context: FilterContext
    ) -> TranscriptionResult
}
```

### Built-in Filters

```swift
/// Removes words with timestamps at or beyond the audio duration.
/// These are hallucinations from WhisperKit's zero-padded silence.
public struct HallucinationTimestampFilter: TranscriptionFilter {
    public func filter(
        _ transcription: TranscriptionResult,
        context: FilterContext
    ) -> TranscriptionResult
}
```

Usage in coordinator:

```swift
func runFilterPipeline(
    _ transcription: TranscriptionResult,
    samples: [Float],
    frontier: Float
) -> TranscriptionResult {
    let context = FilterContext(
        samples: samples,
        sampleRate: WhisperKit.sampleRate,
        frontier: frontier
    )
    var filtered = transcription
    for filter in filters {
        filtered = filter.filter(filtered, context: context)
    }
    return filtered
}
```

### Filter Chain as Logged Configuration

The active filter chain is part of the session's reproducible
configuration. Session records (debug builds) capture which filters
ran and their order. When replaying a session for testing, the same
filter chain is applied.

This means:
- **No defensive clamping in the session.** The session is pure text
  logic and trusts its inputs. Audio-domain invariants (like "no
  timestamps past audio end") are enforced by the filter pipeline.
- **If a filter is missing and a session fails,** the session record
  shows which filters were active, making the misconfiguration visible.
- **Filter changes are testable.** Adding, removing, or reordering
  filters changes the session record output, which integration tests
  catch.

## TranscriptionSession

Pure value type. Tracks the state of one push-to-talk session.
No async, no locks, no audio-domain knowledge. Operates entirely
on word timings and text.

The session receives **pre-filtered** window transcriptions — all
audio-domain cleanup happens in the filter pipeline before reaching
the session.

The session does **not** assemble the final result. It provides its
`transcript` and `frontier` — the coordinator handles tail-word
filtering, segment fallback, and text joining using the same filter
pipeline.

```swift
public struct TranscriptionSession: Sendable {
    // MARK: - Init

    public init(config: TranscriptionSessionConfig = .init())

    // MARK: - State

    /// The current matching strategy.
    public private(set) var strategy: MatchingStrategy

    /// The locked-in transcription text.
    public private(set) var transcript: String

    /// Where confirmed audio ends (seconds). The working window
    /// starts here.
    public private(set) var frontier: Float

    /// Number of consecutive match failures.
    /// Reset to 0 on any successful match.
    public private(set) var consecutiveMatchFailures: Int

    /// Decoder tokens from the last matched boundary words.
    /// The coordinator may pass these to WhisperKit for continuity.
    /// Only meaningful in wordLevel strategy.
    public var overlapTokens: [Int] { get }

    // MARK: - Extending the Transcript

    /// Attempt to extend the transcript with a window transcription.
    ///
    /// The transcription must be pre-filtered (hallucinations removed,
    /// timestamps validated) and must contain at least one word.
    /// The coordinator must not call this with empty transcriptions —
    /// those are not match failures, they're "nothing to work with."
    ///
    /// Compares the window transcription against the previous one using
    /// boundary matching. If the boundaries agree, new words are locked
    /// into the transcript and the frontier advances.
    public mutating func extendTranscript(
        with windowTranscription: TranscriptionResult
    ) -> ExtensionOutcome

    // MARK: - Utilities

    /// Clean transcription text: trim whitespace, remove Whisper
    /// hallucination patterns, collapse double spaces.
    public static func cleanText(_ text: String) -> String
}
```

### Behavioral Contracts

1. **Frontier stability on zero-progress:** When `extendTranscript` returns `.extended(newWords: 0)`, `frontier` must not change from its value before the call.

2. **Strategy transition is one-way:** Once `strategy` becomes `.segmentLevel`, it never returns to `.wordLevel`.

3. **Match failure counter resets on any match:** Any `.extended` outcome (even `newWords: 0`) resets `consecutiveMatchFailures` to 0.

4. **cleanText filters hallucinations:** `[BLANK_AUDIO]`, `(BLANK_AUDIO)`, `[silence]`, `(silence)`, `[no speech]`, `(no speech)`, `[inaudible]`, `(inaudible)`, and any string matching `^\s*[\(\[].+[\)\]]\s*$` must return empty string.

5. **Session trusts its inputs:** The session does not validate timestamps against audio duration or filter hallucinations. Audio-domain invariants are enforced by the filter pipeline. The filter chain is logged as part of the session configuration, making misconfigurations visible in session records.

6. **Empty transcriptions must not reach the session:** The coordinator must guard against calling `extendTranscript` with a `TranscriptionResult` that has zero words (after filtering). Empty results mean "nothing to transcribe" — not a match failure. Feeding empty results would trigger the strategy transition incorrectly.

7. **Word-level matching:** Uses longest-common-prefix on word arrays (case-sensitive, exact string equality on `WordTiming.word`). Requires `matchThreshold` consecutive matching words. Words before the matched boundary pair are locked into the transcript. The matched pair becomes the new boundary for the next comparison.

8. **Segment-level matching:** Confirms all segments except the last (which is still forming). The frontier advances to the end of the last confirmed segment. Operates on segment boundaries and text — does not require word-level timestamps.

## TranscriptionScheduler

Controls when background transcriptions of the working window fire.

```swift
public protocol TranscriptionScheduler: Sendable {
    /// Start scheduling background transcriptions.
    /// Must not call transcribeWorkingWindow() concurrently — serial only.
    func start(engine: SchedulableEngine) async

    /// Stop scheduling and wait for any in-flight transcription to complete.
    ///
    /// **Correctness invariant:** After stop() returns, no further calls
    /// to transcribeWorkingWindow() will occur. The coordinator can then
    /// safely access TranscriptionSession without synchronization.
    func stop() async
}
```

### SchedulableEngine

The interface the scheduler uses to trigger transcriptions.

```swift
public protocol SchedulableEngine: AnyObject, Sendable {
    /// Current number of accumulated audio samples.
    var sampleCount: Int { get }

    /// Transcribe the current working window.
    /// - Parameter maxSampleCount: If provided, clamp the audio
    ///   snapshot to this many samples. Used by test schedulers
    ///   to reproduce exact production conditions.
    func transcribeWorkingWindow(maxSampleCount: Int?) async
}
```

Usage:

```swift
let current = engine.sampleCount
if current - lastSampleCount >= 16_000 {
    await engine.transcribeWorkingWindow(maxSampleCount: nil)
    lastSampleCount = current
}
```

### BackgroundLoopScheduler Behavioral Contracts

1. **Initial delay:** First transcription not attempted until ≥1 second after `start()`.
2. **Threshold:** Only transcribes when ≥16,000 new samples (1s at 16kHz) since last.
3. **Pacing:** 500ms after first transcription, 1000ms after subsequent.
4. **Interruptible:** `stop()` interrupts sleep, returns after current transcription completes.
5. **Serial:** Never calls `transcribeWorkingWindow()` concurrently.

## Coordinator Behavioral Contracts

The WhisperKitEngine coordinator is thin glue. It owns resources,
runs the filter pipeline, and wires components. **No policy logic.**

1. **No branching on strategy:** The coordinator never checks `session.strategy`. It reads `session.frontier` and `session.overlapTokens` and builds WhisperKit options itself.

2. **Runs filter pipeline on all transcriptions:** Both background window transcriptions AND the final tail transcription pass through the same filter chain before use. No transcription bypasses the filters.

3. **Guards empty results:** The coordinator must not call `session.extendTranscript` when the filtered transcription has zero words. Empty results are silently skipped.

4. **Coordinator owns result assembly:** The coordinator stitches the final text from `session.transcript` + filtered tail words. It handles tail-word filtering (timestamps >= frontier), segment-text fallback, and text joining. The session does not assemble results.

5. **Safety net is coordinator-owned:** Rescue transcriptions involve calling WhisperKit again — the session doesn't know about WhisperKit.

6. **Coordinator tracks zero-progress for decoder hints:** The coordinator stores the last `ExtensionOutcome` and derives zero-progress status (`outcome == .extended(newWords: 0)`). When zero-progress, it drops overlap tokens from the WhisperKit decode options to avoid hallucination loops.

7. **Synchronization via structure:** `append()` only touches `samplesLock`. `transcribeWorkingWindow()` reads `samplesLock` then accesses `session` (serial, scheduler guarantee). `finishAndTranscribe()` calls `scheduler.stop()` first, then accesses `session`.

## File Layout

```
Sources/Voice/
├── TranscriptionSession.swift      # Pure session state machine (Task 1)
├── TranscriptionFilter.swift       # Filter protocol + built-in filters (Task 1b)
├── BackgroundLoopScheduler.swift   # Production scheduler (Task 3)
├── WhisperKitEngine.swift          # Coordinator (Task 5 — refactored)
├── AudioFormatConverter.swift      # Unchanged
├── SpeechTranscriber.swift         # Unchanged
├── AppleSpeechEngine.swift         # Unchanged
└── ...

Tests/OperatorTests/
├── TranscriptionSessionTests.swift       # Level 1 algorithmic tests (Task 2)
├── TranscriptionFilterTests.swift        # Filter tests (Task 2)
└── ...

Tests/TranscriptionTests/
├── main.swift                            # Level 2+3 integration tests (Task 4)
└── ...
```

## Mocking Strategy for Parallel Agents

**Task 1 (TranscriptionSession):** No mocks. Pure struct, test with
synthetic `TranscriptionResult` objects via helper:

```swift
func makeWindowTranscription(
    words: [(word: String, start: Float, end: Float)],
    text: String? = nil
) -> TranscriptionResult
```

This helper must create valid `TranscriptionSegment` and `WordTiming`
structures. For segment-level tests, it must support multi-segment
results with configurable segment boundaries.

**Task 1b (Filters):** No mocks. Pure functions, test with synthetic
`TranscriptionResult` and `FilterContext`.

**Task 2 (Level 1 Tests):** Same helpers. Tests compile against the
interface and run once Task 1 lands.

**Task 3 (BackgroundLoopScheduler):** Mock `SchedulableEngine`:

```swift
final class MockSchedulableEngine: SchedulableEngine {
    var sampleCount: Int = 0
    var transcriptionCount: Int = 0
    var timestamps: [ContinuousClock.Instant] = []

    func transcribeWorkingWindow(maxSampleCount: Int?) async {
        transcriptionCount += 1
        timestamps.append(.now)
    }
}
```

**Task 4 (Integration Tests):** Real WhisperKit, no mocks. Structural
updates only.

---

## Full Usage Examples

These show the complete coordinator methods — how all the pieces
fit together in practice.

### Background transcription (called by the scheduler)

```swift
func transcribeWorkingWindow(maxSampleCount: Int?) async {
    // 1. Snapshot the audio
    let samples = samplesLock.withLock { s in
        maxSampleCount.map { Array(s.prefix($0)) } ?? s
    }

    // 2. Read session state
    let frontier = session.frontier

    // 3. Transcribe the working window
    var options = DecodingOptions()
    options.language = "en"
    options.skipSpecialTokens = true
    options.wordTimestamps = true
    options.chunkingStrategy = .none
    if frontier > 0 {
        options.clipTimestamps = [frontier]
    }
    if !lastExtensionWasZeroProgress {
        options.prefixTokens = session.overlapTokens
    }

    guard let windowTranscription = try? await pipe.transcribe(
        audioArray: samples, decodeOptions: options
    ).first else { return }

    // 4. Run filter pipeline
    let filtered = runFilterPipeline(windowTranscription,
                                     samples: samples,
                                     frontier: frontier)

    // 5. Guard: skip empty results
    guard !filtered.allWords.isEmpty else { return }

    // 6. Feed to session
    let outcome = session.extendTranscript(with: filtered)
    lastExtensionWasZeroProgress = (outcome == .extended(newWords: 0))
    log(outcome)
}
```

### Final transcription (user releases key)

```swift
func finishAndTranscribe() async -> String? {
    // 1. Stop the scheduler
    await scheduler.stop()

    // 2. Snapshot the final audio
    let samples = samplesLock.withLock { $0 }
    let audioDuration = Float(samples.count) / Float(WhisperKit.sampleRate)
    let frontier = session.frontier

    // 3. Transcribe the remaining working window
    var options = DecodingOptions()
    options.language = "en"
    options.skipSpecialTokens = true
    options.wordTimestamps = true
    options.chunkingStrategy = .vad
    if frontier > 0 {
        options.clipTimestamps = [frontier]
    }
    if !lastExtensionWasZeroProgress {
        options.prefixTokens = session.overlapTokens
    }

    let tailTranscription = try? await pipe.transcribe(
        audioArray: samples, decodeOptions: options
    ).first

    // 4. Run filter pipeline on tail (same filters as background)
    let filteredTail = tailTranscription.map {
        runFilterPipeline($0, samples: samples, frontier: frontier)
    }

    // 5. Assemble result (coordinator owns this)
    var fullText = assembleResult(
        transcript: session.transcript,
        tailTranscription: filteredTail,
        frontier: frontier
    )

    // 6. Safety net
    let wordCount = Float(fullText.split(separator: " ").count)
    if audioDuration > 3 && wordCount / audioDuration < 0.5 && !fullText.isEmpty {
        if let rescue = try? await pipe.transcribe(audioArray: samples).first {
            let rescueText = TranscriptionSession.cleanText(rescue.text)
            if rescueText.count > fullText.count { fullText = rescueText }
        }
    }
    if fullText.isEmpty && audioDuration > 1 {
        if let rescue = try? await pipe.transcribe(audioArray: samples).first {
            fullText = TranscriptionSession.cleanText(rescue.text)
        }
    }

    guard !fullText.isEmpty else { return nil }

    // 7. Vocabulary correction
    return VocabularyCorrector.correct(text: fullText, vocabulary: vocabulary)
}

/// Coordinator-owned result assembly.
/// Stitches transcript + filtered tail words.
private func assembleResult(
    transcript: String,
    tailTranscription: TranscriptionResult?,
    frontier: Float
) -> String {
    guard let tail = tailTranscription else {
        return TranscriptionSession.cleanText(transcript)
    }

    let allWords = tail.allWords
    let tailText: String

    if !allWords.isEmpty {
        let filtered = allWords.filter { $0.start >= frontier }
        if !filtered.isEmpty {
            tailText = filtered.map(\.word).joined()
        } else {
            // All words before frontier — fall back to segment text
            tailText = tail.text ?? ""
        }
    } else {
        // No word timestamps — use segment text
        tailText = tail.text ?? ""
    }

    return TranscriptionSession.cleanText(transcript + tailText)
}
```

### Background scheduler loop

```swift
func start(engine: SchedulableEngine) async {
    await interruptibleSleep(nanoseconds: 1_000_000_000)

    var lastSampleCount = 0
    var transcriptionCount = 0

    while !stopped {
        let current = engine.sampleCount
        if current - lastSampleCount >= 16_000 {
            await engine.transcribeWorkingWindow(maxSampleCount: nil)
            lastSampleCount = current
            transcriptionCount += 1
        }

        let interval: UInt64 = transcriptionCount <= 1
            ? 500_000_000
            : 1_000_000_000
        await interruptibleSleep(nanoseconds: interval)
    }
}
```

### Algorithmic tests

```swift
func testBoundaryMatchingExtends() {
    var session = TranscriptionSession()

    // First transcription — no previous, so no match possible
    let window1 = makeWindowTranscription(words: [
        ("Hello", 0.0, 0.5),
        ("world", 0.5, 1.0),
    ])
    let outcome1 = session.extendTranscript(with: window1)
    XCTAssertEqual(outcome1, .matchFailed)
    XCTAssertEqual(session.transcript, "")
    XCTAssertEqual(session.frontier, 0.0)

    // Second transcription — same words, boundaries match
    let window2 = makeWindowTranscription(words: [
        ("Hello", 0.0, 0.5),
        ("world", 0.5, 1.0),
        ("how", 1.0, 1.3),
        ("are", 1.3, 1.5),
        ("you", 1.5, 1.8),
    ])
    let outcome2 = session.extendTranscript(with: window2)
    XCTAssertEqual(outcome2, .extended(newWords: 3))
    XCTAssertEqual(session.transcript, " Hello world how")
    XCTAssertEqual(session.frontier, 1.3)
}

func testZeroProgressDoesNotAdvanceFrontier() {
    var session = TranscriptionSession()
    // ... set up with frontier at 5.0, matched words ["the", "end"]

    let same = makeWindowTranscription(words: [
        ("the", 5.0, 5.3),
        ("end", 5.3, 5.6),
    ])
    let outcome = session.extendTranscript(with: same)
    XCTAssertEqual(outcome, .extended(newWords: 0))
    XCTAssertEqual(session.frontier, 5.0)  // unchanged
}

func testStrategyTransitionAfterMatchFailures() {
    var session = TranscriptionSession(config: .init(maxMatchFailures: 3))

    for i in 0..<3 {
        let w = makeWindowTranscription(words: [("word\(i)", 0, 1)])
        session.extendTranscript(with: w)
    }

    XCTAssertEqual(session.strategy, .segmentLevel)
}

func testSegmentLevelConfirmsAllButLast() {
    var session = TranscriptionSession(config: .init(maxMatchFailures: 1))

    // Trigger segment fallback
    let miss = makeWindowTranscription(words: [("x", 0, 1)])
    session.extendTranscript(with: miss)
    XCTAssertEqual(session.strategy, .segmentLevel)

    // Feed a multi-segment transcription
    let segmented = makeWindowTranscription(
        segments: [
            (text: "Hello world.", start: 0.0, end: 1.5),
            (text: "How are you?", start: 1.5, end: 3.0),
            (text: "I'm fine.", start: 3.0, end: 4.0),  // last — not confirmed
        ]
    )
    let outcome = session.extendTranscript(with: segmented)
    XCTAssertEqual(outcome, .extended(newWords: 0))  // segment-level, not word count
    XCTAssertTrue(session.transcript.contains("Hello world."))
    XCTAssertTrue(session.transcript.contains("How are you?"))
    XCTAssertFalse(session.transcript.contains("I'm fine."))
    XCTAssertEqual(session.frontier, 3.0)  // end of last confirmed segment
}

func testHallucinationFilterRemovesWordsPastAudio() {
    let filter = HallucinationTimestampFilter()
    let context = FilterContext(
        samples: [Float](repeating: 0, count: 160_000),
        sampleRate: 16_000,
        frontier: 6.0
    )
    let transcription = makeWindowTranscription(words: [
        ("real", 6.0, 6.5),
        ("hallucinated", 120.0, 121.0),
    ])
    let result = filter.filter(transcription, context: context)
    XCTAssertEqual(result.allWords.count, 1)
    XCTAssertEqual(result.allWords.first?.word, "real")
}

func testCoordinatorSkipsEmptyAfterFiltering() {
    // After filtering, if no words remain, coordinator must NOT
    // call extendTranscript — verify the guard works
    var session = TranscriptionSession()
    let initialFailures = session.consecutiveMatchFailures

    // Simulate: coordinator filters out all words, skips extendTranscript
    let transcription = makeWindowTranscription(words: [
        ("hallucinated", 120.0, 121.0),
    ])
    let filter = HallucinationTimestampFilter()
    let context = FilterContext(
        samples: [Float](repeating: 0, count: 160_000),
        sampleRate: 16_000,
        frontier: 0.0
    )
    let filtered = filter.filter(transcription, context: context)

    // Guard: empty after filtering → do not call extendTranscript
    if !filtered.allWords.isEmpty {
        session.extendTranscript(with: filtered)
    }

    // Match failure counter should NOT have incremented
    XCTAssertEqual(session.consecutiveMatchFailures, initialFailures)
}
```

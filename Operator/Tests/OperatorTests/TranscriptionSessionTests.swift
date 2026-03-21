import Foundation
import Testing
import WhisperKit

@testable import OperatorCore

// MARK: - Test Helpers

/// Creates a TranscriptionResult with word-level timings in a single segment.
private func makeWindowTranscription(
    words: [(word: String, start: Float, end: Float)]
) -> TranscriptionResult {
    let wordTimings = words.map { wt in
        WordTiming(word: wt.word, tokens: [1], start: wt.start, end: wt.end, probability: 1.0)
    }
    let text = wordTimings.map(\.word).joined()
    let segment = TranscriptionSegment(
        start: words.first?.start ?? 0,
        end: words.last?.end ?? 0,
        text: text,
        words: wordTimings
    )
    return TranscriptionResult(
        text: text,
        segments: [segment],
        language: "en",
        timings: TranscriptionTimings()
    )
}

/// Creates a TranscriptionResult with multiple segments (no word-level timings).
private func makeWindowTranscription(
    segments: [(text: String, start: Float, end: Float)]
) -> TranscriptionResult {
    let segs = segments.enumerated().map { idx, seg in
        TranscriptionSegment(
            id: idx,
            start: seg.start,
            end: seg.end,
            text: seg.text
        )
    }
    let fullText = segs.map(\.text).joined(separator: " ")
    return TranscriptionResult(
        text: fullText,
        segments: segs,
        language: "en",
        timings: TranscriptionTimings()
    )
}

/// Creates a TranscriptionResult with word-level timings and custom tokens.
private func makeWindowTranscription(
    words: [(word: String, tokens: [Int], start: Float, end: Float)]
) -> TranscriptionResult {
    let wordTimings = words.map { wt in
        WordTiming(word: wt.word, tokens: wt.tokens, start: wt.start, end: wt.end, probability: 1.0)
    }
    let text = wordTimings.map(\.word).joined()
    let segment = TranscriptionSegment(
        start: words.first?.start ?? 0,
        end: words.last?.end ?? 0,
        text: text,
        words: wordTimings
    )
    return TranscriptionResult(
        text: text,
        segments: [segment],
        language: "en",
        timings: TranscriptionTimings()
    )
}

// MARK: - Tests

@Suite("TranscriptionSession")
internal struct TranscriptionSessionTests {
    // 1. Boundary matching extends transcript
    @Test("first call matchFailed, second call with matching words extends")
    func testBoundaryMatchingExtends() {
        var session = TranscriptionSession()

        // First transcription — no previous, so no match possible
        let window1 = makeWindowTranscription(words: [
            ("Hello", 0.0, 0.5),
            ("world", 0.5, 1.0)
        ])
        let outcome1 = session.extendTranscript(with: window1)
        #expect(outcome1 == .matchFailed)
        #expect(session.transcript.isEmpty)
        #expect(session.frontier == 0.0)

        // Second transcription — same words at boundary, new words after
        let window2 = makeWindowTranscription(words: [
            ("Hello", 0.0, 0.5),
            ("world", 0.5, 1.0),
            ("how", 1.0, 1.3),
            ("are", 1.3, 1.5),
            ("you", 1.5, 1.8)
        ])
        let outcome2 = session.extendTranscript(with: window2)
        // Common prefix is [Hello, world] (2 words = matchThreshold).
        // Both become the boundary pair, nothing to confirm yet.
        #expect(outcome2 == .extended(newWords: 0))

        // Third transcription — same boundary words with more after
        let window3 = makeWindowTranscription(words: [
            ("Hello", 0.0, 0.5),
            ("world", 0.5, 1.0),
            ("how", 1.0, 1.3),
            ("are", 1.3, 1.5),
            ("you", 1.5, 1.8),
            ("doing", 1.8, 2.0)
        ])
        let outcome3 = session.extendTranscript(with: window3)
        // Common prefix with prevWords should confirm words before the new boundary pair
        #expect(outcome3 == .extended(newWords: 3))
        #expect(session.transcript.contains("Hello"))
        #expect(session.transcript.contains("world"))
        #expect(session.transcript.contains("how"))
    }

    // 2. Zero progress does not advance frontier
    @Test("same boundary words keep frontier unchanged")
    func testZeroProgressDoesNotAdvanceFrontier() {
        var session = TranscriptionSession()

        // Build up: first call to set prevWords
        let window1 = makeWindowTranscription(words: [
            (" the", 5.0, 5.3),
            (" end", 5.3, 5.6)
        ])
        _ = session.extendTranscript(with: window1)

        // Second call — same words match, but nothing new beyond match point
        let window2 = makeWindowTranscription(words: [
            (" the", 5.0, 5.3),
            (" end", 5.3, 5.6)
        ])
        let outcome = session.extendTranscript(with: window2)
        #expect(outcome == .extended(newWords: 0))
        #expect(session.frontier == 0.0)  // unchanged — no words before match point
    }

    // 3. Strategy transition after max match failures
    @Test("transitions to segmentLevel after maxMatchFailures consecutive failures")
    func testStrategyTransitionAfterMatchFailures() {
        var session = TranscriptionSession(config: .init(maxMatchFailures: 3))

        // Feed 3 completely different single-word transcriptions
        // First establishes prevWords, second/third/fourth are mismatches
        // But first call is also a miss (no prevWords), so we need 3 total misses
        for i in 0..<3 {
            let window = makeWindowTranscription(words: [("word\(i)", 0, 1)])
            _ = session.extendTranscript(with: window)
        }

        #expect(session.strategy == .segmentLevel)
    }

    // 4. Segment-level confirms all but last
    @Test("segment fallback confirms all segments except the last")
    func testSegmentLevelConfirmsAllButLast() {
        var session = TranscriptionSession(config: .init(maxMatchFailures: 1))

        // Trigger segment fallback with one miss
        let miss = makeWindowTranscription(words: [("x", 0, 1)])
        _ = session.extendTranscript(with: miss)
        #expect(session.strategy == .segmentLevel)

        // Feed a multi-segment transcription
        let segmented = makeWindowTranscription(
            segments: [
                (text: "Hello world.", start: 0.0, end: 1.5),
                (text: "How are you?", start: 1.5, end: 3.0),
                (text: "I'm fine.", start: 3.0, end: 4.0)
            ]
        )
        let outcome = session.extendTranscript(with: segmented)
        #expect(outcome == .extended(newWords: 0))
        #expect(session.transcript.contains("Hello world."))
        #expect(session.transcript.contains("How are you?"))
        #expect(!session.transcript.contains("I'm fine."))
        #expect(session.frontier == 3.0)
    }

    // 5. Clean text filters hallucinations
    @Test("cleanText removes all hallucination patterns")
    func testCleanTextFiltersHallucinations() {
        #expect(TranscriptionSession.cleanText("[BLANK_AUDIO]").isEmpty)
        #expect(TranscriptionSession.cleanText("(BLANK_AUDIO)").isEmpty)
        #expect(TranscriptionSession.cleanText("[silence]").isEmpty)
        #expect(TranscriptionSession.cleanText("(silence)").isEmpty)
        #expect(TranscriptionSession.cleanText("[no speech]").isEmpty)
        #expect(TranscriptionSession.cleanText("(no speech)").isEmpty)
        #expect(TranscriptionSession.cleanText("[inaudible]").isEmpty)
        #expect(TranscriptionSession.cleanText("(inaudible)").isEmpty)
        #expect(TranscriptionSession.cleanText("(air whooshes)").isEmpty)
        #expect(TranscriptionSession.cleanText("[clicking]").isEmpty)
        #expect(TranscriptionSession.cleanText("  [BLANK_AUDIO]  ").isEmpty)
        #expect(TranscriptionSession.cleanText("Hello  world") == "Hello world")
        #expect(TranscriptionSession.cleanText("Hello world") == "Hello world")
    }

    // 6. Coordinator skips empty after filtering
    @Test("empty transcription after filtering does not affect session state")
    func testCoordinatorSkipsEmptyAfterFiltering() {
        var session = TranscriptionSession()
        let initialFailures = session.consecutiveMatchFailures

        let transcription = makeWindowTranscription(words: [
            ("hallucinated", 120.0, 121.0)
        ])
        let filter = HallucinationTimestampFilter()
        let context = FilterContext(
            samples: [Float](repeating: 0, count: 160_000),
            sampleRate: 16_000,
            frontier: 0.0
        )
        let filtered = filter.filter(transcription, context: context)

        // Guard: empty after filtering — do not call extendTranscript
        if !filtered.allWords.isEmpty {
            session.extendTranscript(with: filtered)
        }

        #expect(session.consecutiveMatchFailures == initialFailures)
    }

    // 7. Frontier never exceeds matched word timestamps
    @Test("frontier uses agreed words' timestamps correctly")
    func testFrontierNeverExceedsMatchedWordTimestamps() {
        var session = TranscriptionSession()

        let window1 = makeWindowTranscription(words: [
            (" Hello", 0.0, 0.5),
            (" world", 0.5, 1.0),
            (" how", 1.0, 1.3)
        ])
        _ = session.extendTranscript(with: window1)

        let window2 = makeWindowTranscription(words: [
            (" Hello", 0.0, 0.5),
            (" world", 0.5, 1.0),
            (" how", 1.0, 1.3),
            (" are", 1.3, 1.5),
            (" you", 1.5, 1.8)
        ])
        let outcome = session.extendTranscript(with: window2)
        // Common prefix [Hello, world, how] = 3 words, threshold=2.
        // toConfirm = prefix(1) = [Hello], newAgreed = suffix(2) = [world, how]
        #expect(outcome == .extended(newWords: 1))

        // Frontier should be at the start of the first agreed word
        #expect(session.frontier == 0.5)  // start of "world"
    }

    // 8. Overlap tokens returns tokens from boundary words
    @Test("overlapTokens returns tokens from last agreed boundary words")
    func testOverlapTokensReturnsTokensFromBoundaryWords() {
        var session = TranscriptionSession()

        // Need 3 passes to get real progress (words confirmed)
        let window1 = makeWindowTranscription(words: [
            (word: " Hello", tokens: [10, 20], start: 0.0, end: 0.5),
            (word: " world", tokens: [30, 40], start: 0.5, end: 1.0),
            (word: " how", tokens: [50], start: 1.0, end: 1.3)
        ])
        _ = session.extendTranscript(with: window1)

        let window2 = makeWindowTranscription(words: [
            (word: " Hello", tokens: [10, 20], start: 0.0, end: 0.5),
            (word: " world", tokens: [30, 40], start: 0.5, end: 1.0),
            (word: " how", tokens: [50], start: 1.0, end: 1.3),
            (word: " are", tokens: [60], start: 1.3, end: 1.5)
        ])
        let outcome = session.extendTranscript(with: window2)
        // Common prefix [Hello, world, how] = 3, threshold=2
        // toConfirm = [Hello], newAgreed = [world, how]
        #expect(outcome == .extended(newWords: 1))

        // The agreed boundary words are [world, how]
        #expect(session.overlapTokens == [30, 40, 50])
    }

    // 9. Consecutive match failures reset on real progress only
    @Test("match failure counter resets to 0 only on extended with new words")
    func testConsecutiveMatchFailuresResetOnRealProgress() {
        var session = TranscriptionSession(config: .init(maxMatchFailures: 5))

        // Create 2 misses
        let miss1 = makeWindowTranscription(words: [("a", 0, 1)])
        _ = session.extendTranscript(with: miss1)
        let miss2 = makeWindowTranscription(words: [("b", 0, 1)])
        _ = session.extendTranscript(with: miss2)
        #expect(session.consecutiveMatchFailures == 2)

        // Feed matching words — miss because prevWords=[b] doesn't match [cat,dog]
        let window1 = makeWindowTranscription(words: [
            (" cat", 0.0, 0.5),
            (" dog", 0.5, 1.0)
        ])
        _ = session.extendTranscript(with: window1)
        #expect(session.consecutiveMatchFailures == 3)

        // Same words again — zero-progress match, should NOT reset failures
        let window2 = makeWindowTranscription(words: [
            (" cat", 0.0, 0.5),
            (" dog", 0.5, 1.0)
        ])
        let outcome2 = session.extendTranscript(with: window2)
        #expect(outcome2 == .extended(newWords: 0))
        #expect(session.consecutiveMatchFailures == 3)  // NOT reset

        // Feed with more words so common prefix > threshold → real progress
        let window3 = makeWindowTranscription(words: [
            (" cat", 0.0, 0.5),
            (" dog", 0.5, 1.0),
            (" fish", 1.0, 1.5),
            (" bird", 1.5, 2.0)
        ])
        _ = session.extendTranscript(with: window3)
        // prevWords now includes fish, bird

        let window4 = makeWindowTranscription(words: [
            (" cat", 0.0, 0.5),
            (" dog", 0.5, 1.0),
            (" fish", 1.0, 1.5),
            (" bird", 1.5, 2.0),
            (" tree", 2.0, 2.5)
        ])
        let outcome4 = session.extendTranscript(with: window4)
        // Common prefix [cat, dog, fish, bird] = 4 > threshold(2)
        // toConfirm = [cat, dog] = 2 words → real progress
        #expect(outcome4 == .extended(newWords: 2))
        #expect(session.consecutiveMatchFailures == 0)  // NOW reset
    }

    // 10. Segment level does not return to word level
    @Test("once segmentLevel, strategy never returns to wordLevel")
    func testSegmentLevelDoesNotReturnToWordLevel() {
        var session = TranscriptionSession(config: .init(maxMatchFailures: 1))

        // Trigger transition
        let miss = makeWindowTranscription(words: [("x", 0, 1)])
        _ = session.extendTranscript(with: miss)
        #expect(session.strategy == .segmentLevel)

        // Feed multiple successful segment-level results
        let seg1 = makeWindowTranscription(
            segments: [
                (text: "First.", start: 0.0, end: 1.0),
                (text: "Second.", start: 1.0, end: 2.0),
                (text: "Third.", start: 2.0, end: 3.0)
            ]
        )
        _ = session.extendTranscript(with: seg1)
        #expect(session.strategy == .segmentLevel)

        let seg2 = makeWindowTranscription(
            segments: [
                (text: "Fourth.", start: 3.0, end: 4.0),
                (text: "Fifth.", start: 4.0, end: 5.0)
            ]
        )
        _ = session.extendTranscript(with: seg2)
        #expect(session.strategy == .segmentLevel)
    }

    // MARK: - Parallel Decode Assembly

    @Test("assembleWithFrontier keeps all words when frontier unchanged")
    func testAssembleWithFrontierUnchanged() {
        let words = [
            WordTiming(word: " Hello", tokens: [], start: 4.0, end: 4.5, probability: 1),
            WordTiming(word: " world", tokens: [], start: 4.5, end: 5.0, probability: 1)
        ]
        let result = TranscriptionSession.assembleWithFrontier(
            transcript: "I said",
            tailWords: words,
            tailSegmentText: "",
            frontier: 4.0
        )
        #expect(result == "I said Hello world")
    }

    @Test("assembleWithFrontier filters words before advanced frontier")
    func testAssembleWithFrontierAdvanced() {
        // Pessimistic decode started at frontier=4.0
        // Background pass advanced frontier to 6.0
        // Words at 4.0-5.0 should be filtered, words at 6.0+ kept
        let words = [
            WordTiming(word: " old", tokens: [], start: 4.0, end: 4.5, probability: 1),
            WordTiming(word: " stuff", tokens: [], start: 4.5, end: 5.0, probability: 1),
            WordTiming(word: " new", tokens: [], start: 6.0, end: 6.5, probability: 1),
            WordTiming(word: " words", tokens: [], start: 6.5, end: 7.0, probability: 1)
        ]
        let result = TranscriptionSession.assembleWithFrontier(
            transcript: "Confirmed text",
            tailWords: words,
            tailSegmentText: "",
            frontier: 6.0
        )
        #expect(result == "Confirmed text new words")
    }

    @Test("assembleWithFrontier falls back to segment text when all words filtered")
    func testAssembleWithFrontierAllFiltered() {
        let words = [
            WordTiming(word: " old", tokens: [], start: 2.0, end: 2.5, probability: 1),
            WordTiming(word: " stuff", tokens: [], start: 2.5, end: 3.0, probability: 1)
        ]
        let result = TranscriptionSession.assembleWithFrontier(
            transcript: "Confirmed",
            tailWords: words,
            tailSegmentText: " fallback text",
            frontier: 5.0
        )
        #expect(result == "Confirmed fallback text")
    }

    @Test("assembleWithFrontier uses segment text when no word timestamps")
    func testAssembleWithFrontierNoWords() {
        let result = TranscriptionSession.assembleWithFrontier(
            transcript: "Confirmed",
            tailWords: [],
            tailSegmentText: " the rest",
            frontier: 3.0
        )
        #expect(result == "Confirmed the rest")
    }
}

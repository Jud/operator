import Foundation
import Testing
import WhisperKit

@testable import OperatorCore

// MARK: - Test Helpers

private func makeFilterTranscription(
    words: [(word: String, start: Float, end: Float)]
) -> TranscriptionResult {
    let wordTimings = words.map { w in
        WordTiming(word: w.word, tokens: [1], start: w.start, end: w.end, probability: 1.0)
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

@Suite("TranscriptionFilter")
struct TranscriptionFilterTests {
    // 1. Hallucination filter removes words past audio duration
    @Test("removes words with timestamps at or beyond audio duration")
    func testHallucinationFilterRemovesWordsPastAudio() {
        let filter = HallucinationTimestampFilter()
        let context = FilterContext(
            samples: [Float](repeating: 0, count: 160_000),
            sampleRate: 16_000,
            frontier: 6.0
        )
        let transcription = makeFilterTranscription(words: [
            ("real", 6.0, 6.5),
            ("hallucinated", 120.0, 121.0)
        ])
        let result = filter.filter(transcription, context: context)
        #expect(result.allWords.count == 1)
        #expect(result.allWords.first?.word == "real")
    }

    // 2. Hallucination filter keeps valid words
    @Test("words within audio duration are preserved")
    func testHallucinationFilterKeepsValidWords() {
        let filter = HallucinationTimestampFilter()
        let context = FilterContext(
            samples: [Float](repeating: 0, count: 160_000),
            sampleRate: 16_000,
            frontier: 0.0
        )
        let transcription = makeFilterTranscription(words: [
            (" Hello", 0.0, 0.5),
            (" world", 0.5, 1.0),
            (" how", 1.0, 1.5)
        ])
        let result = filter.filter(transcription, context: context)
        #expect(result.allWords.count == 3)
        #expect(result.allWords[0].word == " Hello")
        #expect(result.allWords[1].word == " world")
        #expect(result.allWords[2].word == " how")
    }

    // 3. Hallucination filter handles empty result after filtering
    @Test("all words filtered leaves empty result")
    func testHallucinationFilterEmptyResult() {
        let filter = HallucinationTimestampFilter()
        let context = FilterContext(
            samples: [Float](repeating: 0, count: 160_000),
            sampleRate: 16_000,
            frontier: 0.0
        )
        // Audio is 10 seconds, both words are way past
        let transcription = makeFilterTranscription(words: [
            ("fake1", 120.0, 121.0),
            ("fake2", 130.0, 131.0)
        ])
        let result = filter.filter(transcription, context: context)
        #expect(result.allWords.isEmpty)
        #expect(result.segments.isEmpty)
    }
}

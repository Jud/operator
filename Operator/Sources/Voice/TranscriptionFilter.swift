import WhisperKit

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

    public init(samples: [Float], sampleRate: Int, frontier: Float) {
        self.samples = samples
        self.sampleRate = sampleRate
        self.frontier = frontier
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

/// Removes words with timestamps at or beyond the audio duration.
/// These are hallucinations from WhisperKit's zero-padded silence.
public struct HallucinationTimestampFilter: TranscriptionFilter, Sendable {
    public init() {}

    public func filter(
        _ transcription: TranscriptionResult,
        context: FilterContext
    ) -> TranscriptionResult {
        let duration = context.audioDuration
        let filteredSegments = transcription.segments.compactMap { segment -> TranscriptionSegment? in
            guard let words = segment.words else {
                // No word-level timestamps; keep segment if its start is within duration
                return segment.start < duration ? segment : nil
            }

            let validWords = words.filter { $0.start < duration }
            if validWords.isEmpty {
                return nil
            }

            var updated = segment
            updated.words = validWords
            updated.text = validWords.map(\.word).joined()
            return updated
        }

        let fullText = filteredSegments.map(\.text).joined(separator: " ")
        return TranscriptionResult(
            text: fullText,
            segments: filteredSegments,
            language: transcription.language,
            timings: transcription.timings
        )
    }
}

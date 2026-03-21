import WhisperKit

// MARK: - Supporting Types

/// Outcome of extending the transcript with a new window transcription.
public enum ExtensionOutcome: Equatable, Sendable {
    /// Transcript was extended. newWords is the number of words
    /// locked into the transcript. May be 0 if boundaries matched
    /// but there was nothing new beyond the match point.
    case extended(newWords: Int)

    /// The window transcription didn't match the previous one
    /// at the boundary. The frontier doesn't move.
    case matchFailed
}

/// Strategy used to match window transcriptions against previous results.
public enum MatchingStrategy: Equatable, Sendable {
    /// Word-level boundary matching. Default.
    /// Uses longest-common-prefix on word arrays. Requires
    /// matchThreshold consecutive matching words for agreement.
    case wordLevel

    /// Segment-level boundary matching. Entered after maxMatchFailures
    /// consecutive failures. One-way — does not return to wordLevel.
    case segmentLevel
}

/// Configuration for transcription session matching behavior.
public struct TranscriptionSessionConfig: Equatable, Sendable {
    /// Number of consecutive matching words required at the boundary.
    public var matchThreshold: Int

    /// Consecutive match failures before switching to segment-level matching.
    public var maxMatchFailures: Int

    /// Creates a session config with the given thresholds.
    public init(matchThreshold: Int = 2, maxMatchFailures: Int = 3) {
        self.matchThreshold = matchThreshold
        self.maxMatchFailures = maxMatchFailures
    }
}

// MARK: - TranscriptionSession

/// Pure value type tracking the state of one push-to-talk session.
///
/// No async, no locks, no audio-domain knowledge. Operates entirely
/// on word timings and text.
public struct TranscriptionSession: Sendable {
    private let config: TranscriptionSessionConfig

    /// The current matching strategy.
    public private(set) var strategy: MatchingStrategy

    /// The locked-in transcription text.
    public private(set) var transcript: String

    /// Where confirmed audio ends (seconds).
    ///
    /// The working window starts here.
    public private(set) var frontier: Float

    /// Number of consecutive match failures.
    ///
    /// Reset to 0 on any successful match.
    public private(set) var consecutiveMatchFailures: Int

    /// Words from the previous window transcription, used for boundary comparison.
    private var prevWords: [WordTiming]

    /// The last agreed boundary words (matchThreshold words that matched).
    private var lastAgreedWords: [WordTiming]

    /// Decoder tokens from the last matched boundary words.
    ///
    /// The coordinator may pass these to WhisperKit for continuity.
    /// Only meaningful in wordLevel strategy.
    public var overlapTokens: [Int] {
        lastAgreedWords.flatMap(\.tokens)
    }

    // MARK: - Segment-level state

    /// Accumulated text from segment-level confirmation.
    private var segmentConfirmedText: String

    /// End time of last confirmed segment.
    private var segmentConfirmedEndSeconds: Float

    // MARK: - Init

    /// Creates a new transcription session with the given configuration.
    public init(config: TranscriptionSessionConfig = .init()) {
        self.config = config
        self.strategy = .wordLevel
        self.transcript = ""
        self.frontier = 0
        self.consecutiveMatchFailures = 0
        self.prevWords = []
        self.lastAgreedWords = []
        self.segmentConfirmedText = ""
        self.segmentConfirmedEndSeconds = 0
    }

    // MARK: - Extending the Transcript

    /// Attempt to extend the transcript with a window transcription.
    ///
    /// The transcription must be pre-filtered (hallucinations removed,
    /// timestamps validated) and must contain at least one word.
    /// The coordinator must not call this with empty transcriptions.
    @discardableResult
    public mutating func extendTranscript(
        with windowTranscription: TranscriptionResult
    ) -> ExtensionOutcome {
        switch strategy {
        case .wordLevel:
            return extendWordLevel(windowTranscription)

        case .segmentLevel:
            return extendSegmentLevel(windowTranscription)
        }
    }

    // MARK: - Word-Level Matching

    private mutating func extendWordLevel(
        _ windowTranscription: TranscriptionResult
    ) -> ExtensionOutcome {
        let hypothesisWords = windowTranscription.allWords.filter { $0.start >= frontier }

        let commonPrefix = TranscriptionUtilities.findLongestCommonPrefix(
            prevWords,
            hypothesisWords
        )

        guard commonPrefix.count >= config.matchThreshold else {
            consecutiveMatchFailures += 1

            // Update prevWords for next comparison
            let clip = lastAgreedWords.first?.start ?? frontier
            prevWords = hypothesisWords.filter { $0.start >= clip }

            if consecutiveMatchFailures >= config.maxMatchFailures {
                // Transition to segment-level matching (one-way)
                strategy = .segmentLevel
                let wordText = transcript + lastAgreedWords.map(\.word).joined()
                segmentConfirmedText = wordText.trimmingCharacters(in: .whitespaces)
                segmentConfirmedEndSeconds = frontier
            }

            return .matchFailed
        }
        // Boundary matched. The last matchThreshold words of the common
        // prefix become the new boundary pair. Words before them in the
        // common prefix are confirmed and locked into the transcript.
        let newAgreed = Array(commonPrefix.suffix(config.matchThreshold))
        let toConfirm = commonPrefix.prefix(commonPrefix.count - config.matchThreshold)

        let newText = toConfirm.map(\.word).joined()
        transcript += newText

        if !toConfirm.isEmpty {
            // Real progress: update frontier, agreed words, reset failures
            lastAgreedWords = newAgreed
            if let first = newAgreed.first {
                frontier = first.start
            }
            consecutiveMatchFailures = 0
        }
        // On zero-progress: lastAgreedWords, frontier, and failures
        // are all left unchanged. The boundary matched but nothing new
        // came through — don't drift state.

        // Update prevWords for next comparison
        let clip = lastAgreedWords.first?.start ?? frontier
        prevWords = hypothesisWords.filter { $0.start >= clip }

        return .extended(newWords: toConfirm.count)
    }

    // MARK: - Segment-Level Matching

    private mutating func extendSegmentLevel(
        _ windowTranscription: TranscriptionResult
    ) -> ExtensionOutcome {
        let segments = windowTranscription.segments

        guard segments.count >= 2 else {
            // Need at least 2 segments to confirm all-but-last
            consecutiveMatchFailures = 0
            return .extended(newWords: 0)
        }

        let toConfirm = Array(segments.prefix(segments.count - 1))
        let newEnd = toConfirm.last?.end ?? segmentConfirmedEndSeconds

        guard newEnd > segmentConfirmedEndSeconds else {
            consecutiveMatchFailures = 0
            return .extended(newWords: 0)
        }

        let newText = toConfirm.map(\.text)
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespaces)

        if segmentConfirmedText.isEmpty {
            segmentConfirmedText = newText
        } else {
            segmentConfirmedText += " " + newText
        }
        segmentConfirmedEndSeconds = newEnd
        transcript = segmentConfirmedText
        frontier = newEnd

        consecutiveMatchFailures = 0

        return .extended(newWords: 0)
    }

    // MARK: - Result Assembly

    /// Assemble final text from the confirmed transcript and a tail decode,
    /// filtering tail words to only those at or after the given frontier.
    ///
    /// This supports the parallel decode strategy: a pessimistic tail decode
    /// starts from an earlier frontier, and if the background pass advances
    /// the frontier, the tail words before the new frontier are discarded.
    ///
    /// Pure function — same inputs always produce the same output.
    ///
    /// - Parameters:
    ///   - transcript: The confirmed text from the session.
    ///   - tailWords: Word timings from the tail decode result.
    ///   - tailSegmentText: Fallback segment-level text if word filtering
    ///     eliminates everything.
    ///   - frontier: The frontier to filter against (may be more advanced
    ///     than the frontier used to start the decode).
    /// - Returns: The assembled text, cleaned of hallucinations.
    public static func assembleWithFrontier(
        transcript: String,
        tailWords: [WordTiming],
        tailSegmentText: String,
        frontier: Float
    ) -> String {
        let tailText: String

        if !tailWords.isEmpty {
            let filtered = tailWords.filter { $0.start >= frontier }
            if !filtered.isEmpty {
                tailText = filtered.map(\.word).joined()
            } else {
                // All words before frontier — fall back to segment text
                tailText = tailSegmentText
            }
        } else {
            // No word timestamps — use segment text
            tailText = tailSegmentText
        }

        return cleanText(transcript + tailText)
    }

    // MARK: - Utilities

    /// Known Whisper hallucination strings produced for silence or very short audio.
    private static let blankPatterns: Set<String> = [
        "[BLANK_AUDIO]", "(BLANK_AUDIO)", "[silence]", "(silence)",
        "[no speech]", "(no speech)", "[inaudible]", "(inaudible)"
    ]

    /// Regex matching text that is entirely parenthesized or bracketed non-speech annotations.
    nonisolated(unsafe) private static let nonSpeechPattern = /^\s*[\(\[].+[\)\]]\s*$/

    /// Clean transcription text: trim whitespace, remove Whisper
    /// hallucination patterns, collapse double spaces.
    public static func cleanText(_ text: String) -> String {
        var result = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if blankPatterns.contains(result) {
            return ""
        }
        if result.wholeMatch(of: nonSpeechPattern) != nil {
            return ""
        }
        while result.contains("  ") {
            result = result.replacingOccurrences(of: "  ", with: " ")
        }
        return result
    }
}

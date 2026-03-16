import AVFoundation
import VocabularyCorrector
import WhisperKit
import os

/// WhisperKit-based implementation of ``TranscriptionEngine``.
///
/// Uses a CoreML Whisper model for high-accuracy on-device speech recognition.
/// Supports vocabulary biasing via prompt tokens and post-transcription vocabulary
/// correction.
///
/// During recording, a background loop re-transcribes accumulated audio every
/// ~1 second, confirming segments whose boundaries are stable across passes.
/// Confirmed segments are locked and skipped in future passes via `clipTimestamps`.
/// On `finishAndTranscribe()`, only the unconfirmed tail needs decoding, giving
/// near-instant results for typical push-to-talk durations.
public final class WhisperKitEngine: TranscriptionEngine, @unchecked Sendable {
    private struct SessionState: @unchecked Sendable {
        var audioSamples: [Float] = []
        var converter: AudioFormatConverter?
        var promptTokens: [Int] = []
        var vocabulary: [String] = []
    }

    // MARK: - Type Properties

    private static let logger = Log.logger(for: "WhisperKitEngine")
    private static let maxPromptTokens = 224

    /// Minimum new audio (in samples at 16 kHz) before triggering a background pass.
    private static let retranscribeThreshold = 16_000  // 1 second

    /// Number of trailing segments to keep unconfirmed (they may shift as more audio arrives).
    private static let segmentsToKeepUnconfirmed = 2

    // MARK: - Instance Properties

    private let pipe: WhisperKit
    private let sessionLock = OSAllocatedUnfairLock<SessionState?>(initialState: nil)

    /// Confirmed segments and the clip point for the next pass.
    private let streamLock = OSAllocatedUnfairLock<StreamState>(
        initialState: StreamState()
    )

    /// Handle to the periodic re-transcription task.
    private var retranscribeTask: Task<Void, Never>?

    // MARK: - Initialization

    /// Create a WhisperKitEngine with a pre-loaded WhisperKit pipeline.
    ///
    /// The pipeline should be initialized before constructing this engine.
    /// Use ``WhisperKitEngine/create(modelFolder:)`` for convenience.
    public init(pipe: WhisperKit) {
        self.pipe = pipe
    }

    /// Create a WhisperKitEngine by loading a model from disk.
    ///
    /// - Parameter modelFolder: Path to the directory containing the WhisperKit model.
    /// - Returns: A configured engine ready for transcription.
    /// - Throws: If the WhisperKit pipeline fails to load.
    public static func create(modelFolder: String) async throws -> WhisperKitEngine {
        let config = WhisperKitConfig(modelFolder: modelFolder)
        let pipe = try await WhisperKit(config)
        logger.info("WhisperKit pipeline loaded from \(modelFolder)")
        return WhisperKitEngine(pipe: pipe)
    }

    // MARK: - TranscriptionEngine

    /// Prepare a new transcription session with optional vocabulary biasing.
    public func prepare(contextualStrings: [String]) throws {
        cancel()

        let resolvedTokens: [Int]
        if !contextualStrings.isEmpty, let tokenizer = pipe.tokenizer {
            let joined = contextualStrings.joined(separator: " ")
            let tokens = tokenizer.encode(text: joined)
            resolvedTokens = Array(tokens.prefix(Self.maxPromptTokens))
            Self.logger.info(
                "Tokenized \(contextualStrings.count) contextual strings → \(resolvedTokens.count) prompt tokens"
            )
        } else {
            resolvedTokens = []
        }

        let vocab = contextualStrings
        sessionLock.withLock { state in
            var session = SessionState(
                promptTokens: resolvedTokens,
                vocabulary: vocab
            )
            session.audioSamples.reserveCapacity(480_000)
            state = session
        }
        streamLock.withLock { $0 = StreamState() }

        startRetranscriptionLoop()
        Self.logger.debug("WhisperKit session prepared")
    }

    /// Append an audio buffer from the microphone tap.
    public func append(_ buffer: AVAudioPCMBuffer) {
        nonisolated(unsafe) let buf = buffer
        let converter: AudioFormatConverter? = sessionLock.withLock { state in
            guard state != nil
            else { return nil }
            if let existing = state?.converter {
                return existing
            }
            guard let conv = try? AudioFormatConverter(inputFormat: buf.format)
            else { return nil }
            state?.converter = conv
            return conv
        }
        guard let converter
        else { return }
        sessionLock.withLock { state in
            guard var session = state
            else { return }
            _ = converter.appendConverted(buf, to: &session.audioSamples)
            state = session
        }
    }

    /// Finish recording and return the transcribed text.
    ///
    /// Decodes only the unconfirmed tail (from the last confirmed segment boundary)
    /// and combines it with the confirmed prefix for the full result.
    public func finishAndTranscribe() async -> String? {  // swiftlint:disable:this function_body_length
        retranscribeTask?.cancel()
        retranscribeTask = nil

        let snapshot = sessionLock.withLock { state -> SessionState? in
            let session = state
            state = nil
            return session
        }
        let samples = snapshot?.audioSamples ?? []
        let promptTokens = snapshot?.promptTokens ?? []
        let vocabulary = snapshot?.vocabulary ?? []

        guard !samples.isEmpty
        else {
            Self.logger.info("No audio samples captured")
            return nil
        }

        let stream = streamLock.withLock { $0 }

        let tailText = await transcribe(
            samples: samples,
            promptTokens: promptTokens,
            clipStart: stream.confirmedEndSeconds
        )

        let fullText: String
        if stream.confirmedText.isEmpty {
            fullText = tailText ?? ""
        } else if let tail = tailText, !tail.isEmpty {
            fullText = stream.confirmedText + " " + tail
        } else {
            fullText = stream.confirmedText
        }

        guard !fullText.isEmpty
        else {
            Self.logger.info("Transcription returned empty")
            return nil
        }

        let corrected = VocabularyCorrector.correct(text: fullText, vocabulary: vocabulary)
        if corrected != fullText {
            Self.logger.notice(
                "Post-correction: \"\(fullText, privacy: .public)\" -> \"\(corrected, privacy: .public)\""
            )
        }
        let clip = stream.confirmedEndSeconds
        Self.logger.notice(
            "Transcription (\(samples.count) samples, clip=\(clip)s): \(corrected, privacy: .public)"
        )
        return corrected
    }

    /// Cancel any in-progress session and release resources.
    public func cancel() {
        retranscribeTask?.cancel()
        retranscribeTask = nil
        resetSession()
    }

    // MARK: - Private

    private func resetSession() {
        sessionLock.withLock { $0 = nil }
        streamLock.withLock { $0 = StreamState() }
    }

    private func startRetranscriptionLoop() {
        retranscribeTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 2_000_000_000)

            while !Task.isCancelled {
                guard let self
                else { return }

                let (samples, promptTokens) = self.sessionLock.withLock { state -> ([Float], [Int]) in
                    guard let session = state
                    else { return ([], []) }
                    return (session.audioSamples, session.promptTokens)
                }

                let stream = self.streamLock.withLock { $0 }
                let processedSamples = Int(
                    stream.confirmedEndSeconds * Float(WhisperKit.sampleRate)
                )
                let newSamples = samples.count - processedSamples

                if newSamples >= Self.retranscribeThreshold {
                    await self.runBackgroundPass(
                        samples: samples,
                        promptTokens: promptTokens,
                        clipStart: stream.confirmedEndSeconds
                    )
                }

                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }
        }
    }

    private func runBackgroundPass(
        samples: [Float],
        promptTokens: [Int],
        clipStart: Float
    ) async {
        let segments = await transcribeSegments(
            samples: samples,
            promptTokens: promptTokens,
            clipStart: clipStart
        )

        guard !segments.isEmpty
        else { return }

        let toConfirm = segments.count - Self.segmentsToKeepUnconfirmed
        guard toConfirm > 0
        else { return }

        let confirmedSegments = Array(segments.prefix(toConfirm))

        let newText =
            confirmedSegments
            .map(\.text)
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let newEnd = confirmedSegments.last?.end ?? clipStart

        guard newEnd > clipStart
        else { return }

        streamLock.withLock { state in
            if state.confirmedText.isEmpty {
                state.confirmedText = newText
            } else {
                state.confirmedText += " " + newText
            }
            state.confirmedEndSeconds = newEnd
        }

        Self.logger.debug(
            "Confirmed \(toConfirm) segment(s) up to \(newEnd)s: \"\(newText.prefix(50))\""
        )
    }

    private func transcribe(
        samples: [Float],
        promptTokens: [Int],
        clipStart: Float = 0
    ) async -> String? {
        let segments = await transcribeSegments(
            samples: samples,
            promptTokens: promptTokens,
            clipStart: clipStart
        )

        let text =
            segments
            .map(\.text)
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return text.isEmpty ? nil : text
    }

    private func transcribeSegments(
        samples: [Float],
        promptTokens: [Int],
        clipStart: Float = 0
    ) async -> [TranscriptionSegment] {
        do {
            var options = DecodingOptions()
            options.language = "en"
            options.skipSpecialTokens = true
            if !promptTokens.isEmpty {
                options.promptTokens = promptTokens
            }
            if clipStart > 0 {
                options.clipTimestamps = [clipStart]
            }

            let results = try await pipe.transcribe(
                audioArray: samples,
                decodeOptions: options
            )
            return results.first?.segments ?? []
        } catch {
            Self.logger.error("Transcription failed: \(error)")
            return []
        }
    }
}

// MARK: - Supporting Types

private struct StreamState {
    /// Combined text from all confirmed segments.
    var confirmedText: String = ""

    /// End timestamp (seconds) of the last confirmed segment.
    ///
    /// Used as clipTimestamps for the next pass so Whisper skips already-confirmed audio.
    var confirmedEndSeconds: Float = 0
}

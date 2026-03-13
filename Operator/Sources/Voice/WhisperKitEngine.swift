import AVFoundation
import WhisperKit
import os

/// WhisperKit-based implementation of ``TranscriptionEngine``.
///
/// Uses a CoreML Whisper model for high-accuracy on-device speech recognition.
/// Supports vocabulary biasing via prompt tokens and post-transcription vocabulary
/// correction.
///
/// Audio is accumulated during recording and transcribed in a single pass on
/// `finishAndTranscribe()`. The WhisperKit pipeline is kept warm in memory
/// between sessions to avoid repeated model loading overhead.
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

    // MARK: - Instance Properties

    private let pipe: WhisperKit
    private let sessionLock = OSAllocatedUnfairLock<SessionState?>(initialState: nil)

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
            state = SessionState(
                promptTokens: resolvedTokens,
                vocabulary: vocab
            )
        }

        Self.logger.debug("WhisperKit session prepared")
    }

    /// Append an audio buffer from the microphone tap.
    public func append(_ buffer: AVAudioPCMBuffer) {
        // Single lock acquisition: check converter, create if needed, convert, append.
        let needsConverter = sessionLock.withLock { $0?.converter == nil && $0 != nil }

        if needsConverter {
            do {
                let converter = try AudioFormatConverter(inputFormat: buffer.format)
                sessionLock.withLock { $0?.converter = converter }
            } catch {
                Self.logger.error("Failed to create audio format converter: \(error)")
                return
            }
        }

        let converter = sessionLock.withLock { $0?.converter }
        guard let samples = converter?.convert(buffer) else {
            return
        }
        sessionLock.withLock { $0?.audioSamples.append(contentsOf: samples) }
    }

    /// Finish recording and return the transcribed text.
    public func finishAndTranscribe() async -> String? {
        let snapshot = sessionLock.withLock { state -> SessionState? in
            guard let session = state else {
                return nil
            }
            return session
        }
        let samples = snapshot?.audioSamples ?? []
        let promptTokens = snapshot?.promptTokens ?? []
        let vocabulary = snapshot?.vocabulary ?? []
        resetSession()

        guard !samples.isEmpty else {
            Self.logger.info("No audio samples captured")
            return nil
        }

        guard let text = await transcribe(samples: samples, promptTokens: promptTokens),
            !text.isEmpty
        else {
            Self.logger.info("Transcription returned empty")
            return nil
        }

        let corrected = VocabularyCorrector.correct(text: text, vocabulary: vocabulary)
        if corrected != text {
            Self.logger.info("Post-correction: \"\(text)\" → \"\(corrected)\"")
        }
        Self.logger.info("Transcription complete: \(corrected)")
        return corrected
    }

    /// Cancel any in-progress session and release resources.
    public func cancel() {
        resetSession()
    }

    // MARK: - Private

    private func resetSession() {
        sessionLock.withLock { $0 = nil }
    }

    private func transcribe(samples: [Float], promptTokens: [Int]) async -> String? {
        do {
            var options = DecodingOptions()
            options.language = "en"
            if !promptTokens.isEmpty {
                options.promptTokens = promptTokens
            }

            let results = try await pipe.transcribe(
                audioArray: samples,
                decodeOptions: options
            )
            return results.first?.text.trimmingCharacters(in: .whitespacesAndNewlines)
        } catch {
            Self.logger.error("Transcription failed: \(error)")
            return nil
        }
    }
}

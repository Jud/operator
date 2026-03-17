import AVFoundation
import VocabularyCorrector
import WhisperKit
import os

// swiftlint:disable:next type_body_length
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

    /// Convert a Duration to milliseconds for logging.
    private static func ms(_ duration: Duration) -> Int {
        let (seconds, attoseconds) = duration.components
        return Int(seconds) * 1000 + Int(attoseconds / 1_000_000_000_000_000 // swiftlint:disable:this number_separator - attoseconds)
    }

    // MARK: - Instance Properties

    private let pipe: WhisperKit
    private let sessionLock = OSAllocatedUnfairLock<SessionState?>(initialState: nil)

    /// Confirmed segments and the clip point for the next pass.
    private let streamLock = OSAllocatedUnfairLock<StreamState>(
        initialState: StreamState()
    )

    /// Flag to stop the background loop without cancelling in-flight inference.
    private let stopLoop = OSAllocatedUnfairLock(initialState: false)

    /// Continuation for the interruptible sleep in the background loop.
    /// Resumed by either a timeout or `finishAndTranscribe` signaling stop.
    private let sleepContinuation = OSAllocatedUnfairLock<UnsafeContinuation<Void, Never>?>(
        initialState: nil
    )

    /// Handle to the periodic re-transcription task.
    private var retranscribeTask: Task<Void, Never>?

    /// Previous session's task, kept so the new loop can await its completion
    /// before running inference. Prevents two concurrent pipe.transcribe() calls.
    private var previousTask: Task<Void, Never>?

    // MARK: - Initialization

    /// Create a WhisperKitEngine with a pre-loaded WhisperKit pipeline.
    ///
    /// The pipeline should be initialized before constructing this engine.
    /// Use ``WhisperKitEngine/create(modelFolder:)`` for convenience.
    public init(pipe: WhisperKit) {
        self.pipe = pipe
    }

    // MARK: - Type Methods

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

    /// Known Whisper hallucination strings produced for silence or very short audio.
    private static let blankPatterns: Set<String> = [
        "[BLANK_AUDIO]", "(BLANK_AUDIO)", "[silence]", "(silence)",
        "[no speech]", "(no speech)", "[inaudible]", "(inaudible)"
    ]

    /// Clean up transcription text — trim whitespace and filter known blank markers.
    private static func cleanText(_ text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return blankPatterns.contains(trimmed) ? "" : trimmed
    }

    // MARK: - TranscriptionEngine

    /// Prepare a new transcription session with optional vocabulary biasing.
    public func prepare(contextualStrings: [String]) throws {
        cancel()

        // Prompt tokens disabled — contextual strings (file paths, session names)
        // were contaminating decoder output under fallback conditions.
        let resolvedTokens: [Int] = []
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
        Self.logger.notice("Session started")
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
    /// Signals the background loop to stop (without cancelling in-flight
    /// inference) and runs a final pass from the last confirmed point.
    public func finishAndTranscribe() async -> String? {  // swiftlint:disable:this function_body_length
        let stopTime = ContinuousClock.now

        // Stop the loop — don't cancel, just flag it to exit after its current pass.
        stopLoop.withLock { $0 = true }
        wakeSleep()
        let pendingTask = retranscribeTask
        retranscribeTask = nil

        // Wait for any in-flight background inference to finish before running the
        // final pass. Two concurrent pipe.transcribe() calls corrupt the pipeline.
        await pendingTask?.value
        let awaitDuration = ContinuousClock.now - stopTime

        let snapshot = sessionLock.withLock { state -> SessionState? in
            let session = state
            state = nil
            return session
        }
        let samples = snapshot?.audioSamples ?? []
        let promptTokens = snapshot?.promptTokens ?? []
        let vocabulary = snapshot?.vocabulary ?? []

        let audioDuration = Float(samples.count) / Float(WhisperKit.sampleRate)

        guard !samples.isEmpty
        else {
            Self.logger.notice("No audio samples captured")
            return nil
        }

        let stream = streamLock.withLock { $0 }

        let awaitMs = Self.ms(awaitDuration)
        let audioFmt = String(format: "%.1f", audioDuration)
        let confirmFmt = String(format: "%.1f", stream.confirmedEndSeconds)
        Self.logger.notice(
            "Stop: \(audioFmt)s audio, confirmed=\(confirmFmt)s, await=\(awaitMs)ms"
        )

        // If no new audio arrived since the last background pass, reuse its
        // result directly — no final decode needed.
        let tailSamples = samples.count - stream.lastPassSampleCount
        let canReuse = !stream.lastPassText.isEmpty && tailSamples == 0

        let finalStart = ContinuousClock.now
        let fullText: String

        if canReuse {
            fullText = stream.lastPassText
            Self.logger.notice("Reused background result (no new audio)")
        } else {
            // Decode from last confirmed point.
            let tailText = await transcribe(
                samples: samples,
                promptTokens: promptTokens,
                clipStart: stream.confirmedEndSeconds
            )
            if stream.confirmedText.isEmpty {
                fullText = tailText ?? ""
            } else if let tail = tailText, !tail.isEmpty {
                fullText = stream.confirmedText + " " + tail
            } else {
                fullText = stream.confirmedText
            }
        }

        let finalDuration = ContinuousClock.now - finalStart
        let cleaned = Self.cleanText(fullText)

        let finalMs = Self.ms(finalDuration)

        guard !cleaned.isEmpty
        else {
            Self.logger.notice("Transcription empty (final=\(finalMs)ms)")
            return nil
        }

        let corrected = VocabularyCorrector.correct(text: cleaned, vocabulary: vocabulary)
        if corrected != cleaned {
            Self.logger.notice(
                "Post-correction: \"\(cleaned, privacy: .public)\" -> \"\(corrected, privacy: .public)\""
            )
        }
        let totalMs = Self.ms(ContinuousClock.now - stopTime)
        let clipSec = stream.confirmedEndSeconds
        let preview = String(corrected.prefix(80))
        let clipFmt = String(format: "%.1f", clipSec)
        Self.logger.notice( // swiftlint:disable:next line_length
            "Result: \(totalMs)ms (await=\(awaitMs)ms final=\(finalMs)ms) clip=\(clipFmt)s \"\(preview, privacy: .public)\""
        )
        return corrected
    }

    /// Cancel any in-progress session and release resources.
    public func cancel() {
        stopLoop.withLock { $0 = true }
        wakeSleep()
        // Stash the old task so the next session's loop can await it before
        // running inference. Two concurrent pipe.transcribe() calls corrupt
        // the pipeline.
        if let old = retranscribeTask {
            previousTask = old
        }
        retranscribeTask = nil
        sessionLock.withLock { $0 = nil }
        streamLock.withLock { $0 = StreamState() }
    }

    // MARK: - Private

    private func startRetranscriptionLoop() {
        stopLoop.withLock { $0 = false }
        let taskToDrain = previousTask
        previousTask = nil
        retranscribeTask = Task { [weak self] in
            // Drain any in-flight inference from the previous session before
            // we touch the pipeline. This prevents pipeline corruption.
            await taskToDrain?.value

            // Adaptive schedule: first pass at 1s, then 500ms, then 1s intervals.
            await self?.interruptibleSleep(nanoseconds: 1_000_000_000)

            var passCount = 0

            while let self, !self.stopLoop.withLock({ $0 }) {
                let (samples, promptTokens) = self.sessionLock.withLock { state -> ([Float], [Int]) in
                    guard let session = state
                    else { return ([], []) }
                    return (session.audioSamples, session.promptTokens)
                }

                let stream = self.streamLock.withLock { $0 }
                let processedSamples = Int(stream.confirmedEndSeconds * Float(WhisperKit.sampleRate))
                let newSamples = samples.count - processedSamples

                if newSamples >= Self.retranscribeThreshold {
                    await self.runBackgroundPass(
                        samples: samples,
                        promptTokens: promptTokens,
                        clipStart: stream.confirmedEndSeconds
                    )
                    passCount += 1
                }

                guard !self.stopLoop.withLock({ $0 })
                else { return }

                // Short interval after first pass to catch short utterances early,
                // then settle into 1s intervals for longer recordings.
                let intervalNs: UInt64 = passCount <= 1 ? 500_000_000 : 1_000_000_000
                await self.interruptibleSleep(nanoseconds: intervalNs)
            }
        }
    }

    /// Sleep for up to `nanoseconds`, but wake immediately if `stopLoop` is set.
    ///
    /// Uses an `UnsafeContinuation` that can be resumed by either:
    /// - A timeout task after the full duration elapses
    /// - `wakeSleep()` called from `finishAndTranscribe()`
    private func interruptibleSleep(nanoseconds: UInt64) async {
        await withUnsafeContinuation { continuation in
            sleepContinuation.withLock { $0 = continuation }

            // If stop was already requested, resume immediately.
            if stopLoop.withLock({ $0 }) {
                sleepContinuation.withLock { cont in
                    cont?.resume()
                    cont = nil
                }
                return
            }

            // Otherwise, resume after the timeout.
            Task { [weak self] in
                try? await Task.sleep(nanoseconds: nanoseconds)
                self?.wakeSleep()
            }
        }
    }

    /// Resume the background loop's sleep immediately.
    private func wakeSleep() {
        sleepContinuation.withLock { cont in
            cont?.resume()
            cont = nil
        }
    }

    /// Run a transcription pass from `clipStart` and confirm stable segments.
    ///
    /// When 2+ segments are produced, all but the last are confirmed (their
    /// boundaries are natural phrase breaks from Whisper's decoder). With only
    /// 1 segment, nothing is confirmed — short utterances fall back to batch.
    private func runBackgroundPass( // swiftlint:disable:this function_body_length
        samples: [Float],
        promptTokens: [Int],
        clipStart: Float
    ) async {
        let sampleCount = samples.count
        let audioDuration = Float(sampleCount) / Float(WhisperKit.sampleRate)
        let passStart = ContinuousClock.now

        let segments = await transcribeSegments(
            samples: samples,
            promptTokens: promptTokens,
            clipStart: clipStart
        )

        let passMs = Self.ms(ContinuousClock.now - passStart)

        // Always cache the latest result so finishAndTranscribe can reuse it
        // instead of re-decoding from scratch when confirmation didn't kick in.
        let allText = Self.cleanText(
            segments.map(\.text).joined(separator: " ")
        )
        if !allText.isEmpty {
            streamLock.withLock { state in
                state.lastPassText = state.confirmedText.isEmpty
                    ? allText
                    : state.confirmedText + " " + allText
                state.lastPassSampleCount = sampleCount
            }
        }

        guard segments.count >= 2
        else {
            let clipFmt = String(format: "%.1f", clipStart)
            let audioFmt = String(format: "%.1f", audioDuration)
            Self.logger.notice(
                "Background: \(segments.count) seg in \(passMs)ms (clip=\(clipFmt)s, audio=\(audioFmt)s)"
            )
            return
        }

        let toConfirm = segments.count - 1

        let confirmedSegments = Array(segments.prefix(toConfirm))

        let newText = Self.cleanText(
            confirmedSegments.map(\.text).joined(separator: " ")
        )
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

        Self.logger.notice(
            "Confirmed \(toConfirm) seg to \(newEnd)s: \"\(newText.prefix(60), privacy: .public)\""
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
            let audioDuration = Float(samples.count) / Float(WhisperKit.sampleRate)
            var options = DecodingOptions()
            options.language = "en"
            options.skipSpecialTokens = true
            options.chunkingStrategy = ChunkingStrategy.none
            // Disable end-of-window clipping for short audio — the default 1s
            // padding causes sub-1s recordings to produce zero segments.
            if audioDuration < 3 {
                options.windowClipTime = 0
            }
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
    /// Used as clipTimestamps for the next pass so Whisper skips confirmed audio.
    var confirmedEndSeconds: Float = 0

    /// Cached result from the most recent background pass (even single-segment).
    /// Used to avoid re-decoding when confirmation didn't kick in.
    var lastPassText: String = ""

    /// Number of audio samples at the time the cached pass started.
    /// The final pass only needs to decode the tail beyond this point.
    var lastPassSampleCount: Int = 0
}

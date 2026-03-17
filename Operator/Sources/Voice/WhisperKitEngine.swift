import AVFoundation
import VocabularyCorrector
import WhisperKit
import os

// swiftlint:disable:next type_body_length
public final class WhisperKitEngine: TranscriptionEngine, @unchecked Sendable {
    private struct SessionState: @unchecked Sendable {
        var audioSamples: [Float] = []
        var converter: AudioFormatConverter?
        var vocabulary: [String] = []
    }

    // MARK: - Type Properties

    private static let logger = Log.logger(for: "WhisperKitEngine")

    /// Minimum new audio (in samples at 16 kHz) before triggering a background pass.
    private static let retranscribeThreshold = 16_000  // 1 second

    /// Number of consecutive agreed words required before confirming.
    private static let agreementCountNeeded = 2

    /// Convert a Duration to milliseconds for logging.
    private static func ms(_ duration: Duration) -> Int {
        let (seconds, attoseconds) = duration.components
        let attoMs: Int64 = 1_000_000_000_000_000
        return Int(seconds) * 1000 + Int(attoseconds / attoMs)
    }

    // MARK: - Instance Properties

    private let pipe: WhisperKit
    private let sessionLock = OSAllocatedUnfairLock<SessionState?>(initialState: nil)

    /// Word-level streaming state, updated by the background loop.
    private let streamLock = OSAllocatedUnfairLock<StreamState>(
        initialState: StreamState()
    )

    /// Flag to stop the background loop without cancelling in-flight inference.
    private let stopLoop = OSAllocatedUnfairLock(initialState: false)

    /// Continuation for the interruptible sleep in the background loop.
    private let sleepContinuation = OSAllocatedUnfairLock<UnsafeContinuation<Void, Never>?>(
        initialState: nil
    )

    /// Handle to the periodic re-transcription task.
    private var retranscribeTask: Task<Void, Never>?

    /// Previous session's task — new loop awaits it to prevent concurrent pipeline use.
    private var previousTask: Task<Void, Never>?

    // MARK: - Initialization

    public init(pipe: WhisperKit) {
        self.pipe = pipe
    }

    // MARK: - Type Methods

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

    private static func cleanText(_ text: String) -> String {
        var result = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if blankPatterns.contains(result) { return "" }
        // Collapse double spaces from word-joining boundaries
        while result.contains("  ") {
            result = result.replacingOccurrences(of: "  ", with: " ")
        }
        return result
    }

    // MARK: - TranscriptionEngine

    public func prepare(contextualStrings: [String]) throws {
        cancel()

        let vocab = contextualStrings
        sessionLock.withLock { state in
            var session = SessionState(vocabulary: vocab)
            session.audioSamples.reserveCapacity(480_000)
            state = session
        }
        streamLock.withLock { $0 = StreamState() }

        startRetranscriptionLoop()
        Self.logger.notice("Session started")
    }

    public func append(_ buffer: AVAudioPCMBuffer) {
        nonisolated(unsafe) let buf = buffer
        let converter: AudioFormatConverter? = sessionLock.withLock { state in
            guard state != nil else { return nil }
            if let existing = state?.converter { return existing }
            guard let conv = try? AudioFormatConverter(inputFormat: buf.format)
            else { return nil }
            state?.converter = conv
            return conv
        }
        guard let converter else { return }
        sessionLock.withLock { state in
            guard var session = state else { return }
            _ = converter.appendConverted(buf, to: &session.audioSamples)
            state = session
        }
    }

    // swiftlint:disable:next function_body_length
    public func finishAndTranscribe() async -> String? {
        let stopTime = ContinuousClock.now

        stopLoop.withLock { $0 = true }
        wakeSleep()
        let pendingTask = retranscribeTask
        retranscribeTask = nil

        await pendingTask?.value
        let awaitDuration = ContinuousClock.now - stopTime

        let snapshot = sessionLock.withLock { state -> SessionState? in
            let session = state
            state = nil
            return session
        }
        let samples = snapshot?.audioSamples ?? []
        let vocabulary = snapshot?.vocabulary ?? []

        guard !samples.isEmpty else {
            Self.logger.notice("No audio samples captured")
            return nil
        }

        let stream = streamLock.withLock { $0 }
        let awaitMs = Self.ms(awaitDuration)
        let audioSec = String(format: "%.1f", Float(samples.count) / Float(WhisperKit.sampleRate))
        let agreedSec = String(format: "%.1f", stream.lastAgreedSeconds)
        Self.logger.notice(
            "Stop: \(audioSec)s audio, agreed=\(agreedSec)s, await=\(awaitMs)ms"
        )

        // Final pass from the last confirmed point.
        let finalStart = ContinuousClock.now
        let fullText: String

        if stream.useSegmentFallback {
            // Segment fallback path: decode tail from segment clip point.
            let tailResult = await transcribeResult(
                samples: samples,
                clipStart: stream.segmentConfirmedEndSeconds
            )
            let tailText = tailResult?.text.trimmingCharacters(in: .whitespaces) ?? ""
            Self.logger.notice("Segment final: confirmed=\"\(stream.segmentConfirmedText.prefix(40), privacy: .public)\" tail=\"\(tailText.prefix(60), privacy: .public)\"")
            if !stream.segmentConfirmedText.isEmpty && !tailText.isEmpty {
                fullText = Self.cleanText(stream.segmentConfirmedText + " " + tailText)
            } else if !stream.segmentConfirmedText.isEmpty {
                // Tail decode failed — fall back to full decode from 0.0s.
                Self.logger.notice("Tail empty — falling back to full decode")
                let fullResult = await transcribeResult(samples: samples)
                let fullDecode = fullResult?.text ?? ""
                Self.logger.notice("Full decode: \"\(fullDecode.prefix(80), privacy: .public)\"")
                fullText = Self.cleanText(fullDecode.isEmpty ? stream.segmentConfirmedText : fullDecode)
            } else {
                fullText = Self.cleanText(tailText)
            }
        } else {
            // Word-level path: decode from last agreed point with prefix context.
            let finalResult = await transcribeResult(
                samples: samples,
                clipStart: stream.lastAgreedSeconds,
                prefixTokens: stream.lastAgreedWords.flatMap { $0.tokens }
            )

            var finalWords = stream.confirmedWords

            if let result = finalResult, result.segments.first?.words != nil {
                let finalPassWords = result.allWords
                    .filter { $0.start >= stream.lastAgreedSeconds }
                finalWords.append(contentsOf: finalPassWords)
            } else {
                finalWords.append(contentsOf: stream.lastAgreedWords)
                if let text = finalResult?.text, !text.isEmpty {
                    finalWords.append(WordTiming(
                        word: text, tokens: [], start: stream.lastAgreedSeconds,
                        end: Float(samples.count) / Float(WhisperKit.sampleRate),
                        probability: 1.0
                    ))
                }
            }

            var candidateText = Self.cleanText(finalWords.map(\.word).joined())

            // Safety: if confirmed+final is suspiciously short relative to audio,
            // the clip-based decode may have failed. Fall back to full decode.
            let audioDur = Float(samples.count) / Float(WhisperKit.sampleRate)
            let wordsPerSec = Float(candidateText.split(separator: " ").count) / audioDur
            if audioDur > 3 && wordsPerSec < 0.5 {
                Self.logger.notice("Word result too short (\(candidateText.count) chars for \(audioSec)s) — full decode")
                let fullResult = await transcribeResult(samples: samples)
                let fullDecode = Self.cleanText(fullResult?.text ?? "")
                if fullDecode.count > candidateText.count {
                    candidateText = fullDecode
                }
            }

            fullText = candidateText
        }
        let finalMs = Self.ms(ContinuousClock.now - finalStart)

        guard !fullText.isEmpty else {
            Self.logger.notice("Transcription empty (final=\(finalMs)ms)")
            return nil
        }

        let corrected = VocabularyCorrector.correct(text: fullText, vocabulary: vocabulary)
        let totalMs = Self.ms(ContinuousClock.now - stopTime)
        let preview = String(corrected.prefix(80))
        Self.logger.notice(
            "Result: \(totalMs)ms (await=\(awaitMs)ms final=\(finalMs)ms) \"\(preview, privacy: .public)\""
        )
        return corrected
    }

    public func cancel() {
        stopLoop.withLock { $0 = true }
        wakeSleep()
        if let old = retranscribeTask { previousTask = old }
        retranscribeTask = nil
        sessionLock.withLock { $0 = nil }
        streamLock.withLock { $0 = StreamState() }
    }

    // MARK: - Background Loop

    private func startRetranscriptionLoop() {
        stopLoop.withLock { $0 = false }
        let taskToDrain = previousTask
        previousTask = nil
        retranscribeTask = Task { [weak self] in
            await taskToDrain?.value
            await self?.interruptibleSleep(nanoseconds: 1_000_000_000)

            var passCount = 0
            while let self, !self.stopLoop.withLock({ $0 }) {
                let samples = self.sessionLock.withLock { state -> [Float] in
                    state?.audioSamples ?? []
                }

                let stream = self.streamLock.withLock { $0 }
                let clipSec = stream.useSegmentFallback
                    ? stream.segmentConfirmedEndSeconds
                    : stream.lastAgreedSeconds
                let processedSamples = Int(clipSec * Float(WhisperKit.sampleRate))
                let newSamples = samples.count - processedSamples

                if newSamples >= Self.retranscribeThreshold {
                    await self.runBackgroundPass(samples: samples)
                    passCount += 1
                }

                guard !self.stopLoop.withLock({ $0 }) else { return }
                let intervalNs: UInt64 = passCount <= 1 ? 500_000_000 : 1_000_000_000
                await self.interruptibleSleep(nanoseconds: intervalNs)
            }
        }
    }

    private func interruptibleSleep(nanoseconds: UInt64) async {
        await withUnsafeContinuation { continuation in
            sleepContinuation.withLock { $0 = continuation }
            if stopLoop.withLock({ $0 }) {
                sleepContinuation.withLock { cont in
                    cont?.resume()
                    cont = nil
                }
                return
            }
            Task { [weak self] in
                try? await Task.sleep(nanoseconds: nanoseconds)
                self?.wakeSleep()
            }
        }
    }

    private func wakeSleep() {
        sleepContinuation.withLock { cont in
            cont?.resume()
            cont = nil
        }
    }

    // MARK: - Word-Level Streaming

    /// Consecutive word-level misses before falling back to segment-level confirmation.
    private static let maxWordMisses = 3

    /// Run a background transcription pass with tiered confirmation:
    /// 1. Word-level agreement (primary — granular, fast)
    /// 2. Segment-level confirmation (fallback after 3 word misses)
    /// 3. Full decode on final pass (when neither can confirm)
    private func runBackgroundPass(samples: [Float]) async {
        let passStart = ContinuousClock.now
        let stream = streamLock.withLock { $0 }

        let clipStart = stream.useSegmentFallback
            ? stream.segmentConfirmedEndSeconds
            : stream.lastAgreedSeconds

        let result = await transcribeResult(
            samples: samples,
            clipStart: clipStart,
            prefixTokens: stream.useSegmentFallback
                ? []
                : stream.lastAgreedWords.flatMap { $0.tokens }
        )

        let passMs = Self.ms(ContinuousClock.now - passStart)
        let audioSec = String(format: "%.1f", Float(samples.count) / Float(WhisperKit.sampleRate))

        guard let result else {
            Self.logger.notice("Background: no result in \(passMs)ms (audio=\(audioSec)s)")
            return
        }

        streamLock.withLock { state in
            if state.useSegmentFallback {
                runSegmentFallback(state: &state, result: result, passMs: passMs, audioSec: audioSec)
            } else {
                runWordAgreement(state: &state, result: result, samples: samples, passMs: passMs, audioSec: audioSec)
            }

            state.prevResult = result
            state.lastPassSampleCount = samples.count
        }
    }

    /// Word-level agreement path.
    private func runWordAgreement(
        state: inout StreamState,
        result: TranscriptionResult,
        samples: [Float],
        passMs: Int,
        audioSec: String
    ) {
        guard result.segments.first?.words != nil else {
            Self.logger.notice("Background: no words in \(passMs)ms (audio=\(audioSec)s)")
            return
        }

        let hypothesisWords = result.allWords.filter { $0.start >= state.lastAgreedSeconds }

        let commonPrefix = TranscriptionUtilities.findLongestCommonPrefix(
            state.prevWords, hypothesisWords
        )

        if commonPrefix.count >= Self.agreementCountNeeded {
            let newAgreed = Array(commonPrefix.suffix(Self.agreementCountNeeded))
            let toConfirm = commonPrefix.prefix(commonPrefix.count - Self.agreementCountNeeded)

            state.confirmedWords.append(contentsOf: toConfirm)
            state.lastAgreedWords = newAgreed
            state.lastAgreedSeconds = newAgreed.first!.start
            state.consecutiveMisses = 0

            let agreedSec = String(format: "%.1f", state.lastAgreedSeconds)
            Self.logger.notice("Confirmed \(toConfirm.count) words to \(agreedSec)s in \(passMs)ms")
            if !toConfirm.isEmpty {
                let newText = toConfirm.map(\.word).joined()
                Self.logger.notice("  new: \"\(newText.prefix(60), privacy: .public)\"")
            }
        } else {
            state.consecutiveMisses += 1

            if state.consecutiveMisses >= Self.maxWordMisses {
                // Switch to segment-level fallback.
                state.useSegmentFallback = true
                // Carry over any confirmed words as text for the segment path.
                let wordText = (state.confirmedWords + state.lastAgreedWords).map(\.word).joined()
                    .trimmingCharacters(in: .whitespaces)
                state.segmentConfirmedText = wordText
                state.segmentConfirmedEndSeconds = state.lastAgreedSeconds

                let misses = state.consecutiveMisses
                Self.logger.notice(
                    "Word agreement stalled (\(misses) misses) — switching to segment fallback"
                )
            } else {
                let misses = state.consecutiveMisses
                Self.logger.notice(
                    "Background: \(hypothesisWords.count) words, \(commonPrefix.count) agreed, miss \(misses)/\(Self.maxWordMisses) in \(passMs)ms"
                )
            }
        }

        let clipSeconds = state.lastAgreedSeconds
        state.prevWords = hypothesisWords.filter { $0.start >= clipSeconds }
    }

    /// Segment-level confirmation fallback.
    private func runSegmentFallback(
        state: inout StreamState,
        result: TranscriptionResult,
        passMs: Int,
        audioSec: String
    ) {
        let segments = result.segments
        guard segments.count >= 2 else {
            Self.logger.notice(
                "Segment fallback: \(segments.count) seg in \(passMs)ms (audio=\(audioSec)s)"
            )
            return
        }

        let toConfirm = Array(segments.prefix(segments.count - 1))
        let newText = toConfirm.map(\.text).joined(separator: " ")
            .trimmingCharacters(in: .whitespaces)
        let newEnd = toConfirm.last?.end ?? state.segmentConfirmedEndSeconds

        guard newEnd > state.segmentConfirmedEndSeconds else { return }

        if state.segmentConfirmedText.isEmpty {
            state.segmentConfirmedText = newText
        } else {
            state.segmentConfirmedText += " " + newText
        }
        state.segmentConfirmedEndSeconds = newEnd

        let endFmt = String(format: "%.1f", newEnd)
        Self.logger.notice(
            "Segment confirmed \(toConfirm.count) seg to \(endFmt)s in \(passMs)ms"
        )
    }

    // MARK: - Transcription

    private func transcribeResult(
        samples: [Float],
        clipStart: Float = 0,
        prefixTokens: [Int] = []
    ) async -> TranscriptionResult? {
        do {
            let audioDuration = Float(samples.count) / Float(WhisperKit.sampleRate)
            var options = DecodingOptions()
            options.language = "en"
            options.skipSpecialTokens = true
            options.wordTimestamps = true
            options.chunkingStrategy = ChunkingStrategy.none
            if audioDuration < 3 {
                options.windowClipTime = 0
            }
            if clipStart > 0 {
                options.clipTimestamps = [clipStart]
            }
            if !prefixTokens.isEmpty {
                options.prefixTokens = prefixTokens
            }

            let results = try await pipe.transcribe(
                audioArray: samples,
                decodeOptions: options
            )
            return results.first
        } catch {
            Self.logger.error("Transcription failed: \(error)")
            return nil
        }
    }
}

// MARK: - Supporting Types

private struct StreamState {
    /// Words confirmed via word-level agreement across consecutive passes.
    var confirmedWords: [WordTiming] = []

    /// The last N agreed words used as decoder context (prefixTokens) for the next pass.
    var lastAgreedWords: [WordTiming] = []

    /// Timestamp (seconds) of the first last-agreed word — used as clipTimestamps origin.
    var lastAgreedSeconds: Float = 0

    /// Words from the previous background pass, used for agreement comparison.
    var prevWords: [WordTiming] = []

    /// Full result from the previous pass (for suffix extraction in finishAndTranscribe).
    var prevResult: TranscriptionResult?

    /// Sample count at the time of the last background pass (for reuse detection).
    var lastPassSampleCount: Int = 0

    /// Consecutive word-agreement misses. Resets on successful agreement.
    var consecutiveMisses: Int = 0

    /// When true, word-level agreement has stalled and we use segment-level confirmation.
    var useSegmentFallback: Bool = false

    /// Combined text from segment-level confirmation (used when in fallback mode).
    var segmentConfirmedText: String = ""

    /// End timestamp of last segment-level confirmation.
    var segmentConfirmedEndSeconds: Float = 0
}

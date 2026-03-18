import AVFoundation
import VocabularyCorrector
import WhisperKit
import os

/// WhisperKit-based speech-to-text with word-level streaming confirmation.
public final class WhisperKitEngine: TranscriptionEngine, @unchecked Sendable {
    private struct SessionState: @unchecked Sendable {
        var audioSamples: [Float] = []
        var converter: AudioFormatConverter?
        var vocabulary: [String] = []
    }

    private static let logger = Log.logger(for: "WhisperKitEngine")

    /// Minimum new audio (in samples at 16 kHz) before triggering a background pass.
    private static let retranscribeThreshold = 16_000

    /// Number of consecutive agreed words required before confirming.
    private static let agreementCountNeeded = 2

    /// Consecutive word-level misses before falling back to segment-level confirmation.
    private static let maxWordMisses = 3

    /// Known Whisper hallucination strings produced for silence or very short audio.
    private static let blankPatterns: Set<String> = [
        "[BLANK_AUDIO]", "(BLANK_AUDIO)", "[silence]", "(silence)",
        "[no speech]", "(no speech)", "[inaudible]", "(inaudible)"
    ]

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

    /// Create with a pre-loaded WhisperKit pipeline.
    public init(pipe: WhisperKit) {
        self.pipe = pipe
    }

    /// Load a WhisperKit model from disk and create an engine.
    public static func create(modelFolder: String) async throws -> WhisperKitEngine {
        let config = WhisperKitConfig(modelFolder: modelFolder)
        let pipe = try await WhisperKit(config)
        logger.info("WhisperKit pipeline loaded from \(modelFolder)")
        return WhisperKitEngine(pipe: pipe)
    }

    private static func ms(_ duration: Duration) -> Int {
        let (seconds, attoseconds) = duration.components
        return Int(seconds) * 1_000 + Int(attoseconds / 1_000_000_000_000_000)
    }

    /// Regex matching text that is entirely parenthesized or bracketed non-speech annotations.
    ///
    /// WhisperKit emits things like "(air whooshes)", "[clicking]", "(popping sound)" when
    /// it hears non-speech audio. These should be treated as empty transcriptions.
    nonisolated(unsafe) private static let nonSpeechPattern = /^\s*[\(\[].+[\)\]]\s*$/

    private static func cleanText(_ text: String) -> String {
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

    // MARK: - TranscriptionEngine

    /// Prepare a new transcription session with optional vocabulary biasing.
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
    public func finishAndTranscribe() async -> String? {
        let stopTime = ContinuousClock.now

        stopLoop.withLock { $0 = true }
        wakeSleep()
        let pendingTask = retranscribeTask
        retranscribeTask = nil

        // Cancel so any in-flight pipe.transcribe() bails at the next
        // Task.checkCancellation() point. We still await to ensure the pipeline is free.
        pendingTask?.cancel()
        await pendingTask?.value
        let awaitDuration = ContinuousClock.now - stopTime

        let snapshot = sessionLock.withLock { state -> SessionState? in
            let session = state
            state = nil
            return session
        }
        let samples = snapshot?.audioSamples ?? []
        let vocabulary = snapshot?.vocabulary ?? []

        guard !samples.isEmpty
        else {
            Self.logger.notice("No audio samples captured")
            return nil
        }

        let stream = streamLock.withLock { $0 }
        let awaitMs = Self.ms(awaitDuration)
        let audioDur = Float(samples.count) / Float(WhisperKit.sampleRate)
        let confirmedWordCount = stream.confirmedText.split(separator: " ").count
        let agreedWordTexts = stream.lastAgreedWords.map(\.word).joined()
        let prefixTokens = stream.lastAgreedWords.flatMap(\.tokens)
        let unconfirmedTail = audioDur - stream.lastAgreedSeconds
        let zp = stream.consecutiveZeroProgress
        let seg = stream.useSegmentFallback
        let agreed = stream.lastAgreedSeconds
        Self.logger.notice(
            "Stop: \(audioDur, privacy: .public)s audio, agreed=\(agreed, privacy: .public)s, await=\(awaitMs)ms"
        )
        Self.logger.notice(
            "Stop: confirmed=\(confirmedWordCount) words, zp=\(zp), seg=\(seg)"
        )

        let finalStart = ContinuousClock.now
        var fullText = await decodeFinalText(stream: stream, samples: samples)

        // Safety net: if result is too short for the audio, the clipped decode
        // likely failed. Fall back to full decode from 0.0s.
        let wordCount = Float(fullText.split(separator: " ").count)
        if audioDur > 3 && wordCount / audioDur < 0.5 && !fullText.isEmpty {
            Self.logger.notice(
                "Result too short (\(Int(wordCount)) words for \(audioDur, privacy: .public)s) — full decode"
            )
            if let rescue = await transcribeResult(samples: samples) {
                let rescueText = Self.cleanText(rescue.text)
                if rescueText.count > fullText.count {
                    fullText = rescueText
                }
            }
        }
        let finalMs = Self.ms(ContinuousClock.now - finalStart)

        if fullText.isEmpty && audioDur > 1 {
            Self.logger.notice("Empty result on \(audioDur, privacy: .public)s audio — full decode rescue")
            if let rescue = await transcribeResult(samples: samples) {
                fullText = Self.cleanText(rescue.text)
            }
        }

        guard !fullText.isEmpty
        else {
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

    /// Cancel any in-progress session and release resources.
    public func cancel() {
        stopLoop.withLock { $0 = true }
        wakeSleep()
        if let old = retranscribeTask {
            previousTask = old
        }
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
                let clipSec =
                    stream.useSegmentFallback
                    ? stream.segmentConfirmedEndSeconds
                    : stream.lastAgreedSeconds
                let processedSamples = Int(clipSec * Float(WhisperKit.sampleRate))
                let newSamples = samples.count - processedSamples

                if newSamples >= Self.retranscribeThreshold {
                    await self.runBackgroundPass(samples: samples)
                    passCount += 1
                }

                guard !self.stopLoop.withLock({ $0 })
                else { return }
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

    // MARK: - Tiered Confirmation

    private func runBackgroundPass(samples: [Float]) async {
        let passStart = ContinuousClock.now
        let stream = streamLock.withLock { $0 }

        let clipStart =
            stream.useSegmentFallback
            ? stream.segmentConfirmedEndSeconds
            : stream.lastAgreedSeconds

        let result = await transcribeResult(
            samples: samples,
            clipStart: clipStart,
            prefixTokens: stream.useSegmentFallback
                ? []
                : stream.lastAgreedWords.flatMap(\.tokens)
        )

        let passMs = Self.ms(ContinuousClock.now - passStart)
        let audioFmt = String(format: "%.1f", Float(samples.count) / Float(WhisperKit.sampleRate))

        guard let result
        else {
            Self.logger.notice("Background: no result in \(passMs)ms (audio=\(audioFmt)s)")
            return
        }

        streamLock.withLock { state in
            if state.useSegmentFallback {
                confirmSegments(state: &state, result: result, passMs: passMs, audioSec: audioFmt)
            } else {
                confirmWords(state: &state, result: result, passMs: passMs, audioSec: audioFmt)
            }
        }
    }

    private func confirmWords(
        state: inout StreamState,
        result: TranscriptionResult,
        passMs: Int,
        audioSec: String
    ) {
        guard result.segments.first?.words != nil
        else {
            Self.logger.notice("Background: no words in \(passMs)ms (audio=\(audioSec)s)")
            return
        }

        let hypothesisWords = result.allWords.filter { $0.start >= state.lastAgreedSeconds }
        let commonPrefix = TranscriptionUtilities.findLongestCommonPrefix(
            state.prevWords,
            hypothesisWords
        )

        if !state.prevWords.isEmpty {
            let prevSnippet = String(state.prevWords.prefix(6).map(\.word).joined().prefix(40))
            let hypoSnippet = String(hypothesisWords.prefix(6).map(\.word).joined().prefix(40))
            let pc = state.prevWords.count
            let hc = hypothesisWords.count
            let cc = commonPrefix.count
            Self.logger.notice(
                "Word compare: prev=\(pc)[\"\(prevSnippet, privacy: .public)\"] common=\(cc)"
            )
            Self.logger.notice(
                "Word compare: hypo=\(hc)[\"\(hypoSnippet, privacy: .public)\"]"
            )
        }

        if commonPrefix.count >= Self.agreementCountNeeded {
            let newAgreed = Array(commonPrefix.suffix(Self.agreementCountNeeded))
            let toConfirm = commonPrefix.prefix(commonPrefix.count - Self.agreementCountNeeded)

            let newText = toConfirm.map(\.word).joined()
            state.confirmedText += newText
            state.lastAgreedWords = newAgreed
            if let first = newAgreed.first {
                state.lastAgreedSeconds = first.start
            }
            state.consecutiveMisses = 0

            if toConfirm.isEmpty {
                state.consecutiveZeroProgress += 1
            } else {
                state.consecutiveZeroProgress = 0
            }

            let agreedAt = state.lastAgreedSeconds
            let zp = state.consecutiveZeroProgress
            let confMsg = "Confirmed \(toConfirm.count) words to \(agreedAt)s in \(passMs)ms (zeroProgress=\(zp))"
            Self.logger.notice("\(confMsg, privacy: .public)")
            if !newText.isEmpty {
                Self.logger.notice("  new: \"\(newText.prefix(60), privacy: .public)\"")
            }
        } else {
            state.consecutiveMisses += 1
            let misses = state.consecutiveMisses

            if misses >= Self.maxWordMisses {
                state.useSegmentFallback = true
                let wordText = state.confirmedText + state.lastAgreedWords.map(\.word).joined()
                state.segmentConfirmedText = wordText.trimmingCharacters(in: .whitespaces)
                state.segmentConfirmedEndSeconds = state.lastAgreedSeconds

                Self.logger.notice("Word agreement stalled (\(misses) misses) — segment fallback")
            } else {
                let wc = hypothesisWords.count
                let ac = commonPrefix.count
                Self.logger.notice(
                    "Background: \(wc) words, \(ac) agreed, miss \(misses)/\(Self.maxWordMisses) in \(passMs)ms"
                )
            }
        }

        let clip = state.lastAgreedSeconds
        state.prevWords = hypothesisWords.filter { $0.start >= clip }
    }

    private func confirmSegments(
        state: inout StreamState,
        result: TranscriptionResult,
        passMs: Int,
        audioSec: String
    ) {
        let segments = result.segments
        guard segments.count >= 2
        else {
            Self.logger.notice("Segment fallback: \(segments.count) seg in \(passMs)ms (audio=\(audioSec)s)")
            return
        }

        let toConfirm = Array(segments.prefix(segments.count - 1))
        let newText = toConfirm.map(\.text).joined(separator: " ").trimmingCharacters(in: .whitespaces)
        let newEnd = toConfirm.last?.end ?? state.segmentConfirmedEndSeconds

        guard newEnd > state.segmentConfirmedEndSeconds
        else { return }

        if state.segmentConfirmedText.isEmpty {
            state.segmentConfirmedText = newText
        } else {
            state.segmentConfirmedText += " " + newText
        }
        state.segmentConfirmedEndSeconds = newEnd

        Self.logger.notice("Segment confirmed \(toConfirm.count) seg to \(newEnd, privacy: .public)s in \(passMs)ms")
    }

    // MARK: - Transcription

    private func decodeFinalText(stream: StreamState, samples: [Float]) async -> String {
        if stream.useSegmentFallback {
            let tail = await transcribeResult(samples: samples, clipStart: stream.segmentConfirmedEndSeconds)
            let tailText = tail?.text.trimmingCharacters(in: .whitespaces) ?? ""
            if !stream.segmentConfirmedText.isEmpty && !tailText.isEmpty {
                return Self.cleanText(stream.segmentConfirmedText + " " + tailText)
            }
            return Self.cleanText(stream.segmentConfirmedText)
        }

        let finalResult = await transcribeResult(
            samples: samples,
            clipStart: stream.lastAgreedSeconds,
            prefixTokens: stream.lastAgreedWords.flatMap(\.tokens)
        )
        let tailText: String
        if let result = finalResult, result.segments.first?.words != nil {
            let allWords = result.allWords
            let filtered = allWords.filter { $0.start >= stream.lastAgreedSeconds }
            tailText = filtered.map(\.word).joined()
            if allWords.count != filtered.count {
                let droppedCount = allWords.count - filtered.count
                let droppedText = allWords.filter { $0.start < stream.lastAgreedSeconds }.map(\.word).joined()
                Self.logger.notice(
                    "Final decode: \(allWords.count) words, \(droppedCount) filtered"
                )
            }
            let tailSnippet = String(tailText.prefix(80))
            Self.logger.notice(
                "Final tail: \(filtered.count) words from \(stream.lastAgreedSeconds, privacy: .public)s"
            )
            Self.logger.notice("  \"\(tailSnippet, privacy: .public)\"")
        } else {
            tailText = finalResult?.text ?? stream.lastAgreedWords.map(\.word).joined()
            Self.logger.notice(
                "Final tail (no words): \"\(tailText.prefix(80), privacy: .public)\""
            )
        }
        let combined = Self.cleanText(stream.confirmedText + tailText)
        let combinedSnippet = String(combined.prefix(80))
        let cc = stream.confirmedText.count
        let tc = tailText.count
        Self.logger.notice(
            "Final combined: confirmed=\(cc) chars + tail=\(tc) chars = \"\(combinedSnippet, privacy: .public)\""
        )
        return combined
    }

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

            let results = try await pipe.transcribe(audioArray: samples, decodeOptions: options)
            return results.first
        } catch {
            Self.logger.error("Transcription failed: \(error)")
            return nil
        }
    }
}

// MARK: - Stream State

private struct StreamState {
    var confirmedText: String = ""
    var lastAgreedWords: [WordTiming] = []
    var lastAgreedSeconds: Float = 0
    var prevWords: [WordTiming] = []
    var consecutiveMisses: Int = 0
    var consecutiveZeroProgress: Int = 0
    var useSegmentFallback: Bool = false
    var segmentConfirmedText: String = ""
    var segmentConfirmedEndSeconds: Float = 0
}

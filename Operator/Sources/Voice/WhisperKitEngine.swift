import AVFoundation
import VocabularyCorrector
import WhisperKit
import os

#if DEBUG
    import AppKit
#endif

/// WhisperKit-based speech-to-text with word-level streaming confirmation.
public final class WhisperKitEngine: TranscriptionEngine, SchedulableEngine, @unchecked Sendable {
    private struct SessionState: @unchecked Sendable {
        var audioSamples: [Float] = []
        var converter: AudioFormatConverter?
        var vocabulary: [String] = []
    }

    private static let logger = Log.logger(for: "WhisperKitEngine")

    private let pipe: WhisperKit
    private let sessionLock = OSAllocatedUnfairLock<SessionState?>(initialState: nil)

    /// Transcription session state machine -- updated by the background loop (serial via scheduler).
    private var session = TranscriptionSession()

    /// Filter pipeline applied to all transcription results.
    private var filters: [TranscriptionFilter] = [HallucinationTimestampFilter()]

    /// Scheduler controlling background transcription timing.
    private var scheduler: TranscriptionScheduler

    /// Task running the scheduler's start(engine:) loop.
    private var schedulerTask: Task<Void, Never>?

    /// Previous session's scheduler task -- new session awaits it to prevent concurrent pipeline use.
    private var previousSchedulerTask: Task<Void, Never>?

    /// Tracks whether the last extendTranscript call produced zero new words.
    /// When true, overlap tokens are dropped to avoid hallucination loops.
    private var lastExtensionWasZeroProgress: Bool = false

    /// Create with a pre-loaded WhisperKit pipeline.
    public init(pipe: WhisperKit, scheduler: TranscriptionScheduler = BackgroundLoopScheduler()) {
        self.pipe = pipe
        self.scheduler = scheduler
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

    // MARK: - SchedulableEngine

    public var sampleCount: Int {
        sessionLock.withLock { $0?.audioSamples.count ?? 0 }
    }

    public func transcribeWorkingWindow(maxSampleCount: Int?) async {
        let passStart = ContinuousClock.now

        let samples = sessionLock.withLock { state -> [Float] in
            let s = state?.audioSamples ?? []
            return maxSampleCount.map { Array(s.prefix($0)) } ?? s
        }

        let frontier = session.frontier

        let result = await transcribeResult(
            samples: samples,
            clipStart: frontier,
            prefixTokens: lastExtensionWasZeroProgress ? [] : session.overlapTokens
        )

        let passMs = Self.ms(ContinuousClock.now - passStart)
        let audioFmt = String(format: "%.1f", Float(samples.count) / Float(WhisperKit.sampleRate))

        guard let result else {
            Self.logger.notice("Background: no result in \(passMs)ms (audio=\(audioFmt)s)")
            return
        }

        let filtered = runFilterPipeline(result, samples: samples, frontier: frontier)

        guard !filtered.allWords.isEmpty else {
            Self.logger.notice("Background: empty after filtering in \(passMs)ms (audio=\(audioFmt)s)")
            return
        }

        let outcome = session.extendTranscript(with: filtered)
        lastExtensionWasZeroProgress = (outcome == .extended(newWords: 0))

        switch outcome {
        case .extended(let newWords):
            let f = session.frontier
            let zp = lastExtensionWasZeroProgress
            Self.logger.notice(
                "Confirmed \(newWords) words to \(f, privacy: .public)s in \(passMs)ms (zeroProgress=\(zp))"
            )

        case .matchFailed:
            let misses = session.consecutiveMatchFailures
            let strategy = session.strategy
            if strategy == .segmentLevel {
                Self.logger.notice("Word agreement stalled (\(misses) misses) -- segment fallback")
            } else {
                Self.logger.notice(
                    "Background: match failed, miss \(misses) in \(passMs)ms"
                )
            }
        }
    }

    // MARK: - TranscriptionEngine

    /// Prepare a new transcription session with optional vocabulary biasing.
    public func prepare(contextualStrings: [String]) throws {
        cancel()

        let vocab = contextualStrings
        sessionLock.withLock { state in
            var s = SessionState(vocabulary: vocab)
            s.audioSamples.reserveCapacity(480_000)
            state = s
        }
        session = TranscriptionSession()
        lastExtensionWasZeroProgress = false

        let taskToDrain = previousSchedulerTask
        previousSchedulerTask = nil
        let sched = scheduler
        schedulerTask = Task { [weak self] in
            await taskToDrain?.value
            guard let self else { return }
            await sched.start(engine: self)
        }
        Self.logger.notice("Session started")
    }

    /// Append an audio buffer from the microphone tap.
    public func append(_ buffer: AVAudioPCMBuffer) {
        nonisolated(unsafe) let buf = buffer
        sessionLock.withLock { state in
            guard var s = state
            else { return }
            if s.converter == nil {
                s.converter = try? AudioFormatConverter(inputFormat: buf.format)
            }
            if let converter = s.converter {
                _ = converter.appendConverted(buf, to: &s.audioSamples)
            }
            state = s
        }
    }

    /// Finish recording and return the transcribed text.
    public func finishAndTranscribe() async -> String? {
        let stopTime = ContinuousClock.now

        await scheduler.stop()
        let pendingTask = schedulerTask
        schedulerTask = nil
        pendingTask?.cancel()
        await pendingTask?.value
        let awaitDuration = ContinuousClock.now - stopTime

        let snapshot = sessionLock.withLock { state -> SessionState? in
            let s = state
            state = nil
            return s
        }
        let samples = snapshot?.audioSamples ?? []
        let vocabulary = snapshot?.vocabulary ?? []

        guard !samples.isEmpty
        else {
            Self.logger.notice("No audio samples captured")
            return nil
        }

        let frontier = session.frontier
        let awaitMs = Self.ms(awaitDuration)
        let audioDur = Float(samples.count) / Float(WhisperKit.sampleRate)
        let confirmedWordCount = session.transcript.split(separator: " ").count
        let strategy = session.strategy
        Self.logger.notice(
            "Stop: \(audioDur, privacy: .public)s audio, frontier=\(frontier, privacy: .public)s, await=\(awaitMs)ms"
        )
        Self.logger.notice(
            "Stop: confirmed=\(confirmedWordCount) words, strategy=\(String(describing: strategy), privacy: .public)"
        )

        let finalStart = ContinuousClock.now

        // Tail transcription with VAD chunking
        let tailResult = await transcribeResult(
            samples: samples,
            clipStart: frontier,
            prefixTokens: lastExtensionWasZeroProgress ? [] : session.overlapTokens,
            chunked: true
        )

        let filteredTail = tailResult.map {
            runFilterPipeline($0, samples: samples, frontier: frontier)
        }

        var fullText = assembleResult(
            transcript: session.transcript,
            tailTranscription: filteredTail,
            frontier: frontier
        )

        // Safety net: if result is too short for the audio, the clipped decode
        // likely failed. Fall back to full decode from 0.0s.
        let wordCount = Float(fullText.split(separator: " ").count)
        if audioDur > 3 && wordCount / audioDur < 0.5 && !fullText.isEmpty {
            Self.logger.notice(
                "Result too short (\(Int(wordCount)) words for \(audioDur, privacy: .public)s) -- full decode"
            )
            if let rescue = await transcribeResult(samples: samples) {
                let rescueText = TranscriptionSession.cleanText(rescue.text)
                if rescueText.count > fullText.count {
                    fullText = rescueText
                }
            }
        }
        let finalMs = Self.ms(ContinuousClock.now - finalStart)

        if fullText.isEmpty && audioDur > 1 {
            Self.logger.notice("Empty result on \(audioDur, privacy: .public)s audio -- full decode rescue")
            if let rescue = await transcribeResult(samples: samples) {
                fullText = TranscriptionSession.cleanText(rescue.text)
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

        #if DEBUG
            // Fire-and-forget ground truth comparison. Runs a clean batch
            // decode in the background and logs the similarity. Zero latency
            // impact — the result is already returned to the user.
            let debugSamples = samples
            let debugResult = corrected
            let debugPipe = self.pipe
            Task.detached {
                await Self.compareGroundTruth(
                    streamingResult: debugResult,
                    samples: debugSamples,
                    audioDuration: audioDur,
                    pipe: debugPipe
                )
            }
        #endif

        return corrected
    }

    /// Cancel any in-progress session and release resources.
    public func cancel() {
        // Stop the scheduler so its loop exits. The schedulerTask
        // is saved as previousSchedulerTask so prepare() can await
        // it before starting a new session.
        Task { await scheduler.stop() }
        if let old = schedulerTask {
            previousSchedulerTask = old
        }
        schedulerTask = nil
        sessionLock.withLock { $0 = nil }
        session = TranscriptionSession()
        lastExtensionWasZeroProgress = false
    }

    // MARK: - Filter Pipeline

    private func runFilterPipeline(
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

    // MARK: - Result Assembly

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
                if allWords.count != filtered.count {
                    Self.logger.notice(
                        "Final decode: \(allWords.count) words, \(allWords.count - filtered.count) filtered by frontier"
                    )
                }
            } else {
                tailText = tail.text
                Self.logger.notice("Final decode: \(allWords.count) words all before frontier, using segment text")
            }
        } else {
            tailText = tail.text
            Self.logger.notice(
                "Final tail (no words): \"\(tailText.prefix(80), privacy: .public)\""
            )
        }

        let combined = TranscriptionSession.cleanText(transcript + tailText)
        let combinedSnippet = String(combined.prefix(80))
        let cc = transcript.count
        let tc = tailText.count
        Self.logger.notice(
            "Final combined: confirmed=\(cc) chars + tail=\(tc) chars = \"\(combinedSnippet, privacy: .public)\""
        )
        return combined
    }

    // MARK: - Transcription

    private func transcribeResult(
        samples: [Float],
        clipStart: Float = 0,
        prefixTokens: [Int] = [],
        chunked: Bool = false
    ) async -> TranscriptionResult? {
        do {
            let audioDuration = Float(samples.count) / Float(WhisperKit.sampleRate)
            var options = DecodingOptions()
            options.language = "en"
            options.skipSpecialTokens = true
            options.wordTimestamps = true
            options.chunkingStrategy = chunked ? .vad : ChunkingStrategy.none
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

    // MARK: - Debug Ground Truth

    #if DEBUG
        private static let qualityLogger = Log.logger(for: "TranscriptionQuality")

        /// Run a clean batch decode and compare against the streaming result.
        /// Called in a detached task — no latency impact on the user.
        private static func compareGroundTruth(
            streamingResult: String,
            samples: [Float],
            audioDuration: Float,
            pipe: WhisperKit
        ) async {
            do {
                var options = DecodingOptions()
                options.language = "en"
                options.skipSpecialTokens = true
                options.wordTimestamps = false
                options.chunkingStrategy = .vad

                let results = try await pipe.transcribe(audioArray: samples, decodeOptions: options)
                guard let groundTruth = results.first else {
                    qualityLogger.notice("Ground truth: decode returned nil")
                    return
                }

                let gtText = TranscriptionSession.cleanText(groundTruth.text)
                guard !gtText.isEmpty else {
                    qualityLogger.notice("Ground truth: empty after cleaning")
                    return
                }

                let similarity = levenshteinSimilarity(streamingResult, gtText)
                let pct = Int(similarity * 100)

                if similarity >= 0.85 {
                    qualityLogger.info(
                        "✅ Quality: \(pct)% match (\(audioDuration, privacy: .public)s audio)"
                    )
                } else {
                    qualityLogger.warning(
                        "⚠️ Quality: \(pct)% match (\(audioDuration, privacy: .public)s audio)"
                    )
                    qualityLogger.warning(
                        "  Streaming: \"\(streamingResult.prefix(100), privacy: .public)\""
                    )
                    qualityLogger.warning(
                        "  Ground truth: \"\(gtText.prefix(100), privacy: .public)\""
                    )

                    // Auto-capture fixture and alert
                    autoCapture(
                        similarity: pct,
                        groundTruth: gtText,
                        audioDuration: audioDuration
                    )
                    // Play a subtle alert so the user knows a bad transcription was caught
                    await MainActor.run {
                        NSSound(named: "Submarine")?.play()
                    }
                }
            } catch {
                qualityLogger.notice("Ground truth decode failed: \(error)")
            }
        }

        /// Auto-capture a test fixture when quality drops below threshold.
        /// Finds the most recent audio trace and copies it + ground truth
        /// to the test-fixtures directory.
        private static func autoCapture(
            similarity: Int,
            groundTruth: String,
            audioDuration: Float
        ) {
            let fm = FileManager.default
            let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            let traceDir = appSupport.appendingPathComponent("Operator/audio-traces")
            let fixtureDir = appSupport.appendingPathComponent("Operator/test-fixtures")

            // Find the most recent audio trace
            guard
                let traces = try? fm.contentsOfDirectory(
                    at: traceDir,
                    includingPropertiesForKeys: [.contentModificationDateKey],
                    options: .skipsHiddenFiles
                )
            else {
                qualityLogger.warning("Auto-capture: no audio traces directory")
                return
            }

            let wavFiles = traces.filter { $0.pathExtension == "wav" }
                .sorted { lhs, rhs in
                    let d1 =
                        (try? lhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
                        ?? .distantPast
                    let d2 =
                        (try? rhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
                        ?? .distantPast
                    return d1 > d2
                }

            guard let latestTrace = wavFiles.first else {
                qualityLogger.warning("Auto-capture: no WAV files in traces")
                return
            }

            let traceName = latestTrace.deletingPathExtension().lastPathComponent
            let fixtureName = "auto-\(similarity)pct-\(traceName)"
            let fixtureWav = fixtureDir.appendingPathComponent("\(fixtureName).wav")
            let fixtureExpected = fixtureDir.appendingPathComponent("\(fixtureName).expected.txt")

            // Skip if already captured
            guard !fm.fileExists(atPath: fixtureWav.path) else {
                qualityLogger.info("Auto-capture: fixture already exists: \(fixtureName)")
                return
            }

            do {
                try fm.createDirectory(at: fixtureDir, withIntermediateDirectories: true)
                try fm.copyItem(at: latestTrace, to: fixtureWav)
                try groundTruth.write(to: fixtureExpected, atomically: true, encoding: .utf8)
                qualityLogger.warning(
                    "📁 Auto-captured fixture: \(fixtureName, privacy: .public)"
                )
                qualityLogger.warning(
                    "  Run: make test-transcription"
                )
            } catch {
                qualityLogger.warning("Auto-capture failed: \(error)")
            }
        }

        /// Levenshtein similarity ratio (0.0 = completely different, 1.0 = identical).
        private static func levenshteinSimilarity(_ a: String, _ b: String) -> Double {
            let a = Array(a.lowercased())
            let b = Array(b.lowercased())
            let m = a.count
            let n = b.count
            if m == 0 && n == 0 { return 1.0 }
            if m == 0 || n == 0 { return 0.0 }

            var prev = Array(0...n)
            var curr = [Int](repeating: 0, count: n + 1)

            for i in 1...m {
                curr[0] = i
                for j in 1...n {
                    let cost = a[i - 1] == b[j - 1] ? 0 : 1
                    curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
                }
                swap(&prev, &curr)
            }

            return 1.0 - Double(prev[n]) / Double(max(m, n))
        }
    #endif
}

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
    ///
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

    /// Create a fresh engine sharing the same WhisperKit pipeline.
    ///
    /// The new engine has a clean session, scheduler, and filter pipeline.
    /// Useful for tests that need isolation between fixtures without
    /// reloading the model.
    public func freshEngine() -> WhisperKitEngine {
        WhisperKitEngine(pipe: pipe)
    }

    /// Create a fresh engine sharing the same WhisperKit pipeline
    /// but with a custom scheduler.
    ///
    /// Used for manifest replay tests that control exactly when
    /// background transcriptions fire.
    public func freshEngine(scheduler: TranscriptionScheduler) -> WhisperKitEngine {
        WhisperKitEngine(pipe: pipe, scheduler: scheduler)
    }

    private static func ms(_ duration: Duration) -> Int {
        let (seconds, attoseconds) = duration.components
        return Int(seconds) * 1_000 + Int(attoseconds / 1_000_000_000_000_000)
    }

    // MARK: - SchedulableEngine

    /// Number of audio samples currently accumulated in the session.
    public var sampleCount: Int {
        sessionLock.withLock { $0?.audioSamples.count ?? 0 }
    }

    /// Transcribe the current working window of audio samples.
    public func transcribeWorkingWindow(maxSampleCount: Int?) async {
        let passStart = ContinuousClock.now

        let samples = sessionLock.withLock { state -> [Float] in
            let allSamples = state?.audioSamples ?? []
            return maxSampleCount.map { Array(allSamples.prefix($0)) } ?? allSamples
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
            let ftr = session.frontier
            let zp = lastExtensionWasZeroProgress
            Self.logger.notice(
                "Confirmed \(newWords) words to \(ftr, privacy: .public)s in \(passMs)ms (zeroProgress=\(zp))"
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
            var newState = SessionState(vocabulary: vocab)
            newState.audioSamples.reserveCapacity(480_000)
            state = newState
        }
        session = TranscriptionSession()
        lastExtensionWasZeroProgress = false

        // Create a fresh scheduler for this session. The previous
        // session's scheduler may have a pending fire-and-forget stop()
        // from cancel() — reusing it would race (stop sets stopped=true
        // after start sets it to false, killing the new loop).
        scheduler = BackgroundLoopScheduler()

        let taskToDrain = previousSchedulerTask
        previousSchedulerTask = nil
        let sched = scheduler
        schedulerTask = Task { [weak self] in
            await taskToDrain?.value
            guard let self
            else { return }
            await sched.start(engine: self)
        }
        Self.logger.notice("Session started")
    }

    /// Append an audio buffer from the microphone tap.
    public func append(_ buffer: AVAudioPCMBuffer) {
        nonisolated(unsafe) let buf = buffer
        sessionLock.withLock { state in
            guard var sess = state
            else { return }
            if sess.converter == nil {
                sess.converter = try? AudioFormatConverter(inputFormat: buf.format)
            }
            if let converter = sess.converter {
                _ = converter.appendConverted(buf, to: &sess.audioSamples)
            }
            state = sess
        }
    }

    /// Finish recording and return the transcribed text.
    ///
    /// Uses a parallel decode strategy: kicks off a pessimistic tail decode
    /// immediately while letting the in-flight background pass finish. The
    /// pessimistic decode starts from the current frontier. If the background
    /// pass advances the frontier, the pessimistic result is filtered to
    /// only include words past the new frontier.
    public func finishAndTranscribe() async -> String? {
        let stopTime = ContinuousClock.now

        // Snapshot state before stopping — the background pass may still
        // advance the frontier while the pessimistic decode runs.
        let snapshotFrontier = session.frontier
        let snapshotOverlapTokens = lastExtensionWasZeroProgress ? [Int]() : session.overlapTokens

        let (samples, vocabulary) = sessionLock.withLock { state in
            (state?.audioSamples ?? [Float](), state?.vocabulary ?? [String]())
        }

        guard !samples.isEmpty
        else {
            Self.logger.notice("No audio samples captured")
            await scheduler.stop()
            schedulerTask?.cancel()
            schedulerTask = nil
            sessionLock.withLock { $0 = nil }
            return nil
        }

        let audioDur = Float(samples.count) / Float(WhisperKit.sampleRate)

        // Start pessimistic tail decode immediately — don't wait for background
        let pessimisticTask = startPessimisticDecode(
            samples: samples,
            frontier: snapshotFrontier,
            overlapTokens: snapshotOverlapTokens,
            audioDur: audioDur
        )

        // Signal scheduler to stop, await background pass completion
        await scheduler.stop()
        let pendingTask = schedulerTask
        schedulerTask = nil
        pendingTask?.cancel()
        await pendingTask?.value
        let awaitDuration = ContinuousClock.now - stopTime

        sessionLock.withLock { $0 = nil }

        let awaitMs = Self.ms(awaitDuration)
        let ftr = session.frontier
        let wc = session.transcript.split(separator: " ").count
        let strat = String(describing: session.strategy)
        let stopMsg = "Stop: \(audioDur)s audio, frontier=\(ftr)s, await=\(awaitMs)ms"
        Self.logger.notice("\(stopMsg, privacy: .public)")
        Self.logger.notice("Stop: confirmed=\(wc) words, strategy=\(strat, privacy: .public)")

        // Assemble result from pessimistic decode + final session state
        let finalStart = ContinuousClock.now
        var fullText = await assemblePessimisticResult(
            pessimisticTask: pessimisticTask
        )

        // Safety net rescues
        fullText = await applySafetyNet(
            fullText,
            samples: samples,
            audioDur: audioDur
        )

        let finalMs = Self.ms(ContinuousClock.now - finalStart)

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
            // Only run ground truth comparison in the live app, not test runners.
            // Test runners trigger this code path and auto-capture garbage fixtures
            // because the "most recent audio trace" belongs to a different session.
            if Bundle.main.bundleIdentifier == "com.operator.app" {
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
            }
        #endif

        return corrected
    }

    // MARK: - Parallel Decode Helpers

    private func startPessimisticDecode(
        samples: [Float],
        frontier: Float,
        overlapTokens: [Int],
        audioDur: Float
    ) -> Task<TranscriptionResult?, Never> {
        Task { [pipe, filters] in
            let result = await self.transcribeResult(
                samples: samples,
                clipStart: frontier,
                prefixTokens: overlapTokens,
                chunked: true
            )
            guard let result
            else { return nil }

            return self.runFilterPipeline(result, samples: samples, frontier: frontier)
        }
    }

    private func assemblePessimisticResult(
        pessimisticTask: Task<TranscriptionResult?, Never>
    ) async -> String {
        let pessResult = await pessimisticTask.value
        let finalFrontier = session.frontier
        let finalTranscript = session.transcript

        if let result = pessResult {
            return TranscriptionSession.assembleWithFrontier(
                transcript: finalTranscript,
                tailWords: result.allWords,
                tailSegmentText: result.text,
                frontier: finalFrontier
            )
        }
        return TranscriptionSession.cleanText(finalTranscript)
    }

    private func applySafetyNet(
        _ text: String,
        samples: [Float],
        audioDur: Float
    ) async -> String {
        var fullText = text
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
        if fullText.isEmpty && audioDur > 1 {
            Self.logger.notice(
                "Empty result on \(audioDur, privacy: .public)s audio -- full decode rescue"
            )
            if let rescue = await transcribeResult(samples: samples) {
                fullText = TranscriptionSession.cleanText(rescue.text)
            }
        }
        return fullText
    }

    /// Cancel any in-progress session and release resources.
    public func cancel() {
        // Capture the current scheduler before replacing it.
        // The fire-and-forget stop must use a local copy to avoid
        // a data race with prepare() replacing self.scheduler.
        let oldScheduler = scheduler
        Task { await oldScheduler.stop() }
        previousSchedulerTask = schedulerTask
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
        ///
        /// Called in a detached task -- no latency impact on the user.
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
                    // Play a subtle alert — only in the live app, not test runners
                    if Bundle.main.bundleIdentifier == "com.operator.app" {
                        await MainActor.run {
                            _ = NSSound(named: "Submarine")?.play()
                        }
                    }
                }
            } catch {
                qualityLogger.notice("Ground truth decode failed: \(error)")
            }
        }

        /// Auto-capture a test fixture when quality drops below threshold.
        ///
        /// Finds the most recent audio trace and copies it + ground truth
        /// to the test-fixtures directory.
        private static func autoCapture(
            similarity: Int,
            groundTruth: String,
            audioDuration: Float
        ) {
            let fm = FileManager.default
            guard
                let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            else {
                qualityLogger.warning("Auto-capture: no application support directory")
                return
            }
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
        private static func levenshteinSimilarity(_ lhs: String, _ rhs: String) -> Double {
            let left = Array(lhs.lowercased())
            let right = Array(rhs.lowercased())
            let leftLen = left.count
            let rightLen = right.count
            if leftLen == 0 && rightLen == 0 {
                return 1.0
            }
            if leftLen == 0 || rightLen == 0 {
                return 0.0
            }

            var prev = Array(0...rightLen)
            var curr = [Int](repeating: 0, count: rightLen + 1)

            for li in 1...leftLen {
                curr[0] = li
                for ri in 1...rightLen {
                    let cost = left[li - 1] == right[ri - 1] ? 0 : 1
                    curr[ri] = min(prev[ri] + 1, curr[ri - 1] + 1, prev[ri - 1] + cost)
                }
                swap(&prev, &curr)
            }

            return 1.0 - Double(prev[rightLen]) / Double(max(leftLen, rightLen))
        }
    #endif
}

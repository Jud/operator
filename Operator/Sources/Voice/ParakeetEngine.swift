@preconcurrency import AVFoundation
import AudioCommon
import ParakeetASR
@preconcurrency import SpeechVAD
import os

private struct ParakeetQueuedSegment {
    let id: Int
    let samples: [Float]
}

private struct ParakeetFinalizePayload {
    let sessionID: Int
    let segments: [ParakeetQueuedSegment]
    let fallback: [Float]?
}

private struct ParakeetSessionState: @unchecked Sendable {
    var sessionID = 0
    var normalizedSamples: [Float] = []
    var inputSampleRate: Int?
    var vadModel: SileroVADModel?
    var vadProcessor: StreamingVADProcessor?
    var vadProcessedSampleCount = 0
    var activeSpeechStartSample: Int?
    var nextSegmentID = 0
}

// swiftlint:disable file_length type_contents_order type_body_length
/// Parakeet CoreML speech-to-text engine conforming to ``TranscriptionEngine``.
///
/// Audio is normalized to 16kHz mono as it arrives, segmented with Silero VAD,
/// and transcribed in the background segment-by-segment while the user is still
/// talking. `finishAndTranscribe()` only needs to flush the tail segment and join
/// already-produced transcript chunks, which keeps end-of-utterance latency low.
public final class ParakeetEngine: TranscriptionEngine, @unchecked Sendable {
    private static let logger = Log.logger(for: "ParakeetEngine")
    private static let targetSampleRate = 16_000
    private static let maxStreamingSegmentSeconds = 2.0
    private static let vadEngine: SileroVADEngine = .coreml

    private let modelManager: ModelManager
    private let sessionState = OSAllocatedUnfairLock<ParakeetSessionState>(initialState: .init())
    private let pendingSegmentTasks = OSAllocatedUnfairLock<[Task<Void, Never>]>(initialState: [])
    private let transcriber: ParakeetSegmentTranscriber

    /// Creates a Parakeet engine with the given model manager for lifecycle tracking.
    public init(modelManager: ModelManager) {
        self.modelManager = modelManager
        self.transcriber = ParakeetSegmentTranscriber(modelManager: modelManager)
        registerDownloadHandler()
    }

    /// Prepare for a new transcription session by resetting streaming state.
    public func prepare() throws {
        let pendingTasks = pendingSegmentTasks.withLock { tasks in
            let copy = tasks
            tasks.removeAll(keepingCapacity: false)
            return copy
        }
        for task in pendingTasks {
            task.cancel()
        }

        let sessionID = sessionState.withLock { state in
            state.sessionID += 1
            state.normalizedSamples.removeAll(keepingCapacity: true)
            state.inputSampleRate = nil
            state.vadProcessedSampleCount = 0
            state.activeSpeechStartSample = nil
            state.nextSegmentID = 0
            if let vadModel = state.vadModel {
                vadModel.resetState()
                state.vadProcessor = StreamingVADProcessor(model: vadModel)
            } else {
                state.vadProcessor = nil
            }
            return state.sessionID
        }

        Task {
            await self.transcriber.resetSession(sessionID)
        }
        startBackgroundWarmup(for: sessionID)
    }

    /// Append an audio buffer captured from the microphone.
    public func append(_ buffer: AVAudioPCMBuffer) {
        let inputSampleRate = Int(buffer.format.sampleRate)
        let normalizedSamples = Self.normalize(buffer, sourceSampleRate: inputSampleRate)
        guard !normalizedSamples.isEmpty else {
            return
        }

        let segments = sessionState.withLock { state in
            if state.inputSampleRate == nil {
                state.inputSampleRate = inputSampleRate
            }
            state.normalizedSamples.append(contentsOf: normalizedSamples)
            return Self.processSamplesIfPossible(normalizedSamples, state: &state)
        }
        submitSegments(segments)
    }

    /// Signal end of audio and return the transcribed text.
    public func finishAndTranscribe() async -> String? {
        let finalizePayload = sessionState.withLock { state -> ParakeetFinalizePayload in
            let flushEvents = state.vadProcessor?.flush() ?? []
            let flushedSegments = Self.makeSegments(from: flushEvents, state: &state)
            let fallback = state.normalizedSamples.isEmpty ? nil : state.normalizedSamples
            return ParakeetFinalizePayload(
                sessionID: state.sessionID,
                segments: flushedSegments,
                fallback: fallback
            )
        }

        for segment in finalizePayload.segments {
            await transcriber.enqueueSegment(
                sessionID: finalizePayload.sessionID,
                id: segment.id,
                samples: segment.samples
            )
        }

        let pendingTasks = pendingSegmentTasks.withLock { tasks in
            let copy = tasks
            tasks.removeAll(keepingCapacity: false)
            return copy
        }
        for task in pendingTasks {
            await task.value
        }

        let text = await transcriber.finishSession(
            sessionID: finalizePayload.sessionID,
            fallbackSamples: finalizePayload.fallback
        )
        if let text {
            Self.logger.info("Parakeet transcription complete (\(text.count) chars)")
        } else {
            Self.logger.warning("Parakeet transcription produced no text")
        }
        return text
    }

    /// Cancel any in-progress transcription and release per-session state.
    public func cancel() {
        let sessionID = sessionState.withLock { state in
            state.sessionID += 1
            state.normalizedSamples.removeAll(keepingCapacity: false)
            state.inputSampleRate = nil
            state.vadProcessedSampleCount = 0
            state.activeSpeechStartSample = nil
            state.nextSegmentID = 0
            if let vadModel = state.vadModel {
                vadModel.resetState()
            }
            state.vadProcessor = nil
            return state.sessionID
        }

        let pendingTasks = pendingSegmentTasks.withLock { tasks in
            let copy = tasks
            tasks.removeAll(keepingCapacity: false)
            return copy
        }
        for task in pendingTasks {
            task.cancel()
        }

        Task {
            await self.transcriber.resetSession(sessionID)
        }
    }

    /// Unload STT models to free memory.
    public func unloadModel() {
        sessionState.withLock { state in
            state.vadModel?.unload()
            state.vadModel = nil
            state.vadProcessor = nil
            state.vadProcessedSampleCount = 0
            state.activeSpeechStartSample = nil
        }
        Task {
            await self.transcriber.unloadModel()
        }
    }

    private func startBackgroundWarmup(for sessionID: Int) {
        Task { [weak self] in
            guard let self else {
                return
            }
            await self.transcriber.prewarm()
        }

        Task { [weak self] in
            guard let self else {
                return
            }
            await self.ensureVADReady(for: sessionID)
        }
    }

    private func ensureVADReady(for sessionID: Int) async {
        if let cachedVAD = sessionState.withLock({ $0.vadModel }) {
            installVADModel(cachedVAD, for: sessionID)
            return
        }

        do {
            let model = try await SileroVADModel.fromPretrained(engine: Self.vadEngine)
            installVADModel(model, for: sessionID)
        } catch {
            Self.logger.error("Silero VAD load failed: \(error.localizedDescription)")
        }
    }

    private func installVADModel(_ vadModel: SileroVADModel, for sessionID: Int) {
        let backlog = sessionState.withLock { state -> [Float]? in
            state.vadModel = vadModel
            guard state.sessionID == sessionID else {
                return nil
            }
            vadModel.resetState()
            state.vadProcessor = StreamingVADProcessor(model: vadModel)
            return Array(state.normalizedSamples.dropFirst(state.vadProcessedSampleCount))
        }

        guard let backlog, !backlog.isEmpty else {
            return
        }

        let segments = sessionState.withLock { state in
            Self.processSamplesIfPossible(backlog, state: &state)
        }
        submitSegments(segments)
    }

    private func submitSegments(_ segments: [ParakeetQueuedSegment]) {
        guard !segments.isEmpty else {
            return
        }
        for segment in segments {
            let sessionID = sessionState.withLock { $0.sessionID }
            let task = Task {
                await self.transcriber.enqueueSegment(
                    sessionID: sessionID,
                    id: segment.id,
                    samples: segment.samples
                )
            }
            pendingSegmentTasks.withLock { $0.append(task) }
        }
    }

    private func registerDownloadHandler() {
        let manager = modelManager
        // swiftlint:disable:next unhandled_throwing_task
        Task {
            await manager.registerDownloadHandler(for: .stt) { _, progressCallback in
                await progressCallback(0.0)

                let parakeetCacheDir = try HuggingFaceDownloader.getCacheDirectory(
                    for: ParakeetASRModel.defaultModelId
                )
                try await HuggingFaceDownloader.downloadWeights(
                    modelId: ParakeetASRModel.defaultModelId,
                    to: parakeetCacheDir,
                    additionalFiles: [
                        "encoder.mlmodelc/**",
                        "decoder.mlmodelc/**",
                        "joint.mlmodelc/**",
                        "vocab.json",
                        "config.json"
                    ]
                )
                await progressCallback(0.85)

                let vadCacheDir = try HuggingFaceDownloader.getCacheDirectory(
                    for: SileroVADModel.defaultCoreMLModelId
                )
                try await HuggingFaceDownloader.downloadWeights(
                    modelId: SileroVADModel.defaultCoreMLModelId,
                    to: vadCacheDir,
                    additionalFiles: ["silero_vad.mlmodelc/**", "config.json"]
                )
                await progressCallback(1.0)

                return parakeetCacheDir
            }
        }
    }

    private static func processSamplesIfPossible(
        _ samples: [Float],
        state: inout ParakeetSessionState
    ) -> [ParakeetQueuedSegment] {
        guard let vadProcessor = state.vadProcessor else {
            return []
        }
        state.vadProcessedSampleCount += samples.count
        let events = vadProcessor.process(samples: samples)
        var segments = makeSegments(from: events, state: &state)
        segments.append(contentsOf: makeForcedSegmentsIfNeeded(state: &state))
        return segments
    }

    private static func makeSegments(
        from events: [VADEvent],
        state: inout ParakeetSessionState
    ) -> [ParakeetQueuedSegment] {
        var segments: [ParakeetQueuedSegment] = []

        for event in events {
            switch event {
            case .speechStarted(let time):
                state.activeSpeechStartSample =
                    state.activeSpeechStartSample
                    ?? sampleIndex(forSeconds: time)

            case .speechEnded(let segment):
                let startIndex =
                    state.activeSpeechStartSample
                    ?? sampleIndex(forSeconds: segment.startTime)
                let endIndex = min(
                    state.normalizedSamples.count,
                    sampleIndex(forSeconds: segment.endTime)
                )
                if let queuedSegment = makeQueuedSegment(
                    startIndex: startIndex,
                    endIndex: endIndex,
                    state: &state
                ) {
                    segments.append(queuedSegment)
                }
                state.activeSpeechStartSample = nil
            }
        }

        return segments
    }

    private static func makeForcedSegmentsIfNeeded(
        state: inout ParakeetSessionState
    ) -> [ParakeetQueuedSegment] {
        guard let startIndex = state.activeSpeechStartSample else {
            return []
        }

        let maxSegmentSamples = Int(Double(Self.targetSampleRate) * Self.maxStreamingSegmentSeconds)
        let currentEndIndex = state.normalizedSamples.count
        guard currentEndIndex - startIndex >= maxSegmentSamples else {
            return []
        }

        guard
            let queuedSegment = makeQueuedSegment(
                startIndex: startIndex,
                endIndex: currentEndIndex,
                state: &state
            )
        else {
            return []
        }

        state.activeSpeechStartSample = currentEndIndex
        return [queuedSegment]
    }

    private static func makeQueuedSegment(
        startIndex: Int,
        endIndex: Int,
        state: inout ParakeetSessionState
    ) -> ParakeetQueuedSegment? {
        let boundedStartIndex = max(0, startIndex)
        let boundedEndIndex = min(state.normalizedSamples.count, endIndex)
        guard boundedEndIndex > boundedStartIndex else {
            return nil
        }

        let segmentSamples = Array(state.normalizedSamples[boundedStartIndex..<boundedEndIndex])
        let queuedSegment = ParakeetQueuedSegment(
            id: state.nextSegmentID,
            samples: segmentSamples
        )
        state.nextSegmentID += 1
        return queuedSegment
    }

    private static func sampleIndex(forSeconds time: Float) -> Int {
        max(0, Int(time * Float(Self.targetSampleRate)))
    }

    private static func normalize(
        _ buffer: AVAudioPCMBuffer,
        sourceSampleRate: Int
    ) -> [Float] {
        guard let channelData = buffer.floatChannelData else {
            return []
        }

        let frameCount = Int(buffer.frameLength)
        let channelCount = Int(buffer.format.channelCount)
        guard frameCount > 0, channelCount > 0 else {
            return []
        }

        let monoSamples: [Float]
        if channelCount == 1 {
            monoSamples = Array(UnsafeBufferPointer(start: channelData[0], count: frameCount))
        } else {
            monoSamples = (0..<frameCount).map { frame in
                var sum: Float = 0
                for channel in 0..<channelCount {
                    sum += channelData[channel][frame]
                }
                return sum / Float(channelCount)
            }
        }

        if sourceSampleRate == Self.targetSampleRate {
            return monoSamples
        }
        return AudioFileLoader.resample(
            monoSamples,
            from: sourceSampleRate,
            to: Self.targetSampleRate
        )
    }
}

// swiftlint:enable file_length type_contents_order type_body_length

@preconcurrency import AVFoundation
import AudioCommon
import ParakeetASR
import os

/// Parakeet CoreML speech-to-text engine conforming to ``TranscriptionEngine``.
///
/// Uses speech-swift's `ParakeetASRModel` (Parakeet TDT 0.6B v3) for on-device
/// transcription via the Apple Neural Engine. Audio samples are accumulated during
/// push-to-talk and processed in a single batch on `finishAndTranscribe()`.
///
/// Memory management: the CoreML model (~400MB) is loaded on first use and
/// unloaded after each transcription cycle completes, keeping peak memory low when
/// TTS and routing models are resident.
///
/// Thread safety: `append(_:)` is called from the audio tap thread. Sample
/// accumulation uses `OSAllocatedUnfairLock`. All other methods are called from
/// `@MainActor` via `SpeechTranscriber`.
public final class ParakeetEngine: TranscriptionEngine, @unchecked Sendable {
    private static let logger = Log.logger(for: "ParakeetEngine")

    private let modelManager: ModelManager

    /// Accumulated Float32 PCM samples, extracted from audio buffers on the tap thread.
    private let audioSamples = OSAllocatedUnfairLock<[Float]>(initialState: [])

    /// Sample rate captured from the first audio buffer in each session.
    private let inputSampleRate = OSAllocatedUnfairLock<Int?>(initialState: nil)

    /// The loaded CoreML model, non-nil only during the transcription window.
    private var model: ParakeetASRModel?

    // MARK: - Initialization

    /// Creates a Parakeet engine with the given model manager for lifecycle tracking.
    ///
    /// Registers a download handler with ModelManager so the Settings UI can trigger
    /// model downloads. No model loading occurs at construction time.
    ///
    /// - Parameter modelManager: The shared model manager for state tracking.
    public init(modelManager: ModelManager) {
        self.modelManager = modelManager
        registerDownloadHandler()
    }

    // MARK: - TranscriptionEngine

    /// Prepare for a new transcription session by clearing the sample accumulator.
    ///
    /// Cancels any previous in-flight session and resets accumulated audio data.
    /// Model loading is deferred to `finishAndTranscribe()` to avoid Sendable
    /// constraints with the non-Sendable `ParakeetASRModel` type.
    public func prepare() throws {
        cancel()
    }

    /// Append an audio buffer captured from the microphone.
    ///
    /// Extracts Float32 PCM samples from the buffer and appends them to the
    /// accumulator. Multi-channel audio is mixed down to mono. Called on the
    /// audio tap thread; thread-safe via `OSAllocatedUnfairLock`.
    public func append(_ buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData else {
            return
        }
        let frameCount = Int(buffer.frameLength)
        let channelCount = Int(buffer.format.channelCount)

        inputSampleRate.withLock { value in
            if value == nil {
                value = Int(buffer.format.sampleRate)
            }
        }

        if channelCount == 1 {
            let samples = Array(UnsafeBufferPointer(start: channelData[0], count: frameCount))
            audioSamples.withLock { $0.append(contentsOf: samples) }
        } else {
            var mixed: [Float] = []
            mixed.reserveCapacity(frameCount)
            for frame in 0..<frameCount {
                var sum: Float = 0
                for ch in 0..<channelCount {
                    sum += channelData[ch][frame]
                }
                mixed.append(sum / Float(channelCount))
            }
            let monoSamples = mixed
            audioSamples.withLock { $0.append(contentsOf: monoSamples) }
        }
    }

    /// Signal end of audio and return the transcribed text.
    ///
    /// Loads the model if not already in memory, runs CoreML inference on the
    /// accumulated audio samples. The model stays resident after transcription
    /// to avoid the ~2s warmup penalty on each call; use ``unloadModel()``
    /// explicitly when memory pressure requires it.
    public func finishAndTranscribe() async -> String? {
        let samples = audioSamples.withLock { buf in
            let copy = buf
            buf.removeAll()
            return copy
        }

        guard !samples.isEmpty else {
            Self.logger.warning("No audio samples to transcribe")
            return nil
        }

        do {
            try await ensureModelLoaded()
        } catch {
            Self.logger.error("Failed to load Parakeet model: \(error.localizedDescription)")
            await modelManager.markError(.stt, message: error.localizedDescription)
            return nil
        }

        guard let currentModel = model else {
            Self.logger.error("Parakeet model is nil after loading")
            return nil
        }

        let sampleRate = inputSampleRate.withLock { $0 ?? 16_000 }

        let text: String
        do {
            text = try currentModel.transcribeAudio(samples, sampleRate: sampleRate)
            Self.logger.info("Parakeet transcription complete (\(text.count) chars)")
        } catch {
            Self.logger.error("Parakeet transcription failed: \(error.localizedDescription)")
            unloadModel()
            return nil
        }

        return text.isEmpty ? nil : text
    }

    /// Cancel any in-progress transcription and release resources.
    public func cancel() {
        audioSamples.withLock { $0.removeAll() }
        inputSampleRate.withLock { $0 = nil }
    }

    // MARK: - Private

    /// Register the download handler with ModelManager for Settings UI downloads.
    private func registerDownloadHandler() {
        let manager = modelManager
        // swiftlint:disable:next unhandled_throwing_task
        Task {
            await manager.registerDownloadHandler(for: .stt) { _, progressCallback in
                await progressCallback(0.0)
                let cacheDir = try HuggingFaceDownloader.getCacheDirectory(
                    for: ParakeetASRModel.defaultModelId
                )
                try await HuggingFaceDownloader.downloadWeights(
                    modelId: ParakeetASRModel.defaultModelId,
                    to: cacheDir,
                    additionalFiles: [
                        "encoder.mlmodelc/**",
                        "decoder.mlmodelc/**",
                        "joint.mlmodelc/**",
                        "vocab.json",
                        "config.json"
                    ]
                )
                await progressCallback(1.0)
                return cacheDir
            }
        }
    }

    /// Load the Parakeet model if not already in memory.
    ///
    /// Uses `ParakeetASRModel.fromPretrained()` which handles downloading from
    /// HuggingFace Hub (if not cached) and loading CoreML models. The first
    /// transcription triggers CoreML graph compilation (~2s one-time cost);
    /// subsequent transcriptions complete in ~30ms.
    private func ensureModelLoaded() async throws {
        guard model == nil else {
            return
        }

        await modelManager.markLoading(.stt)
        Self.logger.info("Loading Parakeet model...")

        let loaded = try await ParakeetASRModel.fromPretrained()
        model = loaded

        await modelManager.markLoaded(.stt)
        Self.logger.info("Parakeet model loaded")
    }

    /// Unload the CoreML model to free memory and update ModelManager state.
    ///
    /// Call this when memory pressure is high and the STT model is not actively
    /// needed. The model will be reloaded on the next transcription.
    public func unloadModel() {
        model?.unload()
        model = nil
        let manager = modelManager
        Task {
            await manager.markUnloaded(.stt)
        }
        Self.logger.info("Parakeet model unloaded")
    }
}

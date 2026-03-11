// swiftlint:disable file_length
@preconcurrency import AVFoundation
import AudioCommon
import Qwen3TTS

/// Serializes access to the non-thread-safe Qwen3TTSModel to prevent
/// concurrent MLX CompilerCache access (causes EXC_BAD_ACCESS).
private actor TTSModelSerializer {
    private let model: Qwen3TTSModel
    private var didWarmUp = false

    init(model: sending Qwen3TTSModel) {
        self.model = model
    }

    func synthesizeStream(
        text: String,
        speaker: String,
        streaming: StreamingConfig,
        instruct: String? = nil
    ) -> AsyncThrowingStream<AudioChunk, Error> {
        model.synthesizeStream(
            text: text,
            speaker: speaker,
            instruct: instruct,
            streaming: streaming
        )
    }

    func warmUpIfNeeded() {
        guard !didWarmUp else {
            return
        }
        model.warmUp()
        didWarmUp = true
    }
}

/// Qwen3-TTS speech synthesis manager with streaming AVAudioEngine playback.
///
/// Uses the Qwen3-TTS model from speech-swift with ``AVAudioPlayerNode`` for
/// low-latency streaming PCM playback. Interruption tracking maps player node
/// render time to character offsets for heard/unheard text splits (~2 word accuracy).
/// The model (~2GB) loads lazily on first speak and remains resident.
@MainActor
public final class Qwen3TTSSpeechManager: SpeechManaging {
    private static let logger = Log.logger(for: "Qwen3TTSSpeechManager")

    /// Qwen3-TTS output sample rate.
    private static let sampleRate: Double = 24_000

    // MARK: - Dependencies

    private let modelManager: ModelManager

    // MARK: - Model State

    private var serializer: TTSModelSerializer?
    private var isModelLoading = false

    // MARK: - Audio Engine

    private let audioEngine = AVAudioEngine()
    private let playerNode = AVAudioPlayerNode()
    private let timePitchNode = AVAudioUnitTimePitch()
    private let outputFormat: AVAudioFormat

    // MARK: - Playback Tracking

    private var currentText: String = ""
    private var currentSession: String = ""
    private var totalScheduledSamples: Int64 = 0
    private var generationTask: Task<Void, Never>?
    private var speaking = false

    // MARK: - SpeechManaging

    /// Whether speech is currently being synthesized or played back.
    public var isSpeaking: Bool { speaking }

    /// Whether a previous model load attempt failed.
    ///
    /// Set to `true`` when ``ensureModelLoaded()`` throws. Checked by
    /// ``AdaptiveSpeechManager`` to proactively fall back to Apple TTS
    /// before attempting to speak, preventing silent message loss.
    public private(set) var modelLoadFailed = false

    /// Playback speed multiplier applied via AVAudioUnitTimePitch.
    ///
    /// 1.0 = normal speed, 1.3 = 30% faster, 0.8 = 20% slower.
    /// Pitch is preserved regardless of rate. Takes effect immediately,
    /// including on currently playing audio.
    public var speechRate: Float = 1.0 {
        didSet { timePitchNode.rate = speechRate }
    }

    /// Natural language instruction for Qwen3-TTS voice style control.
    ///
    /// Influences pacing, tone, and expressiveness at the model level.
    /// Examples: "Speak naturally.", "Speak at a brisk, clear pace.",
    /// "Speak slowly and calmly."
    public var voiceInstruct: String = "Speak naturally."

    /// Stream that yields each time an utterance finishes playing (not interrupted).
    public let finishedSpeaking: AsyncStream<Void>
    private let finishedContinuation: AsyncStream<Void>.Continuation

    // MARK: - Initialization

    /// Creates a Qwen3-TTS speech manager with the given model manager.
    ///
    /// Registers a download handler with ModelManager for Settings UI model downloads.
    /// No model loading occurs at construction time.
    ///
    /// - Parameter modelManager: The shared model manager for lifecycle tracking.
    public init(modelManager: ModelManager) {
        let (stream, continuation) = AsyncStream<Void>.makeStream()
        self.finishedSpeaking = stream
        self.finishedContinuation = continuation
        self.modelManager = modelManager

        guard
            let format = AVAudioFormat(
                standardFormatWithSampleRate: Self.sampleRate,
                channels: 1
            )
        else {
            fatalError("Failed to create 24kHz mono audio format")
        }
        self.outputFormat = format

        setupAudioEngine()
        audioEngine.prepare()
        registerDownloadHandler()
    }

    /// Preload and warm the TTS model without speaking an utterance.
    public func prewarm() async {
        do {
            try await ensureModelLoaded()
            guard let serializer else {
                return
            }
            await serializer.warmUpIfNeeded()
            Self.logger.info("Qwen3-TTS warmup complete")
        } catch {
            Self.logger.error("Qwen3-TTS warmup failed: \(error.localizedDescription)")
            modelLoadFailed = true
        }
    }

    // MARK: - SpeechManaging Methods

    /// Speak text with an agent name prefix using the Qwen3-TTS voice.
    ///
    /// Extracts `qwenSpeakerID` from the voice descriptor for per-agent voice
    /// differentiation. Begins streaming TTS generation and playback in a
    /// background task. If the model is not yet loaded, it loads lazily.
    ///
    /// If speech is already in progress, the current utterance is stopped and
    /// replaced by the new one (latest-utterance-wins).
    public func speak(_ text: String, voice: VoiceDescriptor, prefix: String, pitchMultiplier: Float) {
        stopPlayback()

        let fullText = "\(prefix): \(text)"
        currentText = fullText
        currentSession = prefix
        totalScheduledSamples = 0

        Self.logger.info("Speaking: \"\(prefix): \(text.prefix(60))...\" (speaker: \(voice.qwenSpeakerID))")

        let instruct = voiceInstruct
        generationTask = Task { [weak self] in
            await self?.performSpeech(fullText: fullText, speakerID: voice.qwenSpeakerID, instruct: instruct)
        }
    }

    /// Immediately stop speech and return what was heard vs. what remains.
    ///
    /// Calculates the heard/unheard text split from the player node's current
    /// render time position mapped to a character offset. The character position
    /// is snapped to the nearest word boundary for natural splits.
    public func interrupt() -> InterruptInfo {
        let currentSample = currentPlaybackSample()
        let text = currentText
        let session = currentSession
        let total = totalScheduledSamples

        stopPlayback()

        guard total > 0 && !text.isEmpty else {
            return InterruptInfo(heardText: "", unheardText: text, session: session)
        }

        let charPos = characterPosition(
            atSample: currentSample,
            totalSamples: total,
            textLength: text.count,
            text: text
        )

        let heard = String(text.prefix(charPos))
        let unheard = String(text.dropFirst(charPos))

        Self.logger.info(
            "Interrupted at sample \(currentSample)/\(total) -> char \(charPos)/\(text.count) for \(session)"
        )

        return InterruptInfo(heardText: heard, unheardText: unheard, session: session)
    }

    /// Stop any current speech without tracking interruption state.
    public func stop() {
        stopPlayback()
    }

    // MARK: - Audio Engine Setup

    private func setupAudioEngine() {
        audioEngine.attach(playerNode)
        audioEngine.attach(timePitchNode)
        timePitchNode.pitch = 0
        timePitchNode.rate = speechRate
        audioEngine.connect(playerNode, to: timePitchNode, format: outputFormat)
        audioEngine.connect(timePitchNode, to: audioEngine.mainMixerNode, format: outputFormat)
    }

    // MARK: - Speech Generation

    /// Run the full TTS pipeline: load model, start audio engine, stream chunks, schedule completion.
    private func performSpeech(fullText: String, speakerID: String, instruct: String) async {
        do {
            try await ensureModelLoaded()
        } catch {
            Self.logger.error("Failed to load Qwen3-TTS model: \(error.localizedDescription)")
            modelLoadFailed = true
            speaking = false
            finishedContinuation.yield()
            return
        }

        guard let serializer else {
            Self.logger.error("Qwen3-TTS model is nil after loading")
            modelLoadFailed = true
            speaking = false
            finishedContinuation.yield()
            return
        }

        guard startAudioPlayback() else {
            return
        }

        await streamAndScheduleBuffers(serializer: serializer, text: fullText, speakerID: speakerID, instruct: instruct)

        guard !Task.isCancelled else {
            return
        }

        schedulePlaybackCompletion()
    }

    /// Start the audio engine and player node for playback.
    ///
    /// - Returns: `true` if the engine started successfully, `false` on failure.
    private func startAudioPlayback() -> Bool {
        do {
            if !audioEngine.isRunning {
                try audioEngine.start()
            }
            playerNode.play()
            return true
        } catch {
            Self.logger.error("Failed to start audio engine: \(error.localizedDescription)")
            speaking = false
            return false
        }
    }

    /// Stream TTS chunks from the model and schedule each as an AVAudioPCMBuffer.
    private func streamAndScheduleBuffers(
        serializer: TTSModelSerializer,
        text: String,
        speakerID: String,
        instruct: String
    ) async {
        let stream = await serializer.synthesizeStream(
            text: text,
            speaker: speakerID,
            streaming: .lowLatency,
            instruct: instruct
        )

        do {
            for try await chunk in stream {
                guard !Task.isCancelled else {
                    break
                }
                guard !chunk.samples.isEmpty else {
                    continue
                }

                let buffer = createBuffer(from: chunk.samples)
                totalScheduledSamples += Int64(buffer.frameLength)
                playerNode.scheduleBuffer(buffer, completionCallbackType: .dataConsumed) { _ in }

                if !speaking {
                    speaking = true
                }
            }
        } catch {
            if !Task.isCancelled {
                Self.logger.error("TTS streaming error: \(error.localizedDescription)")
            }
        }
    }

    // MARK: - Playback Helpers

    /// Stop all playback, cancel generation, and reset tracking state.
    private func stopPlayback() {
        generationTask?.cancel()
        generationTask = nil
        playerNode.stop()
        speaking = false
        currentText = ""
        currentSession = ""
        totalScheduledSamples = 0
    }

    /// Schedule a sentinel buffer whose completion signals that all audio has played back.
    private func schedulePlaybackCompletion() {
        guard let sentinel = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: 1) else {
            Self.logger.warning("Failed to create sentinel buffer")
            handlePlaybackComplete()
            return
        }
        sentinel.frameLength = 1

        playerNode.scheduleBuffer(
            sentinel,
            completionCallbackType: .dataPlayedBack
        ) { [weak self] _ in
            Task { @MainActor in
                self?.handlePlaybackComplete()
            }
        }
    }

    /// Called when playback completes naturally (not interrupted).
    private func handlePlaybackComplete() {
        guard speaking else {
            return
        }
        Self.logger.debug("Finished speaking: \"\(self.currentSession)\"")
        speaking = false
        currentText = ""
        currentSession = ""
        totalScheduledSamples = 0
        finishedContinuation.yield()
    }
}

// MARK: - Model Lifecycle

extension Qwen3TTSSpeechManager {
    /// Load the Qwen3-TTS model if not already in memory.
    ///
    /// Uses `Qwen3TTSModel.fromPretrained()` which handles downloading from
    /// HuggingFace Hub (if not cached) and loading model weights. The model
    /// remains resident after loading for the application session.
    private func ensureModelLoaded() async throws {
        guard serializer == nil else {
            return
        }
        guard !isModelLoading else {
            while isModelLoading {
                try await Task.sleep(nanoseconds: 50_000_000)
            }
            return
        }

        isModelLoading = true
        await modelManager.markLoading(.tts)
        Self.logger.info("Loading Qwen3-TTS model...")

        do {
            // Inner scope limits `loaded` lifetime so it's consumed by the
            // actor init before any subsequent suspension points.
            do {
                let loaded = try await Qwen3TTSModel.fromPretrained(
                    modelId: TTSModelVariant.customVoice.rawValue
                )
                serializer = TTSModelSerializer(model: loaded)
            }
            isModelLoading = false
            await modelManager.markLoaded(.tts)
            Self.logger.info("Qwen3-TTS model loaded and ready")
        } catch {
            isModelLoading = false
            await modelManager.markError(.tts, message: error.localizedDescription)
            throw error
        }
    }

    /// Register the download handler with ModelManager for Settings UI downloads.
    func registerDownloadHandler() {
        let manager = modelManager
        // swiftlint:disable:next unhandled_throwing_task
        Task {
            await manager.registerDownloadHandler(for: .tts) { _, progressCallback in
                await progressCallback(0.0)
                let modelId = TTSModelVariant.customVoice.rawValue
                let cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)
                if !HuggingFaceDownloader.weightsExist(in: cacheDir) {
                    try await HuggingFaceDownloader.downloadWeights(
                        modelId: modelId,
                        to: cacheDir,
                        additionalFiles: ["vocab.json", "merges.txt", "tokenizer_config.json"]
                    )
                }

                let tokenizerModelId = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
                let tokenizerDir = try HuggingFaceDownloader.getCacheDirectory(for: tokenizerModelId)
                if !HuggingFaceDownloader.weightsExist(in: tokenizerDir) {
                    try await HuggingFaceDownloader.downloadWeights(
                        modelId: tokenizerModelId,
                        to: tokenizerDir
                    )
                }

                await progressCallback(1.0)
                return cacheDir
            }
        }
    }
}

// MARK: - Position Tracking & Buffer Conversion

extension Qwen3TTSSpeechManager {
    /// Get the current playback position in samples from the player node.
    func currentPlaybackSample() -> Int64 {
        guard let nodeTime = playerNode.lastRenderTime,
            nodeTime.isSampleTimeValid,
            let playerTime = playerNode.playerTime(forNodeTime: nodeTime)
        else {
            return 0
        }
        return max(0, playerTime.sampleTime)
    }

    /// Map a sample position to a character position, snapped to a word boundary.
    func characterPosition(
        atSample sample: Int64,
        totalSamples: Int64,
        textLength: Int,
        text: String
    ) -> Int {
        guard totalSamples > 0 && textLength > 0 else {
            return 0
        }

        var charPos = Int(Double(sample) / Double(totalSamples) * Double(textLength))
        charPos = max(0, min(charPos, textLength))

        guard charPos > 0 && charPos < textLength else {
            return charPos
        }

        let startIndex = text.index(text.startIndex, offsetBy: charPos)
        if let spaceIndex = text[startIndex...].firstIndex(of: " ") {
            let distance = text.distance(from: startIndex, to: spaceIndex)
            if distance <= 10 {
                charPos = text.distance(from: text.startIndex, to: spaceIndex) + 1
            }
        } else if let spaceIndex = text[..<startIndex].lastIndex(of: " ") {
            charPos = text.distance(from: text.startIndex, to: spaceIndex) + 1
        }

        return min(charPos, textLength)
    }

    /// Convert Float32 PCM samples into an AVAudioPCMBuffer for scheduling.
    func createBuffer(from samples: [Float]) -> AVAudioPCMBuffer {
        let frameCount = AVAudioFrameCount(samples.count)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: frameCount) else {
            fatalError("Failed to create AVAudioPCMBuffer with capacity \(frameCount)")
        }
        buffer.frameLength = frameCount

        guard let channelData = buffer.floatChannelData else {
            return buffer
        }
        samples.withUnsafeBufferPointer { src in
            guard let base = src.baseAddress else {
                return
            }
            channelData[0].update(from: base, count: samples.count)
        }

        return buffer
    }
}

@preconcurrency import AVFoundation
import KokoroTTS

/// Speech manager using Kokoro neural TTS via AVAudioEngine + AVAudioPlayerNode.
///
/// Conforms to `SpeechManaging` so it's a drop-in replacement for `SpeechManager`.
/// Audio engine is kept running permanently (no start/stop per utterance).
/// Synthesis runs off the MainActor in a detached task to avoid blocking the UI.
@MainActor
public final class KokoroSpeechManager: NSObject, SpeechManaging {
    private static let logger = Log.logger(for: "KokoroSpeechManager")

    private let engine: KokoroEngine
    private let audioEngine = AVAudioEngine()
    private let playerNode = AVAudioPlayerNode()
    private let playbackFormat: AVAudioFormat

    /// The full text of the currently playing utterance.
    private var currentText: String = ""

    /// The session name for the current utterance.
    private var currentSession: String = ""

    /// Per-phoneme durations from the last synthesis (for interruption tracking).
    private var currentDurations: [Int] = []

    /// Handle to the in-flight synthesis task for cancellation on new speak() calls.
    private var currentSynthesisTask: Task<Void, Never>?

    /// Stream that yields each time an utterance finishes playing.
    public let finishedSpeaking: AsyncStream<Void>

    /// Continuation backing `finishedSpeaking`.
    private let finishedContinuation: AsyncStream<Void>.Continuation

    /// Speech rate multiplier (0.5 = half speed, 2.0 = double speed).
    ///
    /// Clamped to KokoroEngine.speedRange. Default 1.0.
    public var speechRate: Float = 1.0 {
        didSet {
            speechRate = min(
                max(speechRate, KokoroEngine.speedRange.lowerBound),
                KokoroEngine.speedRange.upperBound
            )
        }
    }

    // swiftlint:disable type_contents_order
    /// Create a KokoroSpeechManager with a loaded engine.
    public init(engine: KokoroEngine) {
        self.engine = engine
        let (stream, continuation) = AsyncStream<Void>.makeStream()
        self.finishedSpeaking = stream
        self.finishedContinuation = continuation
        self.playbackFormat =
            AVAudioFormat(
                standardFormatWithSampleRate: Double(KokoroEngine.sampleRate),
                channels: 1
            ) ?? AVAudioFormat()
        super.init()

        audioEngine.attach(playerNode)
        audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: playbackFormat)
        do {
            try audioEngine.start()
            Self.logger.info("Audio engine started")
        } catch {
            Self.logger.error("Failed to start audio engine: \(error)")
        }
    }

    /// Whether the player node is currently playing.
    public var isSpeaking: Bool { playerNode.isPlaying }

    deinit {
        currentSynthesisTask?.cancel()
        finishedContinuation.finish()
    }
    // swiftlint:enable type_contents_order

    /// Synthesize and play text using Kokoro TTS.
    ///
    /// Cancels any in-flight synthesis before starting a new one. Synthesis
    /// runs in a detached task off the MainActor; playback scheduling
    /// dispatches back to MainActor on completion.
    public func speak(_ text: String, voice: VoiceDescriptor, prefix: String, pitchMultiplier: Float) {
        cancelSynthesis()

        let fullText = "\(prefix): \(text)"
        currentText = fullText
        currentSession = prefix
        currentDurations = []

        let voiceName: String
        switch voice {
        case .kokoro(let name):
            voiceName = name

        case .system:
            voiceName = engine.availableVoices.first ?? "af_heart"
        }

        let engine = self.engine
        let speed = self.speechRate
        let logger = Self.logger

        currentSynthesisTask = Task.detached { [weak self] in
            do {
                let result = try engine.synthesize(
                    text: fullText,
                    voice: voiceName,
                    speed: speed
                )
                guard !Task.isCancelled else {
                    return
                }
                await MainActor.run { [weak self] in
                    self?.scheduleAndPlay(result: result, prefix: prefix, text: text)
                }
            } catch {
                guard !Task.isCancelled else {
                    return
                }
                logger.error("Synthesis failed: \(error)")
            }
        }
    }

    /// Interrupt playback and return what was heard vs. unheard.
    public func interrupt() -> InterruptInfo {
        cancelSynthesis()

        let elapsed: TimeInterval
        if let lastRender = playerNode.lastRenderTime,
            let playerTime = playerNode.playerTime(forNodeTime: lastRender)
        {
            elapsed = Double(playerTime.sampleTime) / playerTime.sampleRate
        } else {
            elapsed = 0
        }

        playerNode.stop()

        // Map elapsed time to character position via phoneme durations
        let framesPerSecond = Double(KokoroEngine.sampleRate)
        let elapsedFrames = Int(elapsed * framesPerSecond)

        var charPosition = 0
        var frameAccum = 0
        for dur in currentDurations {
            let frameCost = dur * KokoroEngine.hopSize
            if frameAccum + frameCost > elapsedFrames { break }
            frameAccum += frameCost
            charPosition += 1
        }

        // Approximate character split (phoneme index → rough text position)
        let textLen = currentText.count
        let fraction =
            currentDurations.isEmpty
            ? 0.0
            : Double(charPosition) / Double(currentDurations.count)
        let splitIdx = min(Int(fraction * Double(textLen)), textLen)

        let heard = String(currentText.prefix(splitIdx))
        let unheard = String(currentText.dropFirst(splitIdx))

        let elapsedStr = String(format: "%.1f", elapsed)
        Self.logger.info("Interrupted at ~\(elapsedStr)s for session \(self.currentSession)")

        return InterruptInfo(heardText: heard, unheardText: unheard, session: currentSession)
    }

    /// Stop playback without tracking interruption state.
    public func stop() {
        cancelSynthesis()
        playerNode.stop()
    }

    // MARK: - Private

    private func cancelSynthesis() {
        currentSynthesisTask?.cancel()
        currentSynthesisTask = nil
    }

    private func scheduleAndPlay(result: SynthesisResult, prefix: String, text: String) {
        currentDurations = result.tokenDurations

        let frameCount = AVAudioFrameCount(result.samples.count)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: playbackFormat, frameCapacity: frameCount),
            let channelData = buffer.floatChannelData?[0]
        else {
            Self.logger.error("Failed to create audio buffer")
            return
        }

        buffer.frameLength = frameCount
        result.samples.withUnsafeBufferPointer { src in
            guard let baseAddress = src.baseAddress else {
                return
            }
            channelData.update(from: baseAddress, count: result.samples.count)
        }

        let synthMs = String(format: "%.0f", result.synthesisTime * 1_000)
        let audioDur = String(format: "%.1f", result.duration)
        Self.logger.info("Speaking: \"\(prefix): \(text.prefix(60))...\" (\(synthMs)ms synth, \(audioDur)s audio)")

        playerNode.stop()
        playerNode.scheduleBuffer(buffer) { [weak self] in
            Task { @MainActor [weak self] in
                self?.handlePlaybackComplete()
            }
        }
        playerNode.play()
    }

    private func handlePlaybackComplete() {
        Self.logger.debug("Finished speaking: \"\(self.currentSession)\"")
        currentText = ""
        currentSession = ""
        currentDurations = []
        currentSynthesisTask = nil
        finishedContinuation.yield()
    }
}

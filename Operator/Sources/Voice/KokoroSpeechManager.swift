import AVFoundation
import KokoroTTS

/// Speech manager using Kokoro neural TTS via AVAudioEngine + AVAudioPlayerNode.
///
/// Conforms to `SpeechManaging` so it's a drop-in replacement for `SpeechManager`.
/// Audio engine is kept running permanently (no start/stop per utterance).
@MainActor
public final class KokoroSpeechManager: NSObject, SpeechManaging {
    private static let logger = Log.logger(for: "KokoroSpeechManager")

    private let engine: KokoroEngine
    private let audioEngine = AVAudioEngine()
    private let playerNode = AVAudioPlayerNode()
    private let playbackFormat: AVAudioFormat

    /// Whether the player node is currently playing.
    public var isSpeaking: Bool { playerNode.isPlaying }

    /// The full text of the currently playing utterance.
    private var currentText: String = ""

    /// The session name for the current utterance.
    private var currentSession: String = ""

    /// Per-phoneme durations from the last synthesis (for interruption tracking).
    private var currentDurations: [Int] = []

    /// Total sample count of the current buffer.
    private var currentSampleCount: Int = 0

    /// Playback speed multiplier.
    public var speechRate: Float = 1.0

    /// Stream that yields each time an utterance finishes playing.
    public let finishedSpeaking: AsyncStream<Void>

    /// Continuation backing `finishedSpeaking`.
    private let finishedContinuation: AsyncStream<Void>.Continuation

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
        setupAudioEngine()
    }

    private func setupAudioEngine() {
        audioEngine.attach(playerNode)
        audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: playbackFormat)
        do {
            try audioEngine.start()
            Self.logger.info("Audio engine started")
        } catch {
            Self.logger.error("Failed to start audio engine: \(error)")
        }
    }

    /// Synthesize and play text using Kokoro TTS.
    public func speak(_ text: String, voice: VoiceDescriptor, prefix: String, pitchMultiplier: Float) {
        let fullText = "\(prefix): \(text)"
        currentText = fullText
        currentSession = prefix

        let voiceName: String
        switch voice {
        case .kokoro(let name):
            voiceName = name

        case .system:
            voiceName = engine.availableVoices.first ?? "af_heart"
        }

        do {
            let result = try engine.synthesize(
                text: fullText,
                voice: voiceName,
                speed: speechRate
            )
            scheduleAndPlay(result: result, prefix: prefix, text: text)
        } catch {
            Self.logger.error("Synthesis failed: \(error)")
        }
    }

    private func scheduleAndPlay(result: SynthesisResult, prefix: String, text: String) {
        currentDurations = result.phonemeDurations
        currentSampleCount = result.samples.count

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
            DispatchQueue.main.async {
                self?.handlePlaybackComplete()
            }
        }
        playerNode.play()
    }

    /// Interrupt playback and return what was heard vs. unheard.
    public func interrupt() -> InterruptInfo {
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
            let frameCost = dur * (KokoroEngine.sampleRate / 24)  // scale to audio frames
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
        playerNode.stop()
    }

    private func handlePlaybackComplete() {
        Self.logger.debug("Finished speaking: \"\(self.currentSession)\"")
        currentText = ""
        currentSession = ""
        currentDurations = []
        currentSampleCount = 0
        finishedContinuation.yield()
    }
}

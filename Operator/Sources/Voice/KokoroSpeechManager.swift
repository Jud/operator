@preconcurrency import AVFoundation
import KokoroCoreML

/// Speech manager using Kokoro neural TTS via an externally-owned AVAudioPlayerNode.
///
/// Conforms to `SpeechManaging` so it's a drop-in replacement for `SpeechManager`.
/// The player node is created, attached, and started by `AudioHub` — this class
/// only schedules audio buffers on it. Uses the streaming `speak()` API — audio
/// chunks play as soon as they're synthesized, cutting perceived latency.
@MainActor
public final class KokoroSpeechManager: NSObject, SpeechManaging {
    private static let logger = Log.logger(for: "KokoroSpeechManager")

    private let engine: KokoroEngine
    private let playerNode: AVAudioPlayerNode

    /// The full text of the currently playing utterance.
    private var currentText: String = ""

    /// The session name for the current utterance.
    private var currentSession: String = ""

    /// Total scheduled audio duration (in seconds) for interruption tracking.
    private var scheduledDuration: TimeInterval = 0

    /// Handle to the in-flight streaming task for cancellation on new speak() calls.
    private var currentStreamTask: Task<Void, Never>?

    /// Stream that yields each time an utterance finishes playing.
    public let finishedSpeaking: AsyncStream<Void>

    /// Continuation backing `finishedSpeaking`.
    private let finishedContinuation: AsyncStream<Void>.Continuation

    /// Speech rate multiplier (0.5 = half speed, 2.0 = double speed).
    ///
    /// Default 1.0.
    public var speechRate: Float = 1.0

    /// Create a KokoroSpeechManager with a synthesis engine and an externally-owned player node.
    ///
    /// The player node must already be attached to an AVAudioEngine, connected
    /// to its mixer, and in the playing state (managed by AudioHub).
    public init(engine: KokoroEngine, playerNode: AVAudioPlayerNode) {
        self.engine = engine
        self.playerNode = playerNode
        let (stream, continuation) = AsyncStream<Void>.makeStream()
        self.finishedSpeaking = stream
        self.finishedContinuation = continuation
        super.init()
        Self.logger.info("KokoroSpeechManager initialized (external player node)")
    }

    /// Whether the player node is currently playing.
    public var isSpeaking: Bool { playerNode.isPlaying }

    deinit {
        currentStreamTask?.cancel()
        finishedContinuation.finish()
    }

    /// Synthesize and play text using Kokoro TTS streaming API.
    ///
    /// Cancels any in-flight synthesis before starting a new one. Audio chunks
    /// are scheduled on the player node as they arrive from the stream.
    public func speak(_ text: String, voice: VoiceDescriptor, prefix: String, pitchMultiplier: Float) {
        cancelStream()
        playerNode.stop()
        playerNode.play()

        let fullText = "\(prefix): \(text)"
        currentText = fullText
        currentSession = prefix
        scheduledDuration = 0

        let voiceName: String
        switch voice {
        case .kokoro(let name):
            voiceName = name

        case .system:
            voiceName = engine.availableVoices.first ?? "af_heart"
        }

        let speed = self.speechRate

        do {
            let stream = try engine.speak(fullText, voice: voiceName, speed: speed)
            Self.logger.info(
                "Speaking: \"\(prefix): \(text.prefix(60))...\""
            )

            currentStreamTask = Task { @MainActor [weak self] in
                for await event in stream {
                    guard let self, !Task.isCancelled else { break }
                    switch event {
                    case .audio(let buffer):
                        self.scheduleBuffer(buffer)

                    case .chunkFailed(let error):
                        Self.logger.warning("Chunk failed: \(error)")
                    }
                }
                guard let self, !Task.isCancelled
                else { return }
                self.scheduleCompletionCallback()
            }
        } catch {
            Self.logger.error("Failed to start speech stream: \(error)")
        }
    }

    /// Interrupt playback and return what was heard vs. unheard.
    public func interrupt() -> InterruptInfo {
        cancelStream()

        let elapsed: TimeInterval
        if let lastRender = playerNode.lastRenderTime,
            let playerTime = playerNode.playerTime(forNodeTime: lastRender)
        {
            elapsed = Double(playerTime.sampleTime) / playerTime.sampleRate
        } else {
            elapsed = 0
        }

        playerNode.stop()
        playerNode.play()

        let textLen = currentText.count
        let fraction = scheduledDuration > 0 ? elapsed / scheduledDuration : 0
        let splitIdx = min(Int(fraction * Double(textLen)), textLen)

        let heard = String(currentText.prefix(splitIdx))
        let unheard = String(currentText.dropFirst(splitIdx))

        let elapsedStr = String(format: "%.1f", elapsed)
        Self.logger.info("Interrupted at ~\(elapsedStr)s for session \(self.currentSession)")

        return InterruptInfo(heardText: heard, unheardText: unheard, session: currentSession)
    }

    /// Stop playback without tracking interruption state.
    public func stop() {
        cancelStream()
        playerNode.stop()
        playerNode.play()
    }

    // MARK: - Private

    private func cancelStream() {
        currentStreamTask?.cancel()
        currentStreamTask = nil
    }

    private func scheduleBuffer(_ buffer: AVAudioPCMBuffer) {
        let duration = Double(buffer.frameLength) / buffer.format.sampleRate
        scheduledDuration += duration
        playerNode.scheduleBuffer(buffer)
    }

    /// Schedule a no-op buffer to detect when all previous buffers finish playing.
    ///
    /// AVAudioPlayerNode has no standalone completion API — the only way to get
    /// notified when playback ends is via `scheduleBuffer(_:completionHandler:)`.
    /// A zero-frame buffer adds no audio but its completion fires after all
    /// previously scheduled buffers have played.
    private func scheduleCompletionCallback() {
        playerNode.scheduleBuffer(
            AVAudioPCMBuffer(
                pcmFormat: KokoroEngine.audioFormat,
                frameCapacity: 0
            ) ?? AVAudioPCMBuffer()
        ) { [weak self] in
            Task { @MainActor [weak self] in
                self?.handlePlaybackComplete()
            }
        }
    }

    private func handlePlaybackComplete() {
        Self.logger.debug("Finished speaking: \"\(self.currentSession)\"")
        currentText = ""
        currentSession = ""
        scheduledDuration = 0
        currentStreamTask = nil
        finishedContinuation.yield()
    }
}

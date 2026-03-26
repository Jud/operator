import AVFoundation

/// All recognized audio cue names.
public enum AudioCue: String, CaseIterable, Sendable {
    case listening
    case processing
    case delivered
    case error
    case pending
    case dictation
    case dismissed
}

/// Protocol for playing non-verbal audio feedback tones.
///
/// Extracted to enable silent mock implementations in tests.
@MainActor
public protocol AudioFeedbackProviding: Sendable {
    /// Play an audio feedback tone immediately.
    func play(_ cue: AudioCue)

    /// Play an audio feedback tone after any currently playing cue finishes.
    ///
    /// If nothing is playing, plays immediately. Otherwise schedules playback
    /// after the current cue's remaining duration plus a small gap.
    func playAfterCurrent(_ cue: AudioCue)
}

/// Plays non-verbal audio feedback tones at state transitions.
///
/// When a shared AVAudioEngine is provided (from KokoroSpeechManager), tones
/// play through the same output stream as TTS — no AudioQueue teardown/recreate,
/// no Bluetooth audio blip. Falls back to AVAudioPlayer when no engine is provided.
///
/// Seven bundled .caf files provide immediate audible feedback without speech:
/// - listening: soft rising tone when push-to-talk activates
/// - processing: brief double-tick when push-to-talk releases
/// - delivered: low pleasant chime on successful message delivery
/// - error: descending two-tone on any error
/// - pending: subtle notification when returning to IDLE with queued messages
/// - dictation: short, subtle high-pitched chime confirming text insertion at cursor
/// - dismissed: gentle soft pop when recording is empty or too short (no error)
@MainActor
public final class AudioFeedback: AudioFeedbackProviding {
    private static let logger = Log.logger(for: "AudioFeedback")

    /// Minimum time between the start of one cue and the start of the next.
    private static let minimumSpacing: TimeInterval = 0.3

    /// Pre-loaded audio buffers for engine-based playback.
    private let buffers: [AudioCue: AVAudioPCMBuffer]

    /// Fallback AVAudioPlayer instances (used when no shared engine is available).
    private let fallbackPlayers: [AudioCue: AVAudioPlayer]

    /// Shared audio engine for blip-free playback through the same output stream as TTS.
    private let sharedEngine: AVAudioEngine?

    /// Dedicated player node for feedback tones, attached to the shared engine.
    private let feedbackNode: AVAudioPlayerNode?

    /// Earliest time a follow-up cue (via `playAfterCurrent`) should start.
    private var nextAllowedPlayTime: Date = .distantPast

    /// Creates a new audio feedback player, pre-loading all tone files.
    ///
    /// - Parameter sharedEngine: Optional AVAudioEngine (typically from KokoroSpeechManager).
    ///   When provided, tones play through a dedicated node on the shared engine,
    ///   avoiding AudioQueue teardown that causes Bluetooth audio blips.
    public init(sharedEngine: AVAudioEngine? = nil) {
        self.sharedEngine = sharedEngine

        var loadedBuffers: [AudioCue: AVAudioPCMBuffer] = [:]
        var loadedPlayers: [AudioCue: AVAudioPlayer] = [:]

        for cue in AudioCue.allCases {
            guard let url = Bundle.module.url(forResource: cue.rawValue, withExtension: "caf") else {
                Self.logger.error("Missing audio resource: \(cue.rawValue).caf")
                continue
            }
            do {
                if sharedEngine != nil {
                    let audioFile = try AVAudioFile(forReading: url)
                    guard
                        let buffer = AVAudioPCMBuffer(
                            pcmFormat: audioFile.processingFormat,
                            frameCapacity: AVAudioFrameCount(audioFile.length)
                        )
                    else {
                        Self.logger.error("Failed to create buffer for \(cue.rawValue)")
                        continue
                    }
                    try audioFile.read(into: buffer)
                    loadedBuffers[cue] = buffer
                    Self.logger.debug("Loaded audio cue (buffer): \(cue.rawValue)")
                } else {
                    let player = try AVAudioPlayer(contentsOf: url)
                    player.prepareToPlay()
                    loadedPlayers[cue] = player
                    Self.logger.debug("Loaded audio cue (player): \(cue.rawValue)")
                }
            } catch {
                Self.logger.error("Failed to load \(cue.rawValue).caf: \(error)")
            }
        }

        buffers = loadedBuffers
        fallbackPlayers = loadedPlayers

        if let engine = sharedEngine, let firstBuffer = loadedBuffers.values.first {
            let node = AVAudioPlayerNode()
            engine.attach(node)
            // Connect with the buffer's format (1ch 44.1kHz) — the mixer handles
            // conversion to the engine's output format automatically.
            engine.connect(node, to: engine.mainMixerNode, format: firstBuffer.format)
            node.play()
            self.feedbackNode = node
            Self.logger.info("Feedback tones attached to shared audio engine")
        } else {
            self.feedbackNode = nil
        }
    }

    /// Play an audio feedback tone immediately.
    public func play(_ cue: AudioCue) {
        guard UserDefaults.standard.bool(forKey: "feedbackSoundsEnabled") else {
            return
        }

        let duration: TimeInterval

        if let node = feedbackNode, let buffer = buffers[cue] {
            node.stop()
            node.play()
            node.scheduleBuffer(buffer, at: nil)
            duration = Double(buffer.frameLength) / buffer.format.sampleRate
        } else if let player = fallbackPlayers[cue] {
            player.currentTime = 0
            player.play()
            duration = player.duration
        } else {
            Self.logger.warning("No player available for cue: \(cue.rawValue)")
            return
        }

        let now = Date()
        nextAllowedPlayTime = max(
            now.addingTimeInterval(duration),
            now.addingTimeInterval(Self.minimumSpacing)
        )
        Self.logger.debug("Playing cue: \(cue.rawValue)")
    }

    /// Play an audio feedback tone after any currently playing cue finishes.
    public func playAfterCurrent(_ cue: AudioCue) {
        let delay = nextAllowedPlayTime.timeIntervalSinceNow
        if delay <= 0 {
            play(cue)
        } else {
            Task { @MainActor [weak self] in
                try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                self?.play(cue)
            }
        }
    }
}

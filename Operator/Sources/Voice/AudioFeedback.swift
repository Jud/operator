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
/// Tones play through an externally-owned AVAudioPlayerNode (from AudioHub),
/// sharing the same output stream as TTS. No AudioQueue teardown, no Bluetooth blip.
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

    /// Pre-loaded audio buffers keyed by cue.
    private let buffers: [AudioCue: AVAudioPCMBuffer]

    /// Player node owned by AudioHub, already attached and playing.
    private let playerNode: AVAudioPlayerNode

    /// Earliest time a follow-up cue (via `playAfterCurrent`) should start.
    private var nextAllowedPlayTime: Date = .distantPast

    /// Creates a new audio feedback player, pre-loading all tone files as PCM buffers.
    ///
    /// - Parameter playerNode: An AVAudioPlayerNode already attached to AudioHub's
    ///   engine, connected to the mixer, and in the playing state.
    public init(playerNode: AVAudioPlayerNode) {
        self.playerNode = playerNode

        var loaded: [AudioCue: AVAudioPCMBuffer] = [:]
        for cue in AudioCue.allCases {
            guard let url = Bundle.module.url(forResource: cue.rawValue, withExtension: "caf") else {
                Self.logger.error("Missing audio resource: \(cue.rawValue).caf")
                continue
            }
            do {
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
                loaded[cue] = buffer
                Self.logger.debug("Loaded audio cue: \(cue.rawValue)")
            } catch {
                Self.logger.error("Failed to load \(cue.rawValue).caf: \(error)")
            }
        }
        buffers = loaded
        Self.logger.info("AudioFeedback initialized (\(loaded.count) cues loaded)")
    }

    /// Play an audio feedback tone immediately.
    public func play(_ cue: AudioCue) {
        guard UserDefaults.standard.bool(forKey: "feedbackSoundsEnabled") else {
            return
        }
        guard let buffer = buffers[cue] else {
            Self.logger.warning("No buffer available for cue: \(cue.rawValue)")
            return
        }

        playerNode.stop()
        playerNode.play()
        playerNode.scheduleBuffer(buffer, at: nil)

        let duration = Double(buffer.frameLength) / buffer.format.sampleRate
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

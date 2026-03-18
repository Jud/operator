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
/// Six bundled .caf files provide immediate audible feedback without speech:
/// - listening: soft rising tone when push-to-talk activates
/// - processing: brief double-tick when push-to-talk releases
/// - delivered: low pleasant chime on successful message delivery
/// - error: descending two-tone on any error
/// - pending: subtle notification when returning to IDLE with queued messages
/// - dictation: short, subtle high-pitched chime confirming text insertion at cursor
/// - dismissed: gentle soft pop when recording is empty or too short (no error)
///
/// Each tone is pre-loaded via AVAudioPlayer.prepareToPlay() for zero-latency playback.
/// The play method resets currentTime to 0 before playing, allowing rapid re-trigger
/// without waiting for the previous playback to finish.
@MainActor
public final class AudioFeedback: AudioFeedbackProviding {
    private static let logger = Log.logger(for: "AudioFeedback")

    private let players: [AudioCue: AVAudioPlayer]

    /// Earliest time a follow-up cue (via `playAfterCurrent`) should start.
    ///
    /// Set to the later of: current cue's end time, or current cue's start + minimumSpacing.
    private var nextAllowedPlayTime: Date = .distantPast

    /// Minimum time between the start of one cue and the start of the next.
    private static let minimumSpacing: TimeInterval = 0.3

    /// Creates a new audio feedback player, pre-loading all tone files.
    public init() {
        var loaded: [AudioCue: AVAudioPlayer] = [:]
        for cue in AudioCue.allCases {
            guard let url = Bundle.module.url(forResource: cue.rawValue, withExtension: "caf") else {
                Self.logger.error("Missing audio resource: \(cue.rawValue).caf")
                continue
            }
            do {
                let player = try AVAudioPlayer(contentsOf: url)
                player.prepareToPlay()
                loaded[cue] = player
                Self.logger.debug("Loaded audio cue: \(cue.rawValue)")
            } catch {
                Self.logger.error("Failed to load \(cue.rawValue).caf: \(error.localizedDescription)")
            }
        }
        players = loaded
    }

    /// Play an audio feedback tone immediately.
    ///
    /// Resets playback position to allow rapid re-triggering.
    public func play(_ cue: AudioCue) {
        guard UserDefaults.standard.bool(forKey: "feedbackSoundsEnabled") else {
            return
        }
        guard let player = players[cue] else {
            Self.logger.warning("No player available for cue: \(cue.rawValue)")
            return
        }
        player.currentTime = 0
        player.play()
        let now = Date()
        nextAllowedPlayTime = max(
            now.addingTimeInterval(player.duration),
            now.addingTimeInterval(Self.minimumSpacing)
        )
        Self.logger.debug("Playing cue: \(cue.rawValue)")
    }

    /// Play an audio feedback tone after any currently playing cue finishes.
    ///
    /// If nothing is playing, plays immediately. Otherwise schedules playback
    /// after the current cue's remaining duration plus a small gap.
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

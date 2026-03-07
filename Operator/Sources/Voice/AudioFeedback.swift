import AVFoundation
import OSLog

/// Plays non-verbal audio feedback tones at state transitions.
///
/// Five bundled .caf files provide immediate audible feedback without speech:
/// - listening: soft rising tone when push-to-talk activates
/// - processing: brief double-tick when push-to-talk releases
/// - delivered: low pleasant chime on successful message delivery
/// - error: descending two-tone on any error
/// - pending: subtle notification when returning to IDLE with queued messages
///
/// Each tone is pre-loaded via AVAudioPlayer.prepareToPlay() for zero-latency playback.
/// The play method resets currentTime to 0 before playing, allowing rapid re-trigger
/// without waiting for the previous playback to finish.
public class AudioFeedback: @unchecked Sendable {
    /// All recognized audio cue names.
    public enum Cue: String, CaseIterable, Sendable {
        case listening
        case processing
        case delivered
        case error
        case pending
    }

    private static let logger = Logger(subsystem: "com.operator.app", category: "AudioFeedback")

    private var players: [Cue: AVAudioPlayer] = [:]

    public init() {
        for cue in Cue.allCases {
            guard let url = Bundle.module.url(forResource: cue.rawValue, withExtension: "caf") else {
                Self.logger.error("Missing audio resource: \(cue.rawValue).caf")
                continue
            }
            do {
                let player = try AVAudioPlayer(contentsOf: url)
                player.prepareToPlay()
                players[cue] = player
                Self.logger.debug("Loaded audio cue: \(cue.rawValue)")
            } catch {
                Self.logger.error("Failed to load \(cue.rawValue).caf: \(error.localizedDescription)")
            }
        }
    }

    /// Play an audio feedback tone. Resets playback position to allow rapid re-triggering.
    public func play(_ cue: Cue) {
        guard let player = players[cue] else {
            Self.logger.warning("No player available for cue: \(cue.rawValue)")
            return
        }
        player.currentTime = 0
        player.play()
        Self.logger.debug("Playing cue: \(cue.rawValue)")
    }
}

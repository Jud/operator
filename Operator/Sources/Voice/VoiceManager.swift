import AVFoundation
import OSLog

/// Manages voice selection for Operator and agent speech output.
///
/// Implements a premium -> enhanced -> default fallback chain to select the
/// highest-quality available voices. Operator gets one voice; agents get a
/// different voice (when possible) so the user can distinguish system messages
/// from agent responses by ear.
///
/// Per-agent pitch multiplier variation is provided via `nextAgentPitchMultiplier()`
/// so that each registered agent sounds slightly different, aiding auditory
/// differentiation when multiple agents are active.
public final class VoiceManager: Sendable {
    private static let logger = Logger(subsystem: "com.operator.app", category: "VoiceManager")
    private static let pitchValues: [Float] = [1.0, 1.05, 0.95, 1.1, 0.9, 1.15, 0.85]

    /// The voice used for all Operator system messages (confirmations, errors, prompts).
    public let operatorVoice: AVSpeechSynthesisVoice

    /// The default voice used for agent speech output.
    ///
    /// Individual agents may vary pitch via `pitchMultiplier` to sound distinct.
    public let defaultAgentVoice: AVSpeechSynthesisVoice

    /// Thread-safe counter for pitch assignment, using atomic operations.
    private let pitchCounter = LockedValue<Int>(0)

    /// Creates a new voice manager with premium voice fallback selection.
    public init() {
        let candidates = AVSpeechSynthesisVoice.speechVoices()
            .filter { $0.language.hasPrefix("en") }

        let selectedOperatorVoice =
            candidates.first { $0.quality == .premium }
            ?? candidates.first { $0.quality == .enhanced }
            ?? AVSpeechSynthesisVoice(language: "en-US")
            ?? AVSpeechSynthesisVoice()

        self.operatorVoice = selectedOperatorVoice

        self.defaultAgentVoice =
            candidates.first { $0.quality == .premium && $0.identifier != selectedOperatorVoice.identifier }
            ?? candidates.first { $0.quality == .enhanced && $0.identifier != selectedOperatorVoice.identifier }
            ?? selectedOperatorVoice

        Self.logger.info(
            "Operator voice: \(selectedOperatorVoice.identifier) (quality: \(selectedOperatorVoice.quality.rawValue))"
        )
        Self.logger.info(
            "Agent voice: \(self.defaultAgentVoice.identifier) (quality: \(self.defaultAgentVoice.quality.rawValue))"
        )

        if selectedOperatorVoice.identifier == self.defaultAgentVoice.identifier {
            Self.logger.warning("Operator and agent voices are the same; no distinct alternative available")
        }
    }

    /// Returns the next pitch multiplier for a newly registered agent.
    ///
    /// Each call returns a different value from a cycling set, so agents
    /// registered in sequence get distinct pitches. Thread-safe.
    public func nextAgentPitchMultiplier() -> Float {
        let index = pitchCounter.withLock { value in
            let current = value
            value = (value + 1) % Self.pitchValues.count
            return current
        }

        let pitch = Self.pitchValues[index]

        Self.logger.debug("Assigned pitch multiplier: \(pitch)")
        return pitch
    }
}

/// Simple lock-based thread-safe wrapper for a mutable value.
///
/// Used internally by VoiceManager for the pitch counter.
private final class LockedValue<T>: @unchecked Sendable {
    private var storedValue: T
    private let lock = NSLock()

    init(_ value: T) {
        self.storedValue = value
    }

    func withLock<R>(_ body: (inout T) -> R) -> R {
        lock.lock()
        defer { lock.unlock() }
        return body(&storedValue)
    }
}

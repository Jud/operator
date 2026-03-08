import AVFoundation

/// Manages voice selection for Operator and agent speech output.
///
/// Implements a premium -> enhanced -> default fallback chain to select the
/// highest-quality available voices. Operator gets one voice; agents get a
/// different voice (when possible) so the user can distinguish system messages
/// from agent responses by ear.
///
/// Returns `VoiceDescriptor` values containing both Apple and Qwen3-TTS speaker
/// identifiers, enabling seamless engine fallback without voice remapping.
///
/// Per-agent pitch multiplier variation is provided via `nextAgentPitchMultiplier()`
/// so that each registered agent sounds slightly different, aiding auditory
/// differentiation when multiple agents are active.
public final class VoiceManager: Sendable {
    private static let logger = Log.logger(for: "VoiceManager")
    private static let pitchValues: [Float] = [1.0, 1.05, 0.95, 1.1, 0.9, 1.15, 0.85]

    /// Qwen3-TTS speaker IDs available for voice assignment.
    ///
    /// The first entry is reserved for the Operator system voice.
    /// Remaining entries cycle through agent registrations.
    /// These map to speaker names in the Qwen3-TTS CustomVoice model config.
    static let qwenSpeakers: [String] = [
        "vivian", "ryan", "emma", "lucas", "olivia", "ethan", "sophia"
    ]

    /// The voice used for all Operator system messages (confirmations, errors, prompts).
    public let operatorVoice: VoiceDescriptor

    /// The default voice used for agent speech output.
    ///
    /// Individual agents may vary pitch via `pitchMultiplier` to sound distinct.
    public let defaultAgentVoice: VoiceDescriptor

    /// Thread-safe counter for pitch assignment, using atomic operations.
    private let pitchCounter = LockedValue<Int>(0)

    /// Thread-safe counter for Qwen speaker cycling.
    private let speakerCounter = LockedValue<Int>(0)

    /// Creates a new voice manager with premium voice fallback selection.
    public init() {
        // Avoid eager AVSpeechSynthesisVoice discovery during initialization.
        // On macOS this can block on system speech/accessibility lookups, and
        // Operator uses Qwen speaker IDs as the primary voice identity anyway.
        let selectedOperatorVoice = AVSpeechSynthesisVoice()
        self.operatorVoice = VoiceDescriptor(
            appleVoice: selectedOperatorVoice,
            qwenSpeakerID: Self.qwenSpeakers[0]
        )
        let selectedAgentVoice = selectedOperatorVoice

        self.defaultAgentVoice = VoiceDescriptor(
            appleVoice: selectedAgentVoice,
            qwenSpeakerID: Self.qwenSpeakers.count > 1 ? Self.qwenSpeakers[1] : Self.qwenSpeakers[0]
        )
        let agentSpeaker = Self.qwenSpeakers.count > 1 ? Self.qwenSpeakers[1] : Self.qwenSpeakers[0]

        Self.logger.info(
            "Operator voice initialized with Qwen speaker \(Self.qwenSpeakers[0])"
        )
        Self.logger.info(
            "Agent voice initialized with Qwen speaker \(agentSpeaker)"
        )
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

    /// Returns the next Qwen3-TTS speaker ID for a newly registered agent.
    ///
    /// Cycles through the agent speaker pool (skipping index 0, which is
    /// reserved for the Operator voice). Thread-safe.
    public func nextAgentQwenSpeaker() -> String {
        let agentSpeakers = Array(Self.qwenSpeakers.dropFirst())
        guard !agentSpeakers.isEmpty else {
            return Self.qwenSpeakers[0]
        }

        let index = speakerCounter.withLock { value in
            let current = value
            value = (value + 1) % agentSpeakers.count
            return current
        }

        let speaker = agentSpeakers[index]
        Self.logger.debug("Assigned Qwen speaker: \(speaker)")
        return speaker
    }

    /// Returns a new VoiceDescriptor for an agent, combining the default Apple
    /// agent voice with the next cycling Qwen3-TTS speaker ID.
    public func nextAgentVoice() -> VoiceDescriptor {
        VoiceDescriptor(
            appleVoice: defaultAgentVoice.appleVoice,
            qwenSpeakerID: nextAgentQwenSpeaker()
        )
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

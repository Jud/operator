import AVFoundation

/// Manages voice selection for Operator and agent speech output.
///
/// Returns `VoiceDescriptor` values for Apple TTS. Future engine
/// integrations can extend VoiceDescriptor without changing call sites.
public final class VoiceManager: Sendable {
    private static let logger = Log.logger(for: "VoiceManager")

    /// The voice used for all Operator system messages and agent speech output.
    ///
    /// Built once at init since the underlying voice is immutable.
    public let operatorVoice: VoiceDescriptor

    /// The default voice used for agent speech output (same as operatorVoice).
    public var defaultAgentVoice: VoiceDescriptor { operatorVoice }

    /// Creates a new voice manager.
    public init() {
        self.operatorVoice = VoiceDescriptor(appleVoice: AVSpeechSynthesisVoice())
        Self.logger.info("VoiceManager initialized")
    }

    /// Returns a VoiceDescriptor for the next agent.
    public func nextAgentVoice() -> VoiceDescriptor {
        operatorVoice
    }
}

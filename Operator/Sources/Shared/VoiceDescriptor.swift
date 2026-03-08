import AVFoundation

/// Engine-agnostic voice identity for speech synthesis.
///
/// Stores both Apple and Qwen3-TTS representations so that engine
/// fallback can occur without voice remapping. VoiceManager assigns
/// both representations at session registration time.
public struct VoiceDescriptor: Sendable, Equatable {
    /// Apple AVSpeechSynthesisVoice for Apple TTS engine.
    public let appleVoice: AVSpeechSynthesisVoice

    /// Qwen3-TTS speaker identifier for local TTS engine.
    public let qwenSpeakerID: String

    /// Creates a new voice descriptor pairing an Apple voice with a Qwen speaker.
    public init(appleVoice: AVSpeechSynthesisVoice, qwenSpeakerID: String) {
        self.appleVoice = appleVoice
        self.qwenSpeakerID = qwenSpeakerID
    }

    /// Equality based on Apple voice identifier and Qwen speaker ID.
    public static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.appleVoice.identifier == rhs.appleVoice.identifier
            && lhs.qwenSpeakerID == rhs.qwenSpeakerID
    }
}

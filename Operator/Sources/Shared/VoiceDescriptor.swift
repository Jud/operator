import AVFoundation

/// Voice identity for speech synthesis.
///
/// Wraps an Apple AVSpeechSynthesisVoice. Future engine integrations
/// can add additional identifiers here without changing call sites.
public struct VoiceDescriptor: Sendable, Equatable {
    /// Apple AVSpeechSynthesisVoice for TTS.
    public let appleVoice: AVSpeechSynthesisVoice

    /// Creates a new voice descriptor.
    public init(appleVoice: AVSpeechSynthesisVoice) {
        self.appleVoice = appleVoice
    }

    /// Equality based on Apple voice identifier.
    public static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.appleVoice.identifier == rhs.appleVoice.identifier
    }
}

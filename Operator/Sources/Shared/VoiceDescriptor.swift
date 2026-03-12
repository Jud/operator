import AVFoundation

/// Voice identity for speech synthesis, supporting both Kokoro and Apple TTS.
///
/// Kokoro voices provide natural-sounding output via CoreML neural TTS.
/// System voices fall back to AVSpeechSynthesizer when Kokoro models aren't available.
public enum VoiceDescriptor: Sendable, Equatable {
    /// Kokoro neural TTS voice preset (e.g. "af_heart", "am_adam").
    case kokoro(name: String)
    /// Apple AVSpeechSynthesisVoice fallback.
    case system(AVSpeechSynthesisVoice)

    /// Display name for this voice.
    public var displayName: String {
        switch self {
        case .kokoro(let name):
            return name

        case .system(let voice):
            return voice.name
        }
    }

    /// Whether this is a Kokoro neural voice.
    public var isKokoro: Bool {
        guard case .kokoro = self else {
            return false
        }
        return true
    }

    /// The Apple voice, if this is a system voice.
    public var appleVoice: AVSpeechSynthesisVoice? {
        guard case .system(let voice) = self else {
            return nil
        }
        return voice
    }

    /// The Kokoro voice name, if this is a Kokoro voice.
    public var kokoroName: String? {
        guard case .kokoro(let name) = self else {
            return nil
        }
        return name
    }

    /// Equality based on voice identifier or kokoro name.
    public static func == (lhs: Self, rhs: Self) -> Bool {
        switch (lhs, rhs) {
        case (.kokoro(let lhsName), .kokoro(let rhsName)):
            return lhsName == rhsName

        case (.system(let lhsVoice), .system(let rhsVoice)):
            return lhsVoice.identifier == rhsVoice.identifier

        default:
            return false
        }
    }
}

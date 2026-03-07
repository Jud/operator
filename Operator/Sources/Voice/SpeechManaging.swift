import AVFoundation

/// Protocol abstracting speech synthesis management for dependency injection.
///
/// Enables mock speech in E2E tests while using real SpeechManager in production.
/// Both real and mock types conform to this protocol. All access must occur on
/// the main actor since AVSpeechSynthesizer operates on the main thread.
@MainActor
public protocol SpeechManaging: AnyObject, Sendable {
    var synthesizer: AVSpeechSynthesizer { get }
    var isSpeaking: Bool { get }
    var onFinishedSpeaking: (() -> Void)? { get set }

    func speak(_ text: String, voice: AVSpeechSynthesisVoice, prefix: String, pitchMultiplier: Float)
    func interrupt() -> InterruptInfo
    func stop()
}

/// Result of interrupting speech playback.
public struct InterruptInfo: Sendable {
    /// The portion of the message that was spoken before interruption.
    public let heardText: String

    /// The remaining portion that was not yet spoken.
    public let unheardText: String

    /// The session name whose speech was interrupted.
    public let session: String

    /// Creates a new interrupt info result.
    public init(heardText: String, unheardText: String, session: String) {
        self.heardText = heardText
        self.unheardText = unheardText
        self.session = session
    }
}

/// Default implementation providing a convenience method without pitchMultiplier.
extension SpeechManaging {
    /// Speak with default pitch multiplier of 1.0.
    public func speak(_ text: String, voice: AVSpeechSynthesisVoice, prefix: String) {
        speak(text, voice: voice, prefix: prefix, pitchMultiplier: 1.0)
    }
}

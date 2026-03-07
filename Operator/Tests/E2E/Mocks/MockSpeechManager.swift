import AVFoundation
import OperatorCore

/// Mock speech manager that records spoken messages instead of producing audio.
///
/// Conforms to SpeechManaging so it can be injected into StateMachine and AudioQueue.
/// Uses a real AVSpeechSynthesizer instance (never called to speak) so that
/// synthesizer.isSpeaking always returns false, preventing interruption logic
/// from firing during tests.
///
/// spokenMessages collects every speak() call for test verification.
@MainActor
internal final class MockSpeechManager: SpeechManaging {
    let synthesizer = AVSpeechSynthesizer()
    var onFinishedSpeaking: (() -> Void)?

    /// All messages passed to speak(), in order.
    ///
    /// Each entry records the text and prefix.
    var spokenMessages: [(text: String, prefix: String)] = []

    func speak(_ text: String, voice: AVSpeechSynthesisVoice, prefix: String, pitchMultiplier: Float) {
        spokenMessages.append((text: text, prefix: prefix))
        // Immediately signal finished so AudioQueue advances without delay
        onFinishedSpeaking?()
    }

    func interrupt() -> InterruptInfo {
        InterruptInfo(heardText: "", unheardText: "", session: "")
    }

    func stop() {}
}

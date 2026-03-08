import AVFoundation
import OperatorCore

/// Mock speech manager that records spoken messages instead of producing audio.
///
/// Conforms to SpeechManaging so it can be injected into StateMachine and AudioQueue.
/// isSpeaking always returns false, preventing interruption logic from firing
/// during tests.
///
/// spokenMessages collects every speak() call for test verification.
@MainActor
internal final class MockSpeechManager: SpeechManaging {
    let isSpeaking = false

    /// Stream that yields each time speak() is called, signalling completion.
    let finishedSpeaking: AsyncStream<Void>

    /// Continuation backing `finishedSpeaking`.
    private let finishedContinuation: AsyncStream<Void>.Continuation

    /// All messages passed to speak(), in order.
    ///
    /// Each entry records the text and prefix.
    var spokenMessages: [(text: String, prefix: String)] = []

    init() {
        let (stream, continuation) = AsyncStream<Void>.makeStream()
        self.finishedSpeaking = stream
        self.finishedContinuation = continuation
    }

    func speak(_ text: String, voice: VoiceDescriptor, prefix: String, pitchMultiplier: Float) {
        spokenMessages.append((text: text, prefix: prefix))
        // Immediately signal finished so AudioQueue advances without delay
        finishedContinuation.yield()
    }

    func interrupt() -> InterruptInfo {
        InterruptInfo(heardText: "", unheardText: "", session: "")
    }

    func stop() {}
}

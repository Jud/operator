import Foundation

/// Protocol abstracting speech transcription for dependency injection.
///
/// Enables mock transcription in E2E tests while using real SpeechTranscriber
/// in production. Both real and mock types conform to this protocol.
///
/// All methods are called from @MainActor context (via the StateMachine).
@MainActor
public protocol SpeechTranscribing: AnyObject, Sendable {
    /// Whether the transcriber is currently listening for speech.
    var isListening: Bool { get }

    func startListening() throws
    func stopListening() async -> String?
    func stopListeningWithTimeout(seconds: TimeInterval) async -> String?
}

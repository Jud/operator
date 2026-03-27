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

    /// URL of the WAV file saved from the most recent recording session, or nil
    /// if no audio was captured. Reset on each `startListening()` call.
    var lastAudioFileURL: URL? { get }

    func startListening(contextualStrings: [String]) throws
    func stopListening() async -> String?
    func stopListeningWithTimeout(seconds: TimeInterval) async -> String?

    /// Stop audio capture and check whether the recording contains speech.
    ///
    /// - Returns: `true` if speech detected, `false` if silence.
    func stopAndCheckSilence() async -> Bool

    /// Run the transcription engine on audio from the last session.
    ///
    /// Must be called after ``stopAndCheckSilence()`` returns `true`.
    func finishTranscription() async -> String?

    /// Synchronously tear down audio capture if currently listening.
    ///
    /// Unlike `stopAndCheckSilence()`, this is synchronous and does not
    /// process the captured audio. Used by `enterIdle()` to avoid a race
    /// with the next `triggerStart()`.
    func tearDownIfListening()

    func replaceEngine(_ newEngine: any TranscriptionEngine)
}

import AVFoundation

/// Protocol abstracting the speech recognition engine.
///
/// Decouples audio-to-text conversion from audio capture, allowing different
/// recognition backends (Apple Speech, local models) to be swapped via injection.
///
/// Lifecycle: prepare() -> append(_:) (called per buffer) -> finishAndTranscribe().
/// Call cancel() to abort an in-progress session.
public protocol TranscriptionEngine: Sendable {
    /// Prepare the engine for a new transcription session.
    ///
    /// Called once before audio capture begins. Implementations should allocate
    /// any request or session objects needed for the upcoming transcription.
    func prepare() throws

    /// Append an audio buffer captured from the microphone.
    ///
    /// Called on the audio tap thread for each captured buffer. Implementations
    /// must be safe to call from a non-main thread.
    func append(_ buffer: AVAudioPCMBuffer)

    /// Signal end of audio and return the final transcription.
    ///
    /// - Returns: The transcribed text, or nil if transcription failed or was empty.
    func finishAndTranscribe() async -> String?

    /// Cancel any in-progress transcription and release resources.
    func cancel()
}

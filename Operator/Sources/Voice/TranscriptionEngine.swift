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
    ///
    /// - Parameter contextualStrings: Domain-specific terms (session names, project names)
    ///   that bias the recognizer toward expected vocabulary.
    /// - Throws: If the engine cannot be prepared (e.g. recognizer unavailable).
    func prepare(contextualStrings: [String]) throws

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

/// Placeholder engine used during startup while WhisperKit downloads/loads.
///
/// Returns nil for all transcriptions. The StateMachine plays an error tone
/// if the user tries push-to-talk before the real engine is ready.
public final class PendingEngine: TranscriptionEngine, @unchecked Sendable {
    /// Creates a new pending engine.
    public init() {}

    /// No-op — pending engine does not prepare.
    public func prepare(contextualStrings: [String]) throws {}

    /// No-op — pending engine discards audio.
    public func append(_ buffer: AVAudioPCMBuffer) {}

    /// Always returns nil — no transcription until WhisperKit loads.
    public func finishAndTranscribe() async -> String? { nil }

    /// No-op — nothing to cancel.
    public func cancel() {}
}

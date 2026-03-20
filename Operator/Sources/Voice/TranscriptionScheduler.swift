/// Protocol controlling when background transcriptions fire.
///
/// Implementations decide the pacing strategy; the coordinator
/// just calls ``start(engine:)`` and ``stop()``.
public protocol TranscriptionScheduler: Sendable {
    /// Start scheduling background transcriptions.
    /// Must not call transcribeWorkingWindow() concurrently -- serial only.
    func start(engine: SchedulableEngine) async

    /// Stop scheduling and wait for any in-flight transcription to complete.
    ///
    /// **Correctness invariant:** After stop() returns, no further calls
    /// to transcribeWorkingWindow() will occur. The coordinator can then
    /// safely access TranscriptionSession without synchronization.
    func stop() async
}

/// The interface the scheduler uses to trigger transcriptions.
public protocol SchedulableEngine: AnyObject, Sendable {
    /// Current number of accumulated audio samples.
    var sampleCount: Int { get }

    /// Transcribe the current working window.
    /// - Parameter maxSampleCount: If provided, clamp the audio
    ///   snapshot to this many samples. Used by test schedulers
    ///   to reproduce exact production conditions.
    func transcribeWorkingWindow(maxSampleCount: Int?) async
}

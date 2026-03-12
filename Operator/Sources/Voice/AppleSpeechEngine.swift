import AVFoundation
import Speech
import os

/// Apple Speech framework implementation of ``TranscriptionEngine``.
///
/// Uses SFSpeechRecognizer with on-device recognition (macOS 15+) for
/// privacy-preserving speech-to-text. Audio buffers are streamed to
/// the recognizer via SFSpeechAudioBufferRecognitionRequest.
///
/// Stall detection: partial results are enabled to detect liveness. If no
/// callback fires within 3 seconds of `endAudio()`, the recognition is
/// considered stalled and returns nil immediately. The caller (SpeechTranscriber)
/// handles retry with a fresh session.
public final class AppleSpeechEngine: TranscriptionEngine, @unchecked Sendable {
    // MARK: - Type Properties

    private static let logger = Log.logger(for: "AppleSpeechEngine")

    /// Seconds to wait after `endAudio()` before declaring a stall.
    ///
    /// On-device recognition typically responds within 1-2s for normal speech.
    private static let stallTimeout: TimeInterval = 3

    // MARK: - Instance Properties

    private let recognizer: SFSpeechRecognizer

    // Mutable state guarded by @MainActor callers for prepare/cancel,
    // and by the lock for the recognition callback thread.
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?

    // MARK: - Initialization

    /// Creates the engine with the given locale.
    ///
    /// - Parameter locale: The locale for speech recognition (default: en-US).
    public init(locale: Locale = Locale(identifier: "en-US")) {
        guard let speechRecognizer = SFSpeechRecognizer(locale: locale) else {
            fatalError("SFSpeechRecognizer could not be created for \(locale.identifier)")
        }
        self.recognizer = speechRecognizer

        if #available(macOS 15, *) {
            if speechRecognizer.supportsOnDeviceRecognition {
                Self.logger.info("On-device speech recognition is supported")
            } else {
                Self.logger.warning("On-device speech recognition not supported on this hardware")
            }
        }
    }

    // MARK: - Type Methods

    /// Request speech recognition authorization from the user.
    ///
    /// Must be called before use. The authorization prompt is shown once;
    /// subsequent calls return the cached status.
    public static func requestAuthorization() async -> SFSpeechRecognizerAuthorizationStatus {
        await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status)
            }
        }
    }

    /// Resume the continuation exactly once, guarded by the shared state lock.
    private static func resumeOnce(
        state: OSAllocatedUnfairLock<RecognitionState>,
        continuation: CheckedContinuation<String?, Never>,
        returning value: String?
    ) {
        let alreadyResumed = state.withLock { locked -> Bool in
            if locked.resumed {
                return true
            }
            locked.resumed = true
            return false
        }
        guard !alreadyResumed
        else { return }
        continuation.resume(returning: value)
    }

    // MARK: - Instance Methods

    /// Allocate a new recognition request for the upcoming transcription session.
    public func prepare() throws {
        cancel()

        guard recognizer.isAvailable else {
            Self.logger.error("Speech recognizer is not available")
            throw TranscriptionError.recognizerUnavailable
        }

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true

        if #available(macOS 15, *) {
            request.requiresOnDeviceRecognition = true
        }

        self.recognitionRequest = request
    }

    /// Forward an audio buffer to the speech recognition request.
    public func append(_ buffer: AVAudioPCMBuffer) {
        recognitionRequest?.append(buffer)
    }

    /// Signal end of audio input and return the recognized text.
    public func finishAndTranscribe() async -> String? {
        guard let request = recognitionRequest else {
            Self.logger.error("No recognition request available")
            return nil
        }

        let result = await performRecognition(with: request)

        recognitionRequest = nil
        recognitionTask = nil

        return result
    }

    /// Cancel the in-progress recognition task and release the request.
    public func cancel() {
        if let task = recognitionTask {
            task.cancel()
            recognitionTask = nil
            Self.logger.debug("Cancelled existing recognition task")
        }
        recognitionRequest = nil
    }

    // MARK: - Private

    private func performRecognition(
        with request: SFSpeechAudioBufferRecognitionRequest
    ) async -> String? {
        // Shared state between recognition callback and stall timer.
        // - resumed: ensures the continuation is resumed exactly once
        // - bestPartial: tracks the best partial result in case final never arrives
        // - lastCallbackTime: reset by each callback to detect stalls
        let state = OSAllocatedUnfairLock(
            initialState: RecognitionState()
        )

        return await withCheckedContinuation { (continuation: CheckedContinuation<String?, Never>) in
            self.recognitionTask = self.recognizer.recognitionTask(with: request) { result, error in
                state.withLock { $0.lastCallbackTime = CFAbsoluteTimeGetCurrent() }

                if let result {
                    if result.isFinal {
                        let text = result.bestTranscription.formattedString
                        Self.logger.info("Transcription complete: \(text)")
                        Self.resumeOnce(
                            state: state,
                            continuation: continuation,
                            returning: text.isEmpty ? nil : text
                        )
                    } else {
                        // Partial result — track it as a fallback.
                        let partial = result.bestTranscription.formattedString
                        state.withLock { $0.bestPartial = partial }
                    }
                } else if let error {
                    Self.logger.error("Transcription error: \(error.localizedDescription, privacy: .public)")
                    Self.resumeOnce(state: state, continuation: continuation, returning: nil)
                }
            }

            // Signal end of audio AFTER recognition task is running.
            request.endAudio()
            state.withLock { $0.lastCallbackTime = CFAbsoluteTimeGetCurrent() }

            // Stall detection: poll every second, checking if the recognizer
            // has gone silent for longer than stallTimeout.
            self.scheduleStallDetection(state: state, continuation: continuation)
        }
    }

    /// Schedule periodic stall checks.
    ///
    /// If the recognizer hasn't called back within `stallTimeout`, resume
    /// with the best partial result (or nil).
    private func scheduleStallDetection(
        state: OSAllocatedUnfairLock<RecognitionState>,
        continuation: CheckedContinuation<String?, Never>
    ) {
        DispatchQueue.global().asyncAfter(deadline: .now() + 1) { [weak self] in
            guard let self
            else { return }

            let stallCheck = state.withLock { locked -> StallCheckResult in
                let elapsed = CFAbsoluteTimeGetCurrent() - locked.lastCallbackTime
                return StallCheckResult(
                    isResumed: locked.resumed,
                    elapsed: elapsed,
                    partial: locked.bestPartial
                )
            }

            guard !stallCheck.isResumed
            else { return }

            if stallCheck.elapsed >= Self.stallTimeout {
                Self.logger.error(
                    "Recognizer stalled (\(String(format: "%.1f", stallCheck.elapsed))s since last callback)"
                )
                self.recognitionTask?.cancel()

                // If we got partial results before the stall, use the best one.
                if let partial = stallCheck.partial, !partial.isEmpty {
                    Self.logger.info("Returning best partial result: \"\(partial.prefix(60))\"")
                    Self.resumeOnce(state: state, continuation: continuation, returning: partial)
                } else {
                    Self.resumeOnce(state: state, continuation: continuation, returning: nil)
                }
            } else {
                // Not stalled yet — check again in 1s.
                self.scheduleStallDetection(state: state, continuation: continuation)
            }
        }
    }
}

// MARK: - Supporting Types

/// Mutable state shared between the recognition callback and stall detector.
private struct RecognitionState: Sendable {
    var resumed = false
    var bestPartial: String?
    var lastCallbackTime: CFAbsoluteTime = CFAbsoluteTimeGetCurrent()
}

/// Result of checking stall detection state.
private struct StallCheckResult {
    let isResumed: Bool
    let elapsed: TimeInterval
    let partial: String?
}

/// Errors from the transcription engine.
public enum TranscriptionError: Error, CustomStringConvertible {
    case recognizerUnavailable

    /// Human-readable description of the error.
    public var description: String {
        switch self {
        case .recognizerUnavailable:
            return "Speech recognizer is not available"
        }
    }
}

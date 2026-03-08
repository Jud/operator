import AVFoundation
import Speech
import os

/// Apple Speech framework implementation of ``TranscriptionEngine``.
///
/// Uses SFSpeechRecognizer with on-device recognition (macOS 15+) for
/// privacy-preserving speech-to-text. Audio buffers are streamed to
/// the recognizer via SFSpeechAudioBufferRecognitionRequest.
public final class AppleSpeechEngine: TranscriptionEngine, @unchecked Sendable {
    private static let logger = Log.logger(for: "AppleSpeechEngine")

    private let recognizer: SFSpeechRecognizer

    // Mutable state guarded by @MainActor callers for prepare/cancel,
    // and by the lock for the recognition callback thread.
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?

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

    /// Allocate a new recognition request for the upcoming transcription session.
    public func prepare() throws {
        cancel()

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = false

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

        // Start the recognition task BEFORE calling endAudio().
        // The recognizer must be actively processing when the stream ends;
        // calling endAudio() first produces an immediate error because the
        // recognizer sees a closed stream with no active task.
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
        await withTaskCancellationHandler {
            await withCheckedContinuation { (continuation: CheckedContinuation<String?, Never>) in
                let resumed = OSAllocatedUnfairLock(initialState: false)

                self.recognitionTask = self.recognizer.recognitionTask(with: request) { result, error in
                    let alreadyResumed = resumed.withLock { value -> Bool in
                        if value {
                            return true
                        }
                        value = true
                        return false
                    }
                    guard !alreadyResumed else {
                        return
                    }

                    if let result, result.isFinal {
                        let text = result.bestTranscription.formattedString
                        Self.logger.info("Transcription complete: \(text)")
                        continuation.resume(returning: text.isEmpty ? nil : text)
                    } else if let error {
                        Self.logger.error("Transcription error: \(error.localizedDescription, privacy: .public)")
                        continuation.resume(returning: nil)
                    }
                }

                // Signal end of audio AFTER recognition task is running.
                request.endAudio()
            }
        } onCancel: {
            Self.logger.warning("Transcription cancelled (likely timeout)")
            self.recognitionTask?.cancel()
        }
    }
}

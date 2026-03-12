import AVFoundation
import Speech
import os

/// Apple Speech framework implementation of ``TranscriptionEngine``.
///
/// Uses SFSpeechRecognizer with on-device recognition (macOS 15+) for
/// privacy-preserving speech-to-text.
///
/// The recognition task starts in `prepare()` so audio is processed
/// incrementally as buffers arrive via `append(_:)`. By the time
/// `finishAndTranscribe()` calls `endAudio()`, the recognizer has
/// already been tracking speech and typically produces the final result
/// within 0.5–1s.
///
/// If no partial results arrived during listening (no speech detected),
/// the engine returns nil after a brief 1-second window. If partials
/// were seen but the final result doesn't arrive within 3 seconds, the
/// best partial is returned as a fallback.
public final class AppleSpeechEngine: TranscriptionEngine, @unchecked Sendable {
    // MARK: - Type Properties

    private static let logger = Log.logger(for: "AppleSpeechEngine")

    /// Seconds to wait after `endAudio()` when partials were received.
    private static let finalResultTimeout: TimeInterval = 3

    /// Seconds to wait after `endAudio()` when no partials arrived (no speech).
    private static let noSpeechTimeout: TimeInterval = 1

    // MARK: - Instance Properties

    private let recognizer: SFSpeechRecognizer

    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private var eventContinuation: AsyncStream<RecognitionEvent>.Continuation?
    private var eventStream: AsyncStream<RecognitionEvent>?

    /// Count of partial results received during the listening phase.
    ///
    /// Thread-safe via lock; incremented from the recognition callback,
    /// read from @MainActor in `finishAndTranscribe()`.
    private let partialCount = OSAllocatedUnfairLock(initialState: 0)

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

    // MARK: - Instance Methods

    /// Prepare the engine and start the recognition task for real-time processing.
    ///
    /// The recognition task begins immediately so audio buffers are processed
    /// as they arrive, rather than batch-processing after capture ends.
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

        let (stream, continuation) = AsyncStream<RecognitionEvent>.makeStream()
        self.eventStream = stream
        self.eventContinuation = continuation
        self.recognitionRequest = request
        partialCount.withLock { $0 = 0 }

        startRecognitionTask(request: request, continuation: continuation)
        Self.logger.debug("Recognition task started (real-time processing)")
    }

    /// Forward an audio buffer to the speech recognition request.
    public func append(_ buffer: AVAudioPCMBuffer) {
        recognitionRequest?.append(buffer)
    }

    /// Signal end of audio input and return the recognized text.
    ///
    /// Calls `endAudio()` on the request and waits for the final result.
    /// Since the recognition task has been processing audio in real-time,
    /// the final result typically arrives within 0.5–1s.
    public func finishAndTranscribe() async -> String? {
        guard let request = recognitionRequest, let stream = eventStream
        else {
            Self.logger.error("No recognition session active")
            return nil
        }

        request.endAudio()

        let hadPartials = partialCount.withLock { $0 > 0 }
        let timeout = hadPartials ? Self.finalResultTimeout : Self.noSpeechTimeout

        if !hadPartials {
            Self.logger.info("No partials during listening — likely no speech detected")
        }

        let result = await consumeWithTimeout(stream: stream, timeout: timeout)
        resetSession()
        return result
    }

    /// Cancel any in-progress recognition task and release resources.
    public func cancel() {
        resetSession()
        partialCount.withLock { $0 = 0 }
    }

    // MARK: - Private

    /// Tear down the current recognition session and release all resources.
    private func resetSession() {
        if let task = recognitionTask {
            task.cancel()
            recognitionTask = nil
            Self.logger.debug("Cancelled recognition task")
        }
        eventContinuation?.finish()
        eventContinuation = nil
        eventStream = nil
        recognitionRequest = nil
    }

    /// Start the recognition task and wire its callback to the AsyncStream.
    private func startRecognitionTask(
        request: SFSpeechAudioBufferRecognitionRequest,
        continuation: AsyncStream<RecognitionEvent>.Continuation
    ) {
        let partials = partialCount

        self.recognitionTask = recognizer.recognitionTask(with: request) { result, error in
            if let result {
                if result.isFinal {
                    let text = result.bestTranscription.formattedString
                    Self.logger.info("Transcription complete: \(text)")
                    continuation.yield(.final(text.isEmpty ? nil : text))
                    continuation.finish()
                } else {
                    let partial = result.bestTranscription.formattedString
                    let count = partials.withLock { value -> Int in
                        value += 1
                        return value
                    }
                    if count <= 3 {
                        Self.logger.debug("Partial #\(count): \"\(partial.prefix(40))\"")
                    }
                    continuation.yield(.partial(partial))
                }
            } else if let error {
                Self.logger.error(
                    "Recognition error: \(error.localizedDescription, privacy: .public)"
                )
                continuation.yield(.failed)
                continuation.finish()
            }
        }
    }

    /// Consume the event stream until the final result arrives or the timeout fires.
    ///
    /// On timeout, cancels the consumer task. The `for await` loop exits and
    /// the best partial result seen so far is returned.
    private func consumeWithTimeout(
        stream: AsyncStream<RecognitionEvent>,
        timeout: TimeInterval
    ) async -> String? {
        let consumeTask = Task { () -> String? in
            var bestPartial: String?
            for await event in stream {
                switch event {
                case .partial(let text):
                    bestPartial = text

                case .final(let text):
                    return text

                case .failed:
                    return bestPartial
                }
            }
            // Stream ended without a final result.
            return bestPartial
        }

        let timeoutTask = Task {
            try? await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
            guard !Task.isCancelled
            else { return }
            consumeTask.cancel()
            Self.logger.warning("Recognition timed out after \(timeout)s")
        }

        let result = await consumeTask.value
        timeoutTask.cancel()
        return result
    }

    // MARK: - Deinitialization

    deinit {
        eventContinuation?.finish()
        recognitionTask?.cancel()
    }
}

// MARK: - Supporting Types

/// Events emitted by the recognition callback via AsyncStream.
private enum RecognitionEvent: Sendable {
    /// A partial transcription result (speech detected, not yet final).
    case partial(String)

    /// The final transcription result. Nil if the transcription was empty.
    case final(String?)

    /// An error occurred during recognition.
    case failed
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

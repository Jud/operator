import AVFoundation
import OSLog
import Speech

/// Wraps SFSpeechRecognizer and AVAudioEngine for on-device speech-to-text.
///
/// Designed for push-to-talk: call startListening() when the user presses the key,
/// and stopListening() when they release. The recognizer accumulates audio buffers
/// during the listening period and returns the final transcription on stop.
///
/// On-device recognition is required on macOS 15+ for privacy (no audio leaves the device).
/// The recognizer uses the native recording format from AVAudioEngine's input node,
/// avoiding any format conversion overhead.
///
/// Concurrency: This class is not actor-isolated. It is designed to be called from
/// @MainActor (via the StateMachine), and its callbacks dispatch results back.
/// The AVAudioEngine tap runs on its own thread; the recognition task callback
/// runs on an internal Speech framework thread.
public final class SpeechTranscriber: SpeechTranscribing, @unchecked Sendable {
    private static let logger = Logger(subsystem: "com.operator.app", category: "SpeechTranscriber")

    private let recognizer: SFSpeechRecognizer
    private let audioEngine = AVAudioEngine()

    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?

    /// Whether the audio engine is currently capturing microphone input.
    public private(set) var isListening = false

    /// Creates a new speech transcriber with on-device recognition.
    public init() {
        guard let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US")) else {
            fatalError("SFSpeechRecognizer could not be created for en-US locale")
        }
        self.recognizer = speechRecognizer

        // On-device recognition keeps voice data private (no cloud transcription).
        // macOS 15+ supports this; the package already targets macOS 15+,
        // but we use #available as the spec requires explicit gating.
        if #available(macOS 15, *) {
            if self.recognizer.supportsOnDeviceRecognition {
                Self.logger.info("On-device speech recognition is supported")
            } else {
                Self.logger.warning("On-device speech recognition not supported on this hardware")
            }
        }
    }

    /// Request speech recognition authorization from the user.
    ///
    /// Must be called before startListening(). The authorization prompt is shown
    /// once; subsequent calls return the cached status.
    public static func requestAuthorization() async -> SFSpeechRecognizerAuthorizationStatus {
        await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status)
            }
        }
    }

    /// Begin capturing audio from the microphone and feeding it to the speech recognizer.
    ///
    /// - Throws: If the audio engine fails to start (e.g., no microphone access).
    public func startListening() throws {
        cancelExistingTask()

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = false

        if #available(macOS 15, *) {
            request.requiresOnDeviceRecognition = true
        }

        self.recognitionRequest = request

        let inputNode = audioEngine.inputNode

        let recordingFormat = inputNode.outputFormat(forBus: 0)

        inputNode.installTap(onBus: 0, bufferSize: 1_024, format: recordingFormat) { [weak self] buffer, _ in
            self?.recognitionRequest?.append(buffer)
        }

        audioEngine.prepare()
        try audioEngine.start()
        isListening = true
        Self.logger.info("Audio engine started, listening for speech")
    }

    /// Stop capturing audio and return the final transcription.
    ///
    /// - Returns: The transcribed text, or nil if transcription failed or was empty.
    public func stopListening() async -> String? {
        guard isListening else {
            Self.logger.warning("stopListening called but not currently listening")
            return nil
        }

        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        recognitionRequest?.endAudio()
        isListening = false
        Self.logger.info("Audio engine stopped, processing transcription")

        guard let request = recognitionRequest else {
            Self.logger.error("No recognition request available")
            return nil
        }

        let result = await performRecognition(with: request)

        self.recognitionRequest = nil
        self.recognitionTask = nil

        return result
    }

    /// Run speech recognition and return the transcribed text.
    private func performRecognition(with request: SFSpeechAudioBufferRecognitionRequest) async -> String? {
        await withTaskCancellationHandler {
            await withCheckedContinuation { (continuation: CheckedContinuation<String?, Never>) in
                let resumed = UnsafeMutablePointer<Bool>.allocate(capacity: 1)
                resumed.initialize(to: false)

                self.recognitionTask = self.recognizer.recognitionTask(with: request) { result, error in
                    guard !resumed.pointee else {
                        return
                    }
                    if let result, result.isFinal {
                        resumed.pointee = true
                        let text = result.bestTranscription.formattedString
                        Self.logger.info("Transcription complete: \(text)")
                        continuation.resume(returning: text.isEmpty ? nil : text)
                        resumed.deallocate()
                    } else if let error {
                        resumed.pointee = true
                        Self.logger.error("Transcription error: \(error.localizedDescription)")
                        continuation.resume(returning: nil)
                        resumed.deallocate()
                    }
                }
            }
        } onCancel: {
            Self.logger.warning("Transcription cancelled (likely timeout)")
            self.recognitionTask?.cancel()
        }
    }

    /// Perform transcription with a 30-second timeout.
    ///
    /// - Returns: The transcribed text, or nil on timeout/failure/empty transcription.
    public func stopListeningWithTimeout(seconds: TimeInterval = 30) async -> String? {
        let transcriptionTask = Task { () -> String? in
            await self.stopListening()
        }

        let timeoutTask = Task {
            try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
            transcriptionTask.cancel()
            Self.logger.error("Transcription timed out after \(seconds) seconds")
        }

        let result = await transcriptionTask.value
        timeoutTask.cancel()
        return result
    }

    /// Cancel any in-flight recognition task and request.
    private func cancelExistingTask() {
        if let task = recognitionTask {
            task.cancel()
            recognitionTask = nil
            Self.logger.debug("Cancelled existing recognition task")
        }
        recognitionRequest = nil

        if audioEngine.isRunning {
            audioEngine.stop()
            audioEngine.inputNode.removeTap(onBus: 0)
            isListening = false
            Self.logger.debug("Stopped running audio engine from previous cycle")
        }
    }
}

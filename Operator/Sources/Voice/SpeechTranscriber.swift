import AVFoundation
import AudioToolbox
import CoreAudio

/// Wraps a ``TranscriptionEngine`` and AVAudioEngine for on-device speech-to-text.
///
/// Designed for push-to-talk: call startListening() when the user presses the key,
/// and stopListening() when they release. The engine accumulates audio buffers
/// during the listening period and returns the final transcription on stop.
///
/// Concurrency: @MainActor ensures all mutable state is accessed on a single
/// thread. The AVAudioEngine tap runs on its own thread but only calls into the
/// engine's append method, which is safe by TranscriptionEngine's contract.
@MainActor
public final class SpeechTranscriber: SpeechTranscribing {
    private static let logger = Log.logger(for: "SpeechTranscriber")

    private let engine: any TranscriptionEngine
    private let audioEngine = AVAudioEngine()

    /// Whether the audio engine is currently capturing microphone input.
    public private(set) var isListening = false

    /// Creates a new speech transcriber with the given transcription engine.
    ///
    /// - Parameter engine: The recognition backend (default: ``AppleSpeechEngine``).
    public init(engine: any TranscriptionEngine = AppleSpeechEngine()) {
        self.engine = engine
    }

    /// Begin capturing audio from the microphone and feeding it to the engine.
    ///
    /// - Throws: If the audio engine fails to start (e.g., no microphone access).
    public func startListening() throws {
        cancelExistingSession()

        try engine.prepare()

        applyInputDevice()

        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        let capturedEngine = engine

        inputNode.installTap(
            onBus: 0,
            bufferSize: 1_024,
            format: recordingFormat
        ) { @Sendable buffer, _ in
            capturedEngine.append(buffer)
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
        isListening = false
        Self.logger.info("Audio engine stopped, processing transcription")

        let result = await engine.finishAndTranscribe()
        return result
    }

    /// Perform transcription with a 30-second timeout.
    ///
    /// - Returns: The transcribed text, or nil on timeout/failure/empty transcription.
    public func stopListeningWithTimeout(seconds: TimeInterval = 30) async -> String? {
        let transcriptionTask = Task { @MainActor () -> String? in
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

    /// Apply the user-selected input device to the audio engine, if configured.
    private func applyInputDevice() {
        let uid = UserDefaults.standard.string(forKey: "inputDeviceUID") ?? ""
        guard !uid.isEmpty, let deviceID = AudioDeviceManager.deviceID(forUID: uid) else {
            return
        }

        guard let audioUnit = audioEngine.inputNode.audioUnit else {
            Self.logger.warning("No audio unit on input node, cannot set device")
            return
        }
        var devID = deviceID
        let status = AudioUnitSetProperty(
            audioUnit,
            kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global,
            0,
            &devID,
            UInt32(MemoryLayout<AudioDeviceID>.size)
        )
        if status == noErr {
            Self.logger.info("Set input device to \(uid)")
        } else {
            Self.logger.warning("Failed to set input device (status: \(status)), using default")
        }
    }

    /// Cancel any in-flight session and stop audio capture.
    private func cancelExistingSession() {
        engine.cancel()

        if audioEngine.isRunning {
            audioEngine.stop()
            audioEngine.inputNode.removeTap(onBus: 0)
            isListening = false
            Self.logger.debug("Stopped running audio engine from previous cycle")
        }
    }
}

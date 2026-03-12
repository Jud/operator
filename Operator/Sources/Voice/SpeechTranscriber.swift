@preconcurrency import AVFoundation
import AudioToolbox
import os

/// Wraps a ``TranscriptionEngine`` and AVAudioEngine for on-device speech-to-text.
///
/// Designed for push-to-talk: call startListening() when the user presses the key,
/// and stopListening() when they release. The engine accumulates audio buffers
/// during the listening period and returns the final transcription on stop.
///
/// Audio is also accumulated in a side buffer and written to a WAV file in
/// `~/Library/Application Support/Operator/audio-traces/` for offline analysis.
///
/// Concurrency: @MainActor ensures all mutable state is accessed on a single
/// thread. The AVAudioEngine tap runs on its own thread but only calls into the
/// engine's append method, which is safe by TranscriptionEngine's contract.
@MainActor
public final class SpeechTranscriber: SpeechTranscribing {
    // MARK: - Type Properties

    private static let logger = Log.logger(for: "SpeechTranscriber")

    /// Directory for saved audio files.
    static let audioTraceDir: URL = {
        let appSupport =
            FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support")
        return appSupport.appendingPathComponent("Operator/audio-traces", isDirectory: true)
    }()

    // MARK: - Instance Properties

    private let engine: any TranscriptionEngine
    private let audioEngine = AVAudioEngine()

    /// Whether the audio engine is currently capturing microphone input.
    public private(set) var isListening = false

    /// URL of the WAV file saved from the most recent recording session.
    public private(set) var lastAudioFileURL: URL?

    /// Lock-guarded buffer accumulator shared with the audio tap closure.
    ///
    /// Uses OSAllocatedUnfairLock for async-safe access.
    private let capturedBuffers = OSAllocatedUnfairLock(initialState: [AVAudioPCMBuffer]())

    /// Format of the captured audio (set when the tap is installed).
    private var capturedFormat: AVAudioFormat?

    // MARK: - Initialization

    /// Creates a new speech transcriber with the given transcription engine.
    ///
    /// - Parameter engine: The recognition backend (default: ``AppleSpeechEngine``).
    public init(engine: any TranscriptionEngine = AppleSpeechEngine()) {
        self.engine = engine
    }

    // MARK: - Type Methods

    /// Create a deep copy of an AVAudioPCMBuffer.
    nonisolated static func copyBuffer(_ source: AVAudioPCMBuffer) -> AVAudioPCMBuffer? {
        guard
            let copy = AVAudioPCMBuffer(
                pcmFormat: source.format,
                frameCapacity: source.frameLength
            )
        else {
            return nil
        }
        copy.frameLength = source.frameLength

        let channelCount = Int(source.format.channelCount)
        if let srcFloats = source.floatChannelData, let dstFloats = copy.floatChannelData {
            let frames = Int(source.frameLength)
            for ch in 0..<channelCount {
                dstFloats[ch].update(from: srcFloats[ch], count: frames)
            }
        }
        return copy
    }

    // MARK: - Public Methods

    /// Begin capturing audio from the microphone and feeding it to the engine.
    ///
    /// - Throws: If the audio engine fails to start (e.g., no microphone access).
    public func startListening() throws {
        cancelExistingSession()

        // Reset audio capture state.
        capturedBuffers.withLock { $0.removeAll() }
        lastAudioFileURL = nil

        try engine.prepare()

        applyInputDevice()

        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        capturedFormat = recordingFormat
        let capturedEngine = engine
        let buffers = capturedBuffers

        inputNode.installTap(
            onBus: 0,
            bufferSize: 1_024,
            format: recordingFormat
        ) { @Sendable buffer, _ in
            capturedEngine.append(buffer)

            // Copy the buffer for WAV export.
            if let copy = Self.copyBuffer(buffer) {
                buffers.withLock { $0.append(copy) }
            }
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

        tearDownAudioCapture()
        Self.logger.info("Audio engine stopped, processing transcription")

        // Write WAV file from accumulated buffers.
        writeAudioFile()

        return await engine.finishAndTranscribe()
    }

    /// Perform transcription with a 30-second timeout.
    ///
    /// - Returns: The transcribed text, or nil on timeout/failure/empty transcription.
    public func stopListeningWithTimeout(seconds: TimeInterval = 30) async -> String? {
        let transcriptionTask = Task {
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

    // MARK: - Audio File Writing

    /// Write captured buffers to a timestamped WAV file.
    private func writeAudioFile() {
        let buffers = capturedBuffers.withLock { bufs -> [AVAudioPCMBuffer] in
            let copy = bufs
            bufs.removeAll()
            return copy
        }

        guard !buffers.isEmpty, let format = capturedFormat else {
            Self.logger.debug("No audio buffers to save")
            return
        }

        let dir = Self.audioTraceDir
        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        } catch {
            Self.logger.error("Failed to create audio trace directory: \(error)")
            return
        }

        let timestamp = ISO8601DateFormatter.string(
            from: Date(),
            timeZone: .current,
            formatOptions: [.withFullDate, .withFullTime, .withColonSeparatorInTime]
        )
        let safeTimestamp = timestamp.replacingOccurrences(of: ":", with: "-")
        let fileURL = dir.appendingPathComponent("\(safeTimestamp).wav")

        do {
            let audioFile = try AVAudioFile(
                forWriting: fileURL,
                settings: format.settings,
                commonFormat: format.commonFormat,
                interleaved: format.isInterleaved
            )
            for buffer in buffers {
                try audioFile.write(from: buffer)
            }
            lastAudioFileURL = fileURL
            Self.logger.info("Saved audio trace: \(fileURL.lastPathComponent)")
        } catch {
            Self.logger.error("Failed to write audio trace: \(error)")
        }
    }

    // MARK: - Private

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
            tearDownAudioCapture()
            Self.logger.debug("Stopped running audio engine from previous cycle")
        }
    }

    /// Stop the audio engine, remove the input tap, and clear the listening flag.
    private func tearDownAudioCapture() {
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        isListening = false
    }
}

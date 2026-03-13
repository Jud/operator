@preconcurrency import AVFoundation
import Accelerate
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

    /// Floor of the dB range mapped to 0.0 (below this is silence).
    nonisolated private static let dbFloor: Float = -50
    /// Width of the dB range mapped to 0.0-1.0.
    nonisolated private static let dbRange: Float = 40

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

    /// Audio level monitor for waveform visualization.
    private let levelMonitor: AudioLevelMonitor?

    // MARK: - Initialization

    /// Creates a new speech transcriber with the given transcription engine.
    ///
    /// - Parameters:
    ///   - engine: The recognition backend (default: ``AppleSpeechEngine``).
    ///   - levelMonitor: Optional audio level monitor for waveform visualization.
    public init(engine: any TranscriptionEngine = AppleSpeechEngine(), levelMonitor: AudioLevelMonitor? = nil) {
        self.engine = engine
        self.levelMonitor = levelMonitor
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

    /// Compute normalized RMS level (0.0-1.0) from an audio buffer using dB scaling.
    nonisolated static func computeRMS(_ buffer: AVAudioPCMBuffer) -> Float {
        guard let channelData = buffer.floatChannelData else {
            return 0
        }
        let frames = Int(buffer.frameLength)
        guard frames > 0 else {
            return 0
        }

        var sumSquares: Float = 0
        vDSP_svesq(channelData[0], 1, &sumSquares, vDSP_Length(frames))
        let rms = sqrt(sumSquares / Float(frames))

        let db = 20 * log10(max(rms, 1e-7))
        return max(0, min(1, (db - dbFloor) / dbRange))
    }

    /// Write captured buffers to a timestamped WAV file off the main actor.
    ///
    /// - Returns: The URL of the written file, or nil if nothing was written.
    nonisolated private static func writeAudioFile(
        buffers: [AVAudioPCMBuffer],
        format: AVAudioFormat?,
        directory: URL
    ) async -> URL? {
        guard !buffers.isEmpty, let format
        else { return nil }

        let log = Log.logger(for: "SpeechTranscriber")

        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        } catch {
            log.error("Failed to create audio trace directory: \(error)")
            return nil
        }

        let timestamp = ISO8601DateFormatter.string(
            from: Date(),
            timeZone: .current,
            formatOptions: [.withFullDate, .withFullTime, .withColonSeparatorInTime]
        )
        let safeTimestamp = timestamp.replacingOccurrences(of: ":", with: "-")
        let fileURL = directory.appendingPathComponent("\(safeTimestamp).wav")

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
            log.info("Saved audio trace: \(fileURL.lastPathComponent)")
            return fileURL
        } catch {
            log.error("Failed to write audio trace: \(error)")
            return nil
        }
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
        levelMonitor?.reset()

        try engine.prepare()

        applyInputDevice()

        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        capturedFormat = recordingFormat
        let capturedEngine = engine
        let buffers = capturedBuffers
        let monitor = levelMonitor

        inputNode.installTap(
            onBus: 0,
            bufferSize: 1_024,
            format: recordingFormat
        ) { @Sendable buffer, _ in
            capturedEngine.append(buffer)

            // Push audio level for waveform visualization.
            if let monitor {
                monitor.push(Self.computeRMS(buffer))
            }

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

        let result = await engine.finishAndTranscribe()

        let buffers = capturedBuffers.withLock { bufs -> [AVAudioPCMBuffer] in
            let snapshot = bufs
            bufs.removeAll()
            return snapshot
        }
        let format = capturedFormat
        capturedFormat = nil

        lastAudioFileURL = await Self.writeAudioFile(
            buffers: buffers,
            format: format,
            directory: Self.audioTraceDir
        )
        return result
    }

    /// Perform transcription with a timeout.
    ///
    /// The engine's internal stall detection (1–3s) handles timeouts, so this
    /// simply delegates to ``stopListening()``. The `seconds` parameter is
    /// retained for protocol conformance.
    ///
    /// - Returns: The transcribed text, or nil on timeout/failure/empty transcription.
    public func stopListeningWithTimeout(seconds _: TimeInterval = 30) async -> String? {
        await stopListening()
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
        levelMonitor?.reset()
    }
}

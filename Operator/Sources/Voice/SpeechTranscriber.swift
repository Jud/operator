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

    /// Directory for saved audio files.
    static let audioTraceDir: URL = {
        let appSupport =
            FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support")
        return appSupport.appendingPathComponent("Operator/audio-traces", isDirectory: true)
    }()

    // MARK: - Instance Properties

    private var engine: any TranscriptionEngine
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
            pruneOldTraces(in: directory, keep: 100, log: log)
            return fileURL
        } catch {
            log.error("Failed to write audio trace: \(error)")
            return nil
        }
    }

    /// Keep only the most recent `keep` audio trace files, deleting the rest.
    nonisolated private static func pruneOldTraces(
        in directory: URL,
        keep: Int,
        log: Logger
    ) {
        let fm = FileManager.default
        guard
            let files = try? fm.contentsOfDirectory(
                at: directory,
                includingPropertiesForKeys: [.creationDateKey],
                options: [.skipsHiddenFiles]
            )
        else { return }

        let wavFiles = files.filter { $0.pathExtension == "wav" }
        guard wavFiles.count > keep
        else { return }

        let sorted = wavFiles.sorted { lhs, rhs in
            let dateL = (try? lhs.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
            let dateR = (try? rhs.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
            return dateL < dateR
        }

        let toDelete = sorted.prefix(sorted.count - keep)
        for file in toDelete {
            try? fm.removeItem(at: file)
        }
        log.info("Pruned \(toDelete.count) old audio trace(s), keeping \(keep)")
    }

    // MARK: - Public Methods

    /// Replace the transcription engine (e.g. after async model loading).
    ///
    /// Safe to call between push-to-talk sessions. If called while listening,
    /// the new engine takes effect on the next `startListening()` call.
    public func replaceEngine(_ newEngine: any TranscriptionEngine) {
        engine.cancel()
        engine = newEngine
        Self.logger.info("Transcription engine replaced")
    }

    /// Begin capturing audio from the microphone and feeding it to the engine.
    ///
    /// - Throws: If the audio engine fails to start (e.g., no microphone access).
    public func startListening(contextualStrings: [String] = []) throws {
        cancelExistingSession()

        // Reset audio capture state.
        capturedBuffers.withLock { $0.removeAll() }
        lastAudioFileURL = nil
        levelMonitor?.reset()

        try engine.prepare(contextualStrings: contextualStrings)

        applyInputDevice()

        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        capturedFormat = recordingFormat
        let capturedEngine = engine
        let buffers = capturedBuffers
        let monitor = levelMonitor
        let analyzer: SpectrumAnalyzer? =
            monitor != nil
            ? SpectrumAnalyzer(
                frameCount: 1_024,
                sampleRate: Float(recordingFormat.sampleRate)
            ) : nil

        inputNode.installTap(
            onBus: 0,
            bufferSize: 1_024,
            format: recordingFormat
        ) { @Sendable buffer, _ in
            // Deep-copy first — AVAudioEngine recycles buffer memory after tap returns.
            // Share the copy between the recognizer and WAV export.
            if let copy = Self.copyBuffer(buffer) {
                capturedEngine.append(copy)
                buffers.withLock { $0.append(copy) }
            }

            // Analyzer reads original buffer (valid for the tap callback's duration).
            if let monitor, let analyzer {
                let bands = analyzer.analyze(buffer)
                monitor.pushBands(bands)
            }
        }

        audioEngine.prepare()
        try audioEngine.start()
        isListening = true
        Self.logger.info("Audio engine started, listening for speech")
    }

    /// RMS threshold below which audio is considered silence.
    ///
    /// Typical speech RMS is 0.02–0.2; system tones and background noise sit
    /// well below 0.01. This prevents WhisperKit from hallucinating on feedback
    /// tones or ambient room noise.
    private static let silenceThreshold: Float = 0.008

    /// Seconds of audio to skip at the start when computing RMS.
    ///
    /// The feedback tone (listening cue) plays when recording starts and gets
    /// picked up by the microphone. Skipping this window prevents the tone from
    /// inflating the RMS and fooling silence detection.
    private static let feedbackToneSkipSeconds: Float = 0.4

    /// Stop capturing audio and return the final transcription.
    ///
    /// Returns nil early if the captured audio is below the silence threshold,
    /// skipping the transcription engine entirely.
    ///
    /// - Returns: The transcribed text, or nil if transcription failed or was empty.
    public func stopListening() async -> String? {
        guard isListening else {
            Self.logger.warning("stopListening called but not currently listening")
            return nil
        }

        tearDownAudioCapture()
        Self.logger.info("Audio engine stopped, processing transcription")

        let buffers = capturedBuffers.withLock { bufs -> [AVAudioPCMBuffer] in
            let snapshot = bufs
            bufs.removeAll()
            return snapshot
        }
        let format = capturedFormat
        capturedFormat = nil

        let rms = Self.computeRMS(buffers, skippingSeconds: Self.feedbackToneSkipSeconds)
        if rms < Self.silenceThreshold {
            Self.logger.info("Silence detected (RMS=\(rms, format: .fixed(precision: 4))); skipping transcription")
            engine.cancel()
            nonisolated(unsafe) let traceBuffers = buffers
            nonisolated(unsafe) let traceFormat = format
            lastAudioFileURL = await Self.writeAudioFile(
                buffers: traceBuffers,
                format: traceFormat,
                directory: Self.audioTraceDir
            )
            return nil
        }

        let result = await engine.finishAndTranscribe()

        nonisolated(unsafe) let traceBuffers = buffers
        nonisolated(unsafe) let traceFormat = format
        lastAudioFileURL = await Self.writeAudioFile(
            buffers: traceBuffers,
            format: traceFormat,
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

    /// Compute the RMS (root mean square) energy of captured audio buffers,
    /// skipping the initial feedback tone window.
    ///
    /// Measures overall loudness after the skip window. Returns 0 if no
    /// samples remain after skipping.
    nonisolated private static func computeRMS(
        _ buffers: [AVAudioPCMBuffer],
        skippingSeconds: Float = 0.4
    ) -> Float {
        guard let sampleRate = buffers.first?.format.sampleRate else {
            return 0
        }
        var framesToSkip = Int(Float(sampleRate) * skippingSeconds)
        var sumSquares: Double = 0
        var totalFrames: Int = 0

        for buffer in buffers {
            guard let channelData = buffer.floatChannelData else {
                continue
            }
            let frames = Int(buffer.frameLength)
            let samples = channelData[0]

            let start: Int
            if framesToSkip >= frames {
                framesToSkip -= frames
                continue
            } else {
                start = framesToSkip
                framesToSkip = 0
            }

            for i in start..<frames {
                let sample = Double(samples[i])
                sumSquares += sample * sample
            }
            totalFrames += frames - start
        }

        guard totalFrames > 0 else {
            return 0
        }
        return Float((sumSquares / Double(totalFrames)).squareRoot())
    }

    /// Stop the audio engine, remove the input tap, and clear the listening flag.
    private func tearDownAudioCapture() {
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        isListening = false
        levelMonitor?.reset()
    }
}

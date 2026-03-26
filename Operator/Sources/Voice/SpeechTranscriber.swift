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

    /// Pre-warm the AVAudioEngine so the first real activation captures audio immediately.
    ///
    /// On cold launch, the first `audioEngine.start()` triggers HAL device negotiation
    /// which can take hundreds of milliseconds. During that window the tap receives no
    /// samples, causing the first recording to appear silent. A quick prepare/start/stop
    /// cycle at launch forces this negotiation to happen before the user presses the key.
    ///
    /// Applies the user's preferred input device (or the built-in mic) before starting
    /// so macOS doesn't try to activate a Bluetooth mic profile and cause an audio blip
    /// on wireless headphones.
    public func warmUp() {
        // Intentionally empty — accessing audioEngine.inputNode or calling
        // prepare() triggers a Bluetooth audio blip on wireless headphones.
        // The first real activation may capture silence; this is handled by
        // the too-short/silence detection in the StateMachine which returns
        // to IDLE cleanly, and the second activation works normally.
        Self.logger.info("Audio engine warm-up: skipped (Bluetooth blip avoidance)")
    }

    // MARK: - Type Methods

    /// Extract channel 0 from a multi-channel buffer into a new mono buffer.
    ///
    /// Used when voice processing outputs 3-channel audio — channel 0 is the
    /// echo-cancelled near-end speech.
    nonisolated static func extractChannel0(
        from source: AVAudioPCMBuffer,
        monoFormat: AVAudioFormat
    ) -> AVAudioPCMBuffer? {
        guard let srcChannels = source.floatChannelData
        else { return nil }
        let frames = source.frameLength
        guard
            let mono = AVAudioPCMBuffer(pcmFormat: monoFormat, frameCapacity: frames)
        else { return nil }
        mono.frameLength = frames
        guard let dstData = mono.floatChannelData
        else { return nil }
        dstData[0].update(from: srcChannels[0], count: Int(frames))
        return mono
    }

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

        let inputNode = audioEngine.inputNode

        // Voice processing (AEC) intentionally NOT used — it creates a
        // system-wide aggregate audio device that degrades other apps' audio
        // and forces Bluetooth into low-quality HFP mode. Push-to-talk with
        // mute-on-talk means the speaker is silent during recording, so
        // there is no echo to cancel. The feedback tone is handled by the
        // RMS skip window in computeRMS().

        applyInputDevice()

        let tapFormat = inputNode.outputFormat(forBus: 0)

        // Without voice processing the input is typically mono. Handle
        // multi-channel gracefully in case an external device provides it.
        let needsMonoExtract = tapFormat.channelCount > 1
        let monoFormat =
            needsMonoExtract
            ? (AVAudioFormat(
                commonFormat: tapFormat.commonFormat,
                sampleRate: tapFormat.sampleRate,
                channels: 1,
                interleaved: false
            ) ?? tapFormat)
            : tapFormat

        let extractNote = needsMonoExtract ? " -> mono extraction" : ""
        Self.logger.info(
            "Recording format: \(tapFormat.sampleRate)Hz, \(tapFormat.channelCount)ch\(extractNote)"
        )
        capturedFormat = monoFormat
        let capturedEngine = engine
        let buffers = capturedBuffers
        let monitor = levelMonitor
        let analyzer: SpectrumAnalyzer? =
            monitor != nil
            ? SpectrumAnalyzer(
                frameCount: 1_024,
                sampleRate: Float(monoFormat.sampleRate)
            ) : nil

        inputNode.installTap(
            onBus: 0,
            bufferSize: 1_024,
            format: tapFormat
        ) { @Sendable buffer, _ in
            let monoBuffer: AVAudioPCMBuffer
            if needsMonoExtract {
                guard let extracted = Self.extractChannel0(from: buffer, monoFormat: monoFormat)
                else { return }
                monoBuffer = extracted
            } else {
                guard let copy = Self.copyBuffer(buffer)
                else { return }
                monoBuffer = copy
            }

            capturedEngine.append(monoBuffer)
            buffers.withLock { $0.append(monoBuffer) }

            if let monitor, let analyzer {
                let bands = analyzer.analyze(monoBuffer)
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

    /// Captured buffers and format from the most recent stopAndCheckSilence() call.
    ///
    /// Held between stopAndCheckSilence() and finishTranscription() so the caller
    /// can play feedback tones in between.
    private var pendingBuffers: [AVAudioPCMBuffer] = []
    private var pendingFormat: AVAudioFormat?

    /// Stop audio capture and check whether the recording contains speech.
    ///
    /// Tears down the audio engine and computes RMS energy (skipping the initial
    /// feedback tone window). If the audio is silence, cancels the engine and
    /// writes the trace file immediately.
    ///
    /// - Returns: `true` if the audio contains speech and should be transcribed,
    ///   `false` if it was silence (already handled).
    public func stopAndCheckSilence() async -> Bool {
        guard isListening else {
            Self.logger.warning("stopAndCheckSilence called but not currently listening")
            return false
        }

        tearDownAudioCapture()
        Self.logger.info("Audio engine stopped, checking for speech")

        pendingBuffers = capturedBuffers.withLock { bufs -> [AVAudioPCMBuffer] in
            let snapshot = bufs
            bufs.removeAll()
            return snapshot
        }
        pendingFormat = capturedFormat
        capturedFormat = nil

        let rms = Self.computeRMS(pendingBuffers, skippingSeconds: Self.feedbackToneSkipSeconds)
        if rms < Self.silenceThreshold {
            Self.logger.info("Silence detected (RMS=\(rms, format: .fixed(precision: 4))); skipping transcription")
            engine.cancel()
            nonisolated(unsafe) let traceBuffers = pendingBuffers
            nonisolated(unsafe) let traceFormat = pendingFormat
            pendingBuffers = []
            pendingFormat = nil
            lastAudioFileURL = await Self.writeAudioFile(
                buffers: traceBuffers,
                format: traceFormat,
                directory: Self.audioTraceDir
            )
            return false
        }
        return true
    }

    /// Run the transcription engine on the audio captured by the last session.
    ///
    /// Must be called after ``stopAndCheckSilence()`` returns `true`.
    /// Writes the audio trace file after transcription completes.
    ///
    /// - Returns: The transcribed text, or nil if transcription failed or was empty.
    public func finishTranscription() async -> String? {
        let result = await engine.finishAndTranscribe()

        nonisolated(unsafe) let traceBuffers = pendingBuffers
        nonisolated(unsafe) let traceFormat = pendingFormat
        pendingBuffers = []
        pendingFormat = nil
        lastAudioFileURL = await Self.writeAudioFile(
            buffers: traceBuffers,
            format: traceFormat,
            directory: Self.audioTraceDir
        )
        return result
    }

    /// Stop capturing audio and return the final transcription.
    ///
    /// Convenience that combines ``stopAndCheckSilence()`` and ``finishTranscription()``.
    ///
    /// - Returns: The transcribed text, or nil if transcription failed or was empty.
    public func stopListening() async -> String? {
        let hasSpeech = await stopAndCheckSilence()
        guard hasSpeech
        else { return nil }
        return await finishTranscription()
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

    /// Apply the user-selected input device to the audio engine.
    ///
    /// If no preference is set, falls back to the built-in mic to avoid
    /// triggering a Bluetooth profile switch on wireless headphones.
    private func applyInputDevice() {
        let uid = UserDefaults.standard.string(forKey: "inputDeviceUID") ?? ""
        let deviceID: AudioDeviceID
        if !uid.isEmpty, let preferred = AudioDeviceManager.deviceID(forUID: uid) {
            deviceID = preferred
        } else if let builtIn = AudioDeviceManager.builtInMicID() {
            deviceID = builtIn
        } else {
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
            Self.logger.info("Set input device to \(uid.isEmpty ? "built-in mic" : uid)")
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
        var meanSquare: Float = 0
        var totalFrames: Int = 0

        for buffer in buffers {
            guard let channelData = buffer.floatChannelData else {
                continue
            }
            let frames = Int(buffer.frameLength)

            if framesToSkip >= frames {
                framesToSkip -= frames
                continue
            }

            let start = framesToSkip
            framesToSkip = 0
            let count = frames - start
            guard count > 0 else { continue }

            var bufferMeanSquare: Float = 0
            vDSP_measqv(channelData[0] + start, 1, &bufferMeanSquare, vDSP_Length(count))
            meanSquare += bufferMeanSquare * Float(count)
            totalFrames += count
        }

        guard totalFrames > 0 else {
            return 0
        }
        return (meanSquare / Float(totalFrames)).squareRoot()
    }

    /// Stop the audio engine, remove the input tap, and clear the listening flag.
    private func tearDownAudioCapture() {
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        isListening = false
        levelMonitor?.reset()
    }
}

import AVFoundation
import AudioToolbox
import KokoroCoreML

/// Central owner of all audio I/O.
///
/// ```
/// AudioHub
///  ├─ outputEngine (AVAudioEngine, permanent, starts at launch)
///  │   ├─ ttsPlayerNode (24kHz mono — Kokoro TTS)
///  │   └─ feedbackPlayerNode (44.1kHz mono — .caf tones)
///  └─ inputUnit (raw AUHAL, start/stop per push-to-talk)
///      └─ render callback → tap block
/// ```
///
/// The output side uses AVAudioEngine for convenient player node scheduling.
/// The input side uses a raw AUHAL AudioUnit configured as input-only with
/// output disabled BEFORE initialization. This prevents CoreAudio from
/// creating a `CADefaultDeviceAggregate` that would trigger Bluetooth
/// renegotiation and cause music playback to blip.
@MainActor
public final class AudioHub {
    private static let logger = Log.logger(for: "AudioHub")

    /// Called when the output engine restarts after a route change.
    ///
    /// AudioQueue should use this to recover from a stuck speaking state.
    public var onOutputReset: (() async -> Void)?

    // MARK: - Output Engine

    /// Persistent output engine for TTS and feedback tones.
    private let outputEngine = AVAudioEngine()

    /// Player node for Kokoro TTS output (24kHz mono float32).
    public let ttsPlayerNode = AVAudioPlayerNode()

    /// Player node for feedback tone output (44.1kHz mono).
    public let feedbackPlayerNode = AVAudioPlayerNode()

    // MARK: - Input (Raw AUHAL)

    /// The raw AUHAL audio unit for mic capture, created on first `startInput()`.
    ///
    /// Configured as input-only (output disabled before initialize) to avoid
    /// aggregate device creation. Reused across push-to-talk activations.
    /// Accessed from the render callback — `nonisolated(unsafe)` because the
    /// value is only mutated while the unit is stopped.
    nonisolated(unsafe) var inputUnit: AudioUnit?

    /// Whether the input unit is currently running.
    private var inputRunning = false

    /// The tap block installed by the caller, invoked from the render callback.
    ///
    /// Accessed from the real-time audio thread — set/cleared only while stopped.
    nonisolated(unsafe) var tapBlock: AVAudioNodeTapBlock?

    /// The audio format of the input device.
    ///
    /// Set during `startInput()`, accessed from the render callback.
    nonisolated(unsafe) var inputFormat: AVAudioFormat?

    // MARK: - Formats

    /// Kokoro TTS format: 24kHz mono float32.
    private static let ttsFormat = KokoroEngine.audioFormat

    /// Feedback tone format: 44.1kHz mono float32.
    private static let feedbackFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: 44_100,
        channels: 1,
        interleaved: false
    )

    // MARK: - Initialization

    /// Create the audio hub, attaching output player nodes.
    ///
    /// Does NOT start either engine — call `start()` after permissions.
    public init() {
        outputEngine.attach(ttsPlayerNode)
        outputEngine.attach(feedbackPlayerNode)

        outputEngine.connect(
            ttsPlayerNode,
            to: outputEngine.mainMixerNode,
            format: Self.ttsFormat
        )

        if let feedbackFmt = Self.feedbackFormat {
            outputEngine.connect(
                feedbackPlayerNode,
                to: outputEngine.mainMixerNode,
                format: feedbackFmt
            )
        }

        Self.logger.info("AudioHub initialized")

        // Observe route changes on the output engine.
        NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange,
            object: outputEngine,
            queue: nil
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.handleOutputConfigChange()
            }
        }
    }

    // MARK: - Output Lifecycle

    /// Start the output engine (TTS + feedback).
    ///
    /// Disables the input side of the output engine's audio unit so macOS
    /// doesn't create an aggregate device that includes the mic — which
    /// would cause a Bluetooth audio blip at launch.
    ///
    /// Runs permanently for the app's lifetime.
    public func start() throws {
        disableOutputEngineInput()
        outputEngine.prepare()
        try outputEngine.start()
        ttsPlayerNode.play()
        feedbackPlayerNode.play()
        Self.logger.info("Output engine started (TTS + feedback armed)")
    }

    /// Restart the output engine after an audio route change.
    private func handleOutputConfigChange() {
        Self.logger.warning("Output engine config changed — restarting")
        disableOutputEngineInput()
        outputEngine.prepare()
        do {
            try outputEngine.start()
            ttsPlayerNode.play()
            feedbackPlayerNode.play()
            Self.logger.info("Output engine restarted after config change")
            if let onReset = onOutputReset {
                Task { await onReset() }
            }
        } catch {
            Self.logger.error("Failed to restart output engine: \(error)")
        }
    }

    /// Disable the input scope on the output engine's audio unit.
    private func disableOutputEngineInput() {
        guard let audioUnit = outputEngine.outputNode.audioUnit else {
            Self.logger.error("No audio unit on output node — Bluetooth blip mitigation inactive")
            return
        }
        var enableInput: UInt32 = 0
        let status = AudioUnitSetProperty(
            audioUnit,
            kAudioOutputUnitProperty_EnableIO,
            kAudioUnitScope_Input,
            1,
            &enableInput,
            UInt32(MemoryLayout<UInt32>.size)
        )
        if status == noErr {
            Self.logger.info("Disabled input on output engine audio unit")
        } else {
            Self.logger.error("Failed to disable input on output engine (status: \(status))")
        }
    }

    // MARK: - Input Lifecycle

    /// Start mic capture and return the hardware input format.
    ///
    /// Creates a raw AUHAL audio unit on first call, configured as input-only
    /// (output disabled before initialization) to avoid aggregate device
    /// creation. Subsequent calls reuse the existing unit.
    public func startInput() throws -> AVAudioFormat {
        if inputUnit == nil {
            try createInputUnit()
        }

        guard let unit = inputUnit else {
            throw NSError(
                domain: "AudioHub",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Input unit not available"]
            )
        }

        let status = AudioOutputUnitStart(unit)
        guard status == noErr else {
            throw NSError(
                domain: "AudioHub",
                code: Int(status),
                userInfo: [NSLocalizedDescriptionKey: "AudioOutputUnitStart failed: \(status)"]
            )
        }
        inputRunning = true

        guard let format = inputFormat else {
            throw NSError(
                domain: "AudioHub",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Input format not available after start"]
            )
        }
        Self.logger.info("Input unit started (\(format.sampleRate)Hz, \(format.channelCount)ch)")
        return format
    }

    /// Install a tap to receive mic audio buffers.
    ///
    /// Must be called after `startInput()`. The block is invoked from the
    /// real-time audio thread via the AUHAL render callback.
    public func installInputTap(
        bufferSize: AVAudioFrameCount,
        format: AVAudioFormat?,
        block: @escaping AVAudioNodeTapBlock
    ) {
        guard tapBlock == nil else {
            Self.logger.warning("installInputTap called but tap already installed")
            return
        }
        tapBlock = block
    }

    /// Stop mic capture and remove the tap.
    ///
    /// Does NOT affect the output engine — TTS and feedback keep playing.
    public func stopInput() {
        if let unit = inputUnit, inputRunning {
            AudioOutputUnitStop(unit)
            inputRunning = false
        }
        tapBlock = nil
        Self.logger.debug("Input unit stopped")
    }

    // MARK: - AUHAL Setup

    /// Create and configure the raw AUHAL audio unit for input-only capture.
    ///
    /// The critical sequence: disable output → enable input → set device →
    /// set format → set callback → initialize. Output MUST be disabled
    /// before initialize to prevent aggregate device creation.
    private func createInputUnit() throws {
        let unit = try createHALOutputUnit()

        // Disable output → enable input → set device (order matters).
        try configureInputOnly(unit)

        var deviceID = resolveInputDevice()
        try setAUHALDevice(unit, deviceID: &deviceID)

        let sampleRate = queryNominalSampleRate(deviceID: deviceID)
        try setInputFormat(unit, sampleRate: sampleRate)
        try installRenderCallback(unit)

        let status = AudioUnitInitialize(unit)
        guard status == noErr else {
            AudioComponentInstanceDispose(unit)
            throw auhalError("AudioUnitInitialize failed: \(status)", status)
        }

        inputUnit = unit
        Self.logger.info("AUHAL input unit created (\(sampleRate)Hz mono, input-only)")
    }

    /// Create a raw HALOutput audio unit instance.
    private func createHALOutputUnit() throws -> AudioUnit {
        var desc = AudioComponentDescription(
            componentType: kAudioUnitType_Output,
            componentSubType: kAudioUnitSubType_HALOutput,
            componentManufacturer: kAudioUnitManufacturer_Apple,
            componentFlags: 0,
            componentFlagsMask: 0
        )
        guard let component = AudioComponentFindNext(nil, &desc) else {
            throw auhalError("HALOutput component not found", -1)
        }
        var unit: AudioUnit?
        let status = AudioComponentInstanceNew(component, &unit)
        guard status == noErr, let unit else {
            throw auhalError("AudioComponentInstanceNew failed: \(status)", status)
        }
        return unit
    }

    /// Disable output and enable input on the AUHAL — must happen before initialize.
    private func configureInputOnly(_ unit: AudioUnit) throws {
        var disableIO: UInt32 = 0
        var status = AudioUnitSetProperty(
            unit,
            kAudioOutputUnitProperty_EnableIO,
            kAudioUnitScope_Output,
            0,
            &disableIO,
            UInt32(MemoryLayout<UInt32>.size)
        )
        guard status == noErr else {
            AudioComponentInstanceDispose(unit)
            throw auhalError("Failed to disable output: \(status)", status)
        }

        var enableIO: UInt32 = 1
        status = AudioUnitSetProperty(
            unit,
            kAudioOutputUnitProperty_EnableIO,
            kAudioUnitScope_Input,
            1,
            &enableIO,
            UInt32(MemoryLayout<UInt32>.size)
        )
        guard status == noErr else {
            AudioComponentInstanceDispose(unit)
            throw auhalError("Failed to enable input: \(status)", status)
        }
    }

    /// Set the current device on the AUHAL.
    private func setAUHALDevice(_ unit: AudioUnit, deviceID: inout AudioDeviceID) throws {
        let status = AudioUnitSetProperty(
            unit,
            kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global,
            0,
            &deviceID,
            UInt32(MemoryLayout<AudioDeviceID>.size)
        )
        guard status == noErr else {
            AudioComponentInstanceDispose(unit)
            throw auhalError("Failed to set input device: \(status)", status)
        }
    }

    /// Query the hardware device's nominal sample rate.
    private func queryNominalSampleRate(deviceID: AudioDeviceID) -> Float64 {
        var nominalRate: Float64 = 0
        var rateSize = UInt32(MemoryLayout<Float64>.size)
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyNominalSampleRate,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        let status = AudioObjectGetPropertyData(
            deviceID,
            &address,
            0,
            nil,
            &rateSize,
            &nominalRate
        )
        let rate: Float64 = (status == noErr && nominalRate > 0) ? nominalRate : 48_000
        Self.logger.info("Input device nominal sample rate: \(rate)")
        return rate
    }

    /// Set mono Float32 stream format on the AUHAL's input element output scope.
    private func setInputFormat(_ unit: AudioUnit, sampleRate: Float64) throws {
        var asbd = AudioStreamBasicDescription(
            mSampleRate: sampleRate,
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked
                | kAudioFormatFlagIsNonInterleaved,
            mBytesPerPacket: 4,
            mFramesPerPacket: 1,
            mBytesPerFrame: 4,
            mChannelsPerFrame: 1,
            mBitsPerChannel: 32,
            mReserved: 0
        )
        let status = AudioUnitSetProperty(
            unit,
            kAudioUnitProperty_StreamFormat,
            kAudioUnitScope_Output,
            1,
            &asbd,
            UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        )
        guard status == noErr else {
            AudioComponentInstanceDispose(unit)
            throw auhalError("Failed to set input format: \(status)", status)
        }
        guard
            let avFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: sampleRate,
                channels: 1,
                interleaved: false
            )
        else {
            AudioComponentInstanceDispose(unit)
            throw auhalError("Failed to create AVAudioFormat", -1)
        }
        inputFormat = avFormat
    }

    /// Install the render callback on the AUHAL.
    private func installRenderCallback(_ unit: AudioUnit) throws {
        var callbackStruct = AURenderCallbackStruct(
            inputProc: auhalInputCallback,
            inputProcRefCon: Unmanaged.passUnretained(self).toOpaque()
        )
        let status = AudioUnitSetProperty(
            unit,
            kAudioOutputUnitProperty_SetInputCallback,
            kAudioUnitScope_Global,
            0,
            &callbackStruct,
            UInt32(MemoryLayout<AURenderCallbackStruct>.size)
        )
        guard status == noErr else {
            AudioComponentInstanceDispose(unit)
            throw auhalError("Failed to set input callback: \(status)", status)
        }
    }

    /// Create an NSError for AUHAL setup failures.
    private func auhalError(_ message: String, _ status: OSStatus) -> NSError {
        NSError(
            domain: "AudioHub",
            code: Int(status),
            userInfo: [NSLocalizedDescriptionKey: message]
        )
    }

    /// Resolve the user's preferred input device, falling back to the built-in mic.
    private func resolveInputDevice() -> AudioDeviceID {
        let uid = UserDefaults.standard.string(forKey: "inputDeviceUID") ?? ""
        if !uid.isEmpty, let preferred = AudioDeviceManager.deviceID(forUID: uid) {
            Self.logger.info("Input device: \(uid)")
            return preferred
        }
        if let builtIn = AudioDeviceManager.builtInMicID() {
            Self.logger.info("Input device: built-in mic")
            return builtIn
        }
        Self.logger.warning("No input device found, using system default")
        return AudioDeviceID(kAudioObjectUnknown)
    }

    deinit {
        if let unit = inputUnit {
            AudioOutputUnitStop(unit)
            AudioUnitUninitialize(unit)
            AudioComponentInstanceDispose(unit)
        }
        tapBlock = nil
        inputFormat = nil
    }
}

// MARK: - AUHAL Render Callback

/// C-function render callback invoked on the real-time audio thread.
///
/// Calls `AudioUnitRender` to pull samples from the input device, wraps
/// them in an `AVAudioPCMBuffer`, and forwards to the installed tap block.
private func auhalInputCallback(
    inRefCon: UnsafeMutableRawPointer,
    ioActionFlags: UnsafeMutablePointer<AudioUnitRenderActionFlags>,
    inTimeStamp: UnsafePointer<AudioTimeStamp>,
    inBusNumber: UInt32,
    inNumberFrames: UInt32,
    ioData: UnsafeMutablePointer<AudioBufferList>?
) -> OSStatus {
    let hub = Unmanaged<AudioHub>.fromOpaque(inRefCon).takeUnretainedValue()

    guard let unit = hub.inputUnit, let format = hub.inputFormat else {
        return noErr
    }

    // Use AVAudioPCMBuffer's own buffer list for proper format alignment.
    guard let pcmBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: inNumberFrames) else {
        return noErr
    }
    pcmBuffer.frameLength = inNumberFrames

    let status = AudioUnitRender(
        unit,
        ioActionFlags,
        inTimeStamp,
        inBusNumber,
        inNumberFrames,
        pcmBuffer.mutableAudioBufferList
    )

    if status != noErr {
        return noErr
    }

    if let tapBlock = hub.tapBlock {
        let avTime = AVAudioTime(audioTimeStamp: inTimeStamp, sampleRate: format.sampleRate)
        tapBlock(pcmBuffer, avTime)
    }

    return noErr
}

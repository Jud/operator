import AVFoundation
import AudioToolbox
import KokoroCoreML

/// Central owner of all audio I/O, with separate engines for output and input.
///
/// ```
/// AudioHub
///  ├─ outputEngine (permanent, starts at launch)
///  │   ├─ ttsPlayerNode (24kHz mono — Kokoro TTS)
///  │   └─ feedbackPlayerNode (44.1kHz mono — .caf tones)
///  └─ inputEngine (start/stop per push-to-talk)
///      └─ inputNode (mic tap)
/// ```
///
/// Two separate engines avoid the shared-aggregate-device problem: stopping the
/// input engine for mic release does not interrupt TTS or feedback output. The
/// output engine runs permanently; the input engine starts/stops per activation.
@MainActor
public final class AudioHub {
    private static let logger = Log.logger(for: "AudioHub")

    // MARK: - Output Engine

    /// Persistent output engine for TTS and feedback tones.
    private let outputEngine = AVAudioEngine()

    /// Player node for Kokoro TTS output (24kHz mono float32).
    public let ttsPlayerNode = AVAudioPlayerNode()

    /// Player node for feedback tone output (44.1kHz mono).
    public let feedbackPlayerNode = AVAudioPlayerNode()

    // MARK: - Input Engine

    /// Per-activation input engine for mic capture.
    private let inputEngine = AVAudioEngine()

    /// Whether an input tap is currently installed.
    private var inputTapInstalled = false

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

        // Observe route changes on both engines.
        NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange,
            object: outputEngine,
            queue: nil
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.handleOutputConfigChange()
            }
        }
        NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange,
            object: inputEngine,
            queue: nil
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.handleInputConfigChange()
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
        // DisableIO before prepare — the audio unit exists from init()'s
        // mixer access. Setting EnableIO after prepare may be too late to
        // prevent the full-duplex aggregate device path.
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
        } catch {
            Self.logger.error("Failed to restart output engine: \(error)")
        }
    }

    /// Handle input engine config change (e.g., mic disconnected mid-recording).
    private func handleInputConfigChange() {
        Self.logger.warning("Input engine config changed")
        // If actively recording, the tap is now invalid. Stop the input engine
        // so the next activation starts fresh.
        if inputTapInstalled {
            inputEngine.inputNode.removeTap(onBus: 0)
            inputTapInstalled = false
        }
        inputEngine.stop()
        Self.logger.info("Input engine stopped after config change (will restart on next PTT)")
    }

    /// Disable the input scope on the output engine's audio unit.
    ///
    /// By default, AVAudioEngine creates an aggregate device combining
    /// the system default input + output. Disabling input I/O prevents
    /// the aggregate from including the mic, avoiding a Bluetooth blip.
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
            1,  // input element
            &enableInput,
            UInt32(MemoryLayout<UInt32>.size)
        )
        if status == noErr {
            Self.logger.info("Disabled input on output engine audio unit")
        } else {
            Self.logger.error("Failed to disable input on output engine (status: \(status)) — Bluetooth blip likely")
        }
    }

    // MARK: - Input Lifecycle

    /// Start the input engine and return the hardware input format.
    ///
    /// Called at the beginning of each push-to-talk activation. Applies the
    /// preferred input device, prepares, and starts the input engine.
    /// The first call creates the aggregate device (one-time blip on BT);
    /// subsequent calls reuse the warmed HAL path (no blip).
    public func startInput() throws -> AVAudioFormat {
        // Touch inputNode to materialize the audio unit. Then set
        // EnableIO + device BEFORE prepare, per Apple's HAL sequence:
        // "enable/disable IO, set current device, then initialize/start."
        _ = inputEngine.inputNode
        applyInputDevice()
        disableInputEngineOutput()
        inputEngine.prepare()
        let format = inputEngine.inputNode.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            throw NSError(
                domain: "AudioHub",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Degenerate input format: \(format)"]
            )
        }
        try inputEngine.start()
        Self.logger.info(
            "Input engine started (\(format.sampleRate)Hz, \(format.channelCount)ch)"
        )
        return format
    }

    /// Install a tap on the input node to capture mic audio.
    ///
    /// Must be called after `startInput()`. Guards against double-install.
    public func installInputTap(
        bufferSize: AVAudioFrameCount,
        format: AVAudioFormat?,
        block: @escaping AVAudioNodeTapBlock
    ) {
        guard !inputTapInstalled else {
            Self.logger.warning("installInputTap called but tap already installed")
            return
        }
        inputEngine.inputNode.installTap(
            onBus: 0,
            bufferSize: bufferSize,
            format: format,
            block: block
        )
        inputTapInstalled = true
    }

    /// Stop the input engine and remove the tap.
    ///
    /// Does NOT affect the output engine — TTS and feedback keep playing.
    public func stopInput() {
        if inputTapInstalled {
            inputEngine.inputNode.removeTap(onBus: 0)
            inputTapInstalled = false
        }
        inputEngine.stop()
        Self.logger.debug("Input engine stopped")
    }

    // MARK: - Input Device

    /// Disable the output scope on the input engine's audio unit.
    ///
    /// Prevents the input engine's aggregate device from including the
    /// AirPods output, which would cause a Bluetooth renegotiation blip.
    private func disableInputEngineOutput() {
        guard let audioUnit = inputEngine.inputNode.audioUnit else {
            Self.logger.error("No audio unit on input node — Bluetooth blip mitigation inactive")
            return
        }
        var enableOutput: UInt32 = 0
        let status = AudioUnitSetProperty(
            audioUnit,
            kAudioOutputUnitProperty_EnableIO,
            kAudioUnitScope_Output,
            0,  // output element
            &enableOutput,
            UInt32(MemoryLayout<UInt32>.size)
        )
        if status == noErr {
            Self.logger.info("Disabled output on input engine audio unit")
        } else {
            Self.logger.error("Failed to disable output on input engine (status: \(status)) — Bluetooth blip likely")
        }
    }

    /// Apply the user's preferred input device, falling back to the built-in mic.
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

        guard let audioUnit = inputEngine.inputNode.audioUnit else {
            Self.logger.warning("No audio unit on input node")
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
            Self.logger.info("Input device: \(uid.isEmpty ? "built-in mic" : uid)")
        } else {
            Self.logger.warning("Failed to set input device (status: \(status))")
        }
    }
}

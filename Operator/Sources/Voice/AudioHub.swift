import AVFoundation
import AudioToolbox
import KokoroCoreML

/// Central owner of the single AVAudioEngine shared by all audio consumers.
///
/// ```
/// AudioHub
///  ├─ engine: AVAudioEngine (single, persistent, created once at startup)
///  ├─ ttsPlayerNode: AVAudioPlayerNode (24kHz mono — Kokoro TTS buffers)
///  ├─ feedbackPlayerNode: AVAudioPlayerNode (44.1kHz mono — .caf feedback tones)
///  └─ engine.inputNode (mic capture — tap installed/removed per push-to-talk)
/// ```
///
/// Consumers receive their node, not the engine. KokoroSpeechManager schedules
/// TTS buffers on `ttsPlayerNode`. AudioFeedback schedules tone buffers on
/// `feedbackPlayerNode`. SpeechTranscriber installs/removes a tap on the input
/// node via `installInputTap` / `removeInputTap`.
///
/// The engine starts once after permissions are granted and runs for the app's
/// lifetime. This avoids the Bluetooth audio blip caused by repeated aggregate
/// device creation/destruction.
@MainActor
public final class AudioHub {
    private static let logger = Log.logger(for: "AudioHub")

    // MARK: - Public Nodes

    /// Player node for Kokoro TTS output (24kHz mono float32).
    public let ttsPlayerNode = AVAudioPlayerNode()

    /// Player node for feedback tone output (44.1kHz mono Int16).
    public let feedbackPlayerNode = AVAudioPlayerNode()

    // MARK: - Engine

    /// The single audio engine.
    ///
    /// Not exposed — consumers use nodes.
    private let engine = AVAudioEngine()

    /// Whether an input tap is currently installed.
    private var inputTapInstalled = false

    // MARK: - Formats

    /// Kokoro TTS format: 24kHz mono float32.
    private static let ttsFormat = KokoroEngine.audioFormat

    /// Feedback tone format: 44.1kHz mono Int16 (all .caf files use this).
    private static let feedbackFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: 44_100,
        channels: 1,
        interleaved: false
    )

    // MARK: - Initialization

    /// Create the audio hub, attaching and connecting all player nodes.
    ///
    /// Does NOT start the engine — call `start()` after permissions are granted.
    public init() {
        engine.attach(ttsPlayerNode)
        engine.attach(feedbackPlayerNode)

        engine.connect(
            ttsPlayerNode,
            to: engine.mainMixerNode,
            format: Self.ttsFormat
        )

        if let feedbackFmt = Self.feedbackFormat {
            engine.connect(
                feedbackPlayerNode,
                to: engine.mainMixerNode,
                format: feedbackFmt
            )
        }

        Self.logger.info("AudioHub initialized (nodes attached, engine not started)")
    }

    // MARK: - Lifecycle

    /// Start the engine after permissions are granted.
    ///
    /// Applies the preferred input device, prepares the engine, starts it,
    /// and arms both player nodes. The aggregate device is created here and
    /// persists for the app's lifetime.
    public func start() throws {
        applyInputDevice()
        engine.prepare()
        try engine.start()
        ttsPlayerNode.play()
        feedbackPlayerNode.play()
        Self.logger.info("AudioHub started (engine running, nodes armed)")
    }

    // MARK: - Input Tap

    /// The current input format (sample rate, channels) from the hardware.
    public var inputFormat: AVAudioFormat {
        engine.inputNode.outputFormat(forBus: 0)
    }

    /// Install a tap on the input node to begin capturing mic audio.
    ///
    /// Guards against double-install — `installTap` raises an ObjC exception
    /// if a tap is already installed on the bus.
    public func installInputTap(
        bufferSize: AVAudioFrameCount,
        format: AVAudioFormat?,
        block: @escaping AVAudioNodeTapBlock
    ) {
        guard !inputTapInstalled else {
            Self.logger.warning("installInputTap called but tap already installed")
            return
        }
        engine.inputNode.installTap(
            onBus: 0,
            bufferSize: bufferSize,
            format: format,
            block: block
        )
        inputTapInstalled = true
        Self.logger.debug("Input tap installed")
    }

    /// Remove the input tap.
    ///
    /// The engine stays running — only the tap is removed.
    public func removeInputTap() {
        guard inputTapInstalled else {
            return
        }
        engine.inputNode.removeTap(onBus: 0)
        inputTapInstalled = false
        Self.logger.debug("Input tap removed")
    }

    // MARK: - Input Device

    /// Apply the user's preferred input device, falling back to the built-in mic.
    ///
    /// Prevents the aggregate device from including a Bluetooth mic, which
    /// would trigger an audio profile renegotiation blip on wireless headphones.
    private func applyInputDevice() {
        let uid = UserDefaults.standard.string(forKey: "inputDeviceUID") ?? ""
        let deviceID: AudioDeviceID
        if !uid.isEmpty, let preferred = AudioDeviceManager.deviceID(forUID: uid) {
            deviceID = preferred
        } else if let builtIn = AudioDeviceManager.builtInMicID() {
            deviceID = builtIn
        } else {
            Self.logger.debug("No preferred or built-in mic found, using system default")
            return
        }

        guard let audioUnit = engine.inputNode.audioUnit else {
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
}

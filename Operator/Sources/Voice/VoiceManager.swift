import AVFoundation
import os

/// Manages voice selection for Operator and agent speech output.
///
/// When Kokoro models are available, returns `.kokoro(name:)` descriptors with
/// distinct voice presets per agent. Falls back to `.system(AVSpeechSynthesisVoice)`
/// when Kokoro is not available.
public final class VoiceManager: Sendable {
    private static let logger = Log.logger(for: "VoiceManager")

    /// Kokoro voice presets for agent differentiation.
    private static let kokoroVoices = [
        "af_heart", "am_adam", "af_bella", "bm_george",
        "af_nicole", "am_michael", "af_sarah", "am_onyx"
    ]

    /// The voice used for Operator system messages.
    public let operatorVoice: VoiceDescriptor

    /// The default voice used for agent speech output (same as operatorVoice).
    public var defaultAgentVoice: VoiceDescriptor { operatorVoice }

    /// Whether Kokoro voices are active.
    public let useKokoro: Bool

    /// Index for cycling through agent voices.
    private let agentVoiceIndex = AgentVoiceCounter()

    /// Creates a new voice manager.
    ///
    /// - Parameter kokoroVoices: Available Kokoro voice names. If empty, falls back to Apple TTS.
    public init(kokoroVoices: [String] = []) {
        if !kokoroVoices.isEmpty {
            self.useKokoro = true
            self.operatorVoice = .kokoro(name: kokoroVoices.first ?? "af_heart")
            Self.logger.info("VoiceManager: using Kokoro voices (\(kokoroVoices.count) available)")
        } else {
            self.useKokoro = false
            self.operatorVoice = .system(AVSpeechSynthesisVoice())
            Self.logger.info("VoiceManager: using Apple TTS fallback")
        }
    }

    /// Returns a distinct VoiceDescriptor for the next agent.
    ///
    /// When using Kokoro, cycles through distinct voice presets so each agent
    /// gets a truly different voice. With Apple TTS, returns the default voice.
    public func nextAgentVoice() -> VoiceDescriptor {
        guard useKokoro else {
            return operatorVoice
        }
        let voices = Self.kokoroVoices
        let idx = agentVoiceIndex.next() % voices.count
        return .kokoro(name: voices[idx])
    }
}

/// Thread-safe counter for cycling through agent voices.
private final class AgentVoiceCounter: Sendable {
    private let counter = OSAllocatedUnfairLock(initialState: 0)

    func next() -> Int {
        counter.withLock { value in
            let current = value
            value += 1
            return current
        }
    }
}

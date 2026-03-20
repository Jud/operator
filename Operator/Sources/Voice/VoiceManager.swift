import AVFoundation
import os

/// Manages voice selection for Operator and agent speech output.
///
/// When Kokoro models are available, returns `.kokoro(name:)` descriptors with
/// distinct voice presets per agent. Falls back to `.system(AVSpeechSynthesisVoice)`
/// when Kokoro is not available.
public final class VoiceManager: Sendable {
    private static let logger = Log.logger(for: "VoiceManager")

    /// Curated Kokoro voice presets: American and British, male and female only.
    ///
    /// Excludes whispery, novelty, and non-English voices.
    /// af_nicole removed — tends to whisper.
    private static let kokoroVoices = [
        // American female
        "af_heart", "af_bella", "af_sarah", "af_nova",
        // American male
        "am_adam", "am_michael", "am_eric", "am_liam",
        // British female
        "bf_emma", "bf_lily", "bf_isabella",
        // British male
        "bm_george", "bm_daniel", "bm_lewis"
    ]

    /// The voice used for Operator system messages.
    public let operatorVoice: VoiceDescriptor

    /// The default voice used for agent speech output (same as operatorVoice).
    public var defaultAgentVoice: VoiceDescriptor { operatorVoice }

    /// Whether Kokoro voices are active.
    public let useKokoro: Bool

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

    /// Returns a deterministic VoiceDescriptor for an agent based on its project name.
    ///
    /// The same project always gets the same voice across sessions.
    /// Operator's own voice (index 0) is excluded from agent assignment.
    public func voiceForAgent(named name: String) -> VoiceDescriptor {
        guard useKokoro else {
            return operatorVoice
        }
        let voices = Self.kokoroVoices
        // Skip index 0 (operator voice) for agents
        let agentVoices = Array(voices.dropFirst())
        guard !agentVoices.isEmpty else { return operatorVoice }
        let hash = abs(name.hashValue)
        let idx = hash % agentVoices.count
        return .kokoro(name: agentVoices[idx])
    }
}

import AVFoundation

/// Manages speech synthesis with word-level tracking for interruption support.
///
/// Wraps AVSpeechSynthesizer and implements AVSpeechSynthesizerDelegate to track
/// exactly which word the synthesizer has reached during playback. This enables
/// the `interrupt()` method to return precisely what the user heard vs. what
/// remains unheard -- critical context for the interruption handler when deciding
/// whether to replay, skip, or reroute.
///
/// All access must occur on the main actor since AVSpeechSynthesizer and its
/// delegate callbacks operate on the main thread.
@MainActor
public final class SpeechManager: NSObject, SpeechManaging, AVSpeechSynthesizerDelegate {
    private static let logger = Log.logger(for: "SpeechManager")

    /// The underlying speech synthesizer instance.
    private let synthesizer = AVSpeechSynthesizer()

    /// Whether the synthesizer is currently speaking.
    public var isSpeaking: Bool { synthesizer.isSpeaking }

    /// The full text (including prefix) of the currently playing utterance.
    private var currentText: String = ""

    /// Character index marking how far speech has progressed through `currentText`.
    ///
    /// Updated by the willSpeakRangeOfSpeechString delegate callback.
    private var lastSpokenCharIndex: Int = 0

    /// The session name (prefix) associated with the currently playing utterance.
    private var currentSession: String = ""

    /// Stream that yields each time an utterance finishes playing (not interrupted).
    ///
    /// AudioQueue consumes this to trigger playback of the next queued message.
    public let finishedSpeaking: AsyncStream<Void>

    /// Continuation backing `finishedSpeaking`.
    private let finishedContinuation: AsyncStream<Void>.Continuation

    override public init() {
        let (stream, continuation) = AsyncStream<Void>.makeStream()
        self.finishedSpeaking = stream
        self.finishedContinuation = continuation
        super.init()
        synthesizer.delegate = self
    }

    /// Speak text with an agent name prefix and the specified voice.
    ///
    /// The full spoken text is formatted as "[prefix]: [text]". The speech rate
    /// is set to 0.55, slightly faster than the AVSpeechUtterance default of 0.5,
    /// per the technical specification.
    ///
    /// - Parameters:
    ///   - text: The message content to speak.
    ///   - voice: The VoiceDescriptor identifying the voice for this utterance.
    ///   - prefix: The session name prefix (e.g., "Sudo", "Operator").
    ///   - pitchMultiplier: Pitch variation for per-agent differentiation. Defaults to 1.0.
    public func speak(_ text: String, voice: VoiceDescriptor, prefix: String, pitchMultiplier: Float = 1.0) {
        let fullText = "\(prefix): \(text)"
        currentText = fullText
        currentSession = prefix
        lastSpokenCharIndex = 0

        let utterance = AVSpeechUtterance(string: fullText)
        utterance.voice = voice.appleVoice
        utterance.rate = 0.55
        utterance.pitchMultiplier = pitchMultiplier

        Self.logger.info("Speaking: \"\(prefix): \(text.prefix(60))...\" (pitch: \(pitchMultiplier))")
        synthesizer.speak(utterance)
    }

    /// Immediately stop speech and return what was heard vs. what remains.
    ///
    /// Uses the word-level tracking from `willSpeakRangeOfSpeechString` to split
    /// the full utterance text at the exact point where speech was interrupted.
    /// Returns the session name so the caller knows which agent's message was cut off.
    ///
    /// - Returns: An InterruptInfo with heard text, unheard text, and session name.
    public func interrupt() -> InterruptInfo {
        synthesizer.stopSpeaking(at: .immediate)

        let heard = String(currentText.prefix(lastSpokenCharIndex))
        let unheard = String(currentText.dropFirst(lastSpokenCharIndex))

        let charPos = self.lastSpokenCharIndex
        let total = self.currentText.count
        Self.logger.info("Interrupted at char \(charPos)/\(total) for session \(self.currentSession)")
        Self.logger.debug("Heard: \"\(heard.suffix(40))\" | Unheard: \"\(unheard.prefix(40))\"")

        return InterruptInfo(heardText: heard, unheardText: unheard, session: currentSession)
    }

    /// Stop any current speech without tracking interruption state.
    public func stop() {
        synthesizer.stopSpeaking(at: .immediate)
    }

    // MARK: - AVSpeechSynthesizerDelegate

    /// Tracks word-level progress through the utterance text.
    nonisolated public func speechSynthesizer(
        _ synth: AVSpeechSynthesizer,
        willSpeakRangeOfSpeechString range: NSRange,
        utterance: AVSpeechUtterance
    ) {
        MainActor.assumeIsolated {
            lastSpokenCharIndex = range.location + range.length
        }
    }

    /// Called when an utterance finishes playing to completion (not interrupted).
    nonisolated public func speechSynthesizer(
        _ synth: AVSpeechSynthesizer,
        didFinish utterance: AVSpeechUtterance
    ) {
        MainActor.assumeIsolated {
            Self.logger.debug("Finished speaking: \"\(self.currentSession)\"")
            self.currentText = ""
            self.lastSpokenCharIndex = 0
            self.currentSession = ""
            self.finishedContinuation.yield()
        }
    }

    /// Called when an utterance is cancelled/interrupted via stopSpeaking.
    nonisolated public func speechSynthesizer(
        _ synth: AVSpeechSynthesizer,
        didCancel utterance: AVSpeechUtterance
    ) {
        MainActor.assumeIsolated {
            Self.logger.debug("Speech cancelled for session: \"\(self.currentSession)\"")
        }
    }
}

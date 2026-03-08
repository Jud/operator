import AVFoundation

/// Adaptive wrapper that delegates to either ``Qwen3TTSSpeechManager`` (local) or
/// ``SpeechManager`` (Apple) based on user preference and model availability.
///
/// Conforms to ``SpeechManaging``, making it transparent to consumers like
/// ``AudioQueue``. On construction, checks whether the local model is available.
/// If the local engine fails during operation, automatically falls back to Apple
/// TTS and logs the event.
///
/// Engine switching: ``setUseLocal(_:)`` changes the active engine preference.
/// The switch takes effect on the next ``speak(_:voice:prefix:pitchMultiplier:)``
/// call, not mid-utterance. No application restart required.
@MainActor
public final class AdaptiveSpeechManager: SpeechManaging {
    private static let logger = Log.logger(for: "AdaptiveSpeechManager")

    // MARK: - Properties

    private let localEngine: Qwen3TTSSpeechManager
    private let fallbackEngine: SpeechManager
    private let modelManager: ModelManager
    private var useLocal: Bool
    private var didFallBack = false

    /// Tracks which engine handled the current/last speak() call.
    private var activeIsLocal = false

    /// Whether the engine is currently configured to use the local Qwen3-TTS model.
    public var isUsingLocal: Bool {
        useLocal && !didFallBack
    }

    /// Called when the engine falls back from local Qwen3-TTS to Apple TTS.
    ///
    /// The closure receives a human-readable notification message. Wired
    /// during bootstrap to enqueue a spoken alert via AudioQueue so the
    /// user knows the local engine is unavailable.
    public var onFallback: (@MainActor (String) -> Void)?

    // MARK: - SpeechManaging

    /// Whether the active engine is currently speaking.
    public var isSpeaking: Bool {
        if activeIsLocal {
            return localEngine.isSpeaking
        }
        return fallbackEngine.isSpeaking
    }

    /// Unified stream that yields when either engine finishes an utterance.
    public let finishedSpeaking: AsyncStream<Void>
    private let finishedContinuation: AsyncStream<Void>.Continuation

    // MARK: - Initialization

    /// Creates an adaptive speech manager.
    ///
    /// Spawns listener tasks on both engines' `finishedSpeaking` streams to
    /// forward completion events to the unified stream consumed by ``AudioQueue``.
    ///
    /// - Parameters:
    ///   - localEngine: The Qwen3-TTS engine for local speech synthesis.
    ///   - fallbackEngine: The Apple AVSpeechSynthesizer engine as fallback.
    ///   - modelManager: The shared model manager for checking model availability.
    ///   - preferLocal: Initial preference for local engine. When `true`, uses
    ///     Qwen3-TTS. When `false`, uses Apple AVSpeechSynthesizer.
    public init(
        localEngine: Qwen3TTSSpeechManager,
        fallbackEngine: SpeechManager,
        modelManager: ModelManager,
        preferLocal: Bool = true
    ) {
        self.localEngine = localEngine
        self.fallbackEngine = fallbackEngine
        self.modelManager = modelManager
        self.useLocal = preferLocal

        let (stream, continuation) = AsyncStream<Void>.makeStream()
        self.finishedSpeaking = stream
        self.finishedContinuation = continuation

        Self.logger.info(
            "AdaptiveSpeechManager initialized, preferLocal: \(preferLocal)"
        )

        startForwardingStreams()
    }

    // MARK: - Engine Switching

    /// Switch between local Qwen3-TTS and Apple AVSpeechSynthesizer engines.
    ///
    /// Takes effect on the next ``speak(_:voice:prefix:pitchMultiplier:)`` call,
    /// not mid-utterance. Called by the Settings UI when the user changes TTS
    /// engine preferences.
    ///
    /// - Parameter enabled: `true` to use Qwen3-TTS, `false` to use Apple TTS.
    public func setUseLocal(_ enabled: Bool) {
        useLocal = enabled
        didFallBack = false
        Self.logger.info("TTS engine preference changed to \(enabled ? "Qwen3-TTS" : "Apple")")
    }

    // MARK: - SpeechManaging Methods

    /// Speak text using the currently active engine.
    ///
    /// If the local engine is preferred, attempts to use it. Before delegating
    /// to the local engine, checks whether a previous model load failure was
    /// recorded. If so, triggers fallback proactively so the message is spoken
    /// via Apple TTS instead of being silently dropped.
    ///
    /// The fallback is sticky until the user re-enables local in Settings or
    /// calls ``setUseLocal(_:)``.
    public func speak(_ text: String, voice: VoiceDescriptor, prefix: String, pitchMultiplier: Float) {
        if useLocal && !didFallBack {
            guard localEngine.modelLoadFailed else {
                activeIsLocal = true
                localEngine.speak(text, voice: voice, prefix: prefix, pitchMultiplier: pitchMultiplier)
                return
            }
            triggerFallback()
        }

        activeIsLocal = false
        fallbackEngine.speak(text, voice: voice, prefix: prefix, pitchMultiplier: pitchMultiplier)
    }

    /// Interrupt the currently active engine and return heard/unheard text split.
    public func interrupt() -> InterruptInfo {
        if activeIsLocal {
            return localEngine.interrupt()
        }
        return fallbackEngine.interrupt()
    }

    /// Stop speech on the currently active engine without tracking interruption state.
    public func stop() {
        if activeIsLocal {
            localEngine.stop()
        } else {
            fallbackEngine.stop()
        }
    }

    // MARK: - Fallback

    /// Mark that the local engine has failed and all subsequent calls should use Apple TTS.
    ///
    /// Called externally by the engine coordination layer when a local engine
    /// failure is detected, or internally when a prior model load failure is
    /// detected in ``speak(_:voice:prefix:pitchMultiplier:)``. Delivers a
    /// spoken fallback notification to the user.
    public func triggerFallback() {
        guard !didFallBack else {
            return
        }
        didFallBack = true
        activeIsLocal = false
        Self.logger.warning("Falling back to Apple TTS; local model unavailable")
        onFallback?("Falling back to Apple text-to-speech. Local model unavailable.")
    }

    // MARK: - Stream Forwarding

    /// Listen to both engines' `finishedSpeaking` streams and forward events
    /// to the unified stream consumed by AudioQueue.
    ///
    /// Only the active engine produces events at any given time (since only one
    /// engine is speaking), so forwarding from both is safe.
    private func startForwardingStreams() {
        let localStream = localEngine.finishedSpeaking
        let fallbackStream = fallbackEngine.finishedSpeaking
        let continuation = finishedContinuation

        Task { [weak self] in
            for await _ in localStream {
                guard self != nil else { break }
                continuation.yield()
            }
        }

        Task { [weak self] in
            for await _ in fallbackStream {
                guard self != nil else { break }
                continuation.yield()
            }
        }
    }
}

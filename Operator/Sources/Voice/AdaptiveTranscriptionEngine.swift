import AVFoundation
import os

/// Adaptive wrapper that delegates to either ``ParakeetEngine`` (local) or
/// ``AppleSpeechEngine`` (Apple) based on user preference and model availability.
///
/// Conforms to ``TranscriptionEngine``, making it transparent to consumers like
/// ``SpeechTranscriber``. On construction, checks whether the local model is
/// available. If the local engine fails during operation, automatically falls
/// back to Apple STT and logs the event.
///
/// Thread safety: mirrors the same contract as the wrapped engines. `append(_:)`
/// is safe from any thread. All other methods are called from `@MainActor` via
/// `SpeechTranscriber`.
public final class AdaptiveTranscriptionEngine: TranscriptionEngine, @unchecked Sendable {
    private static let logger = Log.logger(for: "AdaptiveTranscriptionEngine")

    // MARK: - Properties

    private let localEngine: ParakeetEngine
    private let fallbackEngine: AppleSpeechEngine
    private let modelManager: ModelManager
    private var useLocal: Bool
    private var activeEngine: any TranscriptionEngine
    private var didFallBack = false

    /// Whether the engine is currently configured to use the local Parakeet model.
    public var isUsingLocal: Bool {
        useLocal && !didFallBack
    }

    /// Called when the engine falls back from local Parakeet to Apple STT.
    ///
    /// The closure receives a human-readable notification message. Wired
    /// during bootstrap to enqueue a spoken alert via AudioQueue so the
    /// user knows the local engine is unavailable.
    public var onFallback: (@Sendable (String) -> Void)?

    // MARK: - Initialization

    /// Creates an adaptive transcription engine.
    ///
    /// - Parameters:
    ///   - localEngine: The Parakeet CoreML engine for local transcription.
    ///   - fallbackEngine: The Apple SFSpeechRecognizer engine as fallback.
    ///   - modelManager: The shared model manager for checking model availability.
    ///   - preferLocal: Initial preference for local engine. When `true`, uses
    ///     Parakeet. When `false`, uses Apple SFSpeechRecognizer.
    public init(
        localEngine: ParakeetEngine,
        fallbackEngine: AppleSpeechEngine,
        modelManager: ModelManager,
        preferLocal: Bool = true
    ) {
        self.localEngine = localEngine
        self.fallbackEngine = fallbackEngine
        self.modelManager = modelManager
        self.useLocal = preferLocal
        self.activeEngine = preferLocal ? localEngine : fallbackEngine

        Self.logger.info(
            "AdaptiveTranscriptionEngine initialized, preferLocal: \(preferLocal)"
        )
    }

    // MARK: - Engine Switching

    /// Switch between local and Apple engines.
    ///
    /// Takes effect on the next `prepare()` call, not mid-operation. Called
    /// by the Settings UI when the user changes engine preferences.
    ///
    /// - Parameter enabled: `true` to use Parakeet, `false` to use Apple STT.
    public func setUseLocal(_ enabled: Bool) {
        useLocal = enabled
        didFallBack = false
        Self.logger.info("STT engine preference changed to \(enabled ? "Parakeet" : "Apple")")
    }

    // MARK: - TranscriptionEngine

    /// Prepare the active engine for a new transcription session.
    ///
    /// If the local engine is preferred but fails to prepare, falls back to
    /// Apple STT with a warning log.
    public func prepare() throws {
        if useLocal && !didFallBack {
            do {
                try localEngine.prepare()
                activeEngine = localEngine
                return
            } catch {
                Self.logger.warning(
                    "Parakeet prepare failed, falling back to Apple STT: \(error.localizedDescription)"
                )
                didFallBack = true
                activeEngine = fallbackEngine
                onFallback?("Falling back to Apple speech recognition. Local model unavailable.")
            }
        }
        try fallbackEngine.prepare()
        activeEngine = fallbackEngine
    }

    /// Forward an audio buffer to the active engine.
    ///
    /// Called from the audio tap thread. Delegates directly to whichever
    /// engine was selected during `prepare()`.
    public func append(_ buffer: AVAudioPCMBuffer) {
        activeEngine.append(buffer)
    }

    /// Signal end of audio and return the transcribed text.
    ///
    /// If the local engine was active but fails during transcription, the
    /// fallback cannot recover mid-session (audio was sent to the local engine).
    /// The failure is logged and `nil` is returned. The next session will use
    /// the fallback engine.
    public func finishAndTranscribe() async -> String? {
        let result = await activeEngine.finishAndTranscribe()

        if result == nil && activeEngine is ParakeetEngine && !didFallBack {
            Self.logger.warning(
                "Parakeet transcription returned nil, will fall back to Apple STT on next session"
            )
            didFallBack = true
            onFallback?("Falling back to Apple speech recognition. Local model unavailable.")
        }

        return result
    }

    /// Cancel any in-progress transcription on the active engine.
    public func cancel() {
        activeEngine.cancel()
    }
}

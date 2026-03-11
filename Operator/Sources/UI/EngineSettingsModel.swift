import Foundation

/// Observable model bridging adaptive engine wrappers and ModelManager
/// state to the Settings UI.
///
/// Follows the same singleton pattern as ``MenuBarModel``. Wired during
/// application bootstrap after adaptive engines and ModelManager are created.
@MainActor
@Observable
public final class EngineSettingsModel {
    /// Shared singleton instance used by both SettingsView and AppDelegate.
    public static let shared = EngineSettingsModel()

    // MARK: - Model State

    /// Current lifecycle state of each local model, updated via ModelManager's AsyncStream.
    public var modelStates: [ModelManager.ModelType: ModelManager.ModelState] = [
        .stt: .notDownloaded,
        .tts: .notDownloaded,
        .routing: .notDownloaded
    ]

    /// Total bytes used by cached models.
    public var totalCacheSizeBytes: UInt64 = 0

    // MARK: - Engine Switch Callbacks

    /// Called when the STT engine preference changes.
    ///
    /// The Bool parameter indicates whether to use the local engine.
    public var onSTTEngineChanged: ((Bool) -> Void)?

    /// Called when the TTS engine preference changes.
    ///
    /// The Bool parameter indicates whether to use the local engine.
    public var onTTSEngineChanged: ((Bool) -> Void)?

    /// Called when the routing engine preference changes.
    ///
    /// The Bool parameter indicates whether to use the local engine.
    public var onRoutingEngineChanged: ((Bool) -> Void)?

    /// Called when the speech rate preference changes.
    public var onSpeechRateChanged: ((Float) -> Void)?

    /// Called when the voice style instruction changes.
    public var onVoiceInstructChanged: ((String) -> Void)?

    // MARK: - Lifecycle

    private var observationTask: Task<Void, Never>?

    /// Creates a new engine settings model.
    public init() {}

    /// Connect to ModelManager for state observation and start listening for changes.
    ///
    /// Reads the current model states snapshot, then subscribes to the
    /// ModelManager's `stateChanged` AsyncStream for real-time updates.
    /// Also refreshes total cache size when models become ready.
    ///
    /// - Parameter modelManager: The shared model manager actor.
    public func attach(modelManager: ModelManager) {
        Task {
            let states = await modelManager.states
            self.modelStates = states
            let size = await modelManager.totalCacheSize()
            self.totalCacheSizeBytes = size
        }

        let stream = modelManager.stateChanged
        observationTask?.cancel()
        observationTask = Task {
            for await (type, state) in stream {
                guard !Task.isCancelled else { break }
                self.modelStates[type] = state
                if case .ready = state {
                    let size = await modelManager.totalCacheSize()
                    self.totalCacheSizeBytes = size
                }
            }
        }
    }
}

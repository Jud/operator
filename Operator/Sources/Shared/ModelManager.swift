import Foundation
import OSLog

/// Manages model download, caching, and lifecycle state for all local inference models.
///
/// ModelManager is a Swift actor that tracks the state of three model types (STT, TTS, routing)
/// through their lifecycle: not downloaded, downloading, ready, loading, loaded, unloaded, or error.
/// It coordinates lazy loading (no models load at startup), coalesces concurrent download requests,
/// and publishes state changes via AsyncStream for Settings UI observation.
///
/// Each model type requires a registered download handler (provided by the engine implementations
/// in T4-T6) that performs the actual download. ModelManager wraps these with state tracking,
/// progress reporting, and cache directory management.
///
/// Cache location: ~/Library/Caches/com.operator.app/models/
///
/// Reference: design.md Section 3.5
public actor ModelManager {
    // MARK: - Type Aliases

    /// A download handler that performs the model-specific download operation.
    ///
    /// - Parameters:
    ///   - cacheDirectory: The directory where the model should be stored.
    ///   - progressCallback: Called with progress values from 0.0 to 1.0 during download.
    /// - Returns: The URL of the downloaded model within the cache directory.
    public typealias DownloadHandler =
        @Sendable (
            _ cacheDirectory: URL,
            _ progressCallback: @Sendable (Double) async -> Void
        ) async throws -> URL

    // MARK: - Subtypes

    /// Identifies which local inference model to manage.
    public enum ModelType: String, CaseIterable, Sendable {
        case stt = "parakeet-coreml"
        case tts = "qwen3-tts"
        case routing = "qwen3.5-0.8b-mlx-4bit"

        /// Human-readable label for UI display and logging.
        public var displayName: String {
            switch self {
            case .stt: return "Parakeet STT"

            case .tts: return "Qwen3-TTS"

            case .routing: return "Qwen 3.5 Routing"
            }
        }
    }

    /// Lifecycle state of a model, from initial download through loading and unloading.
    public enum ModelState: Sendable, Equatable {
        /// Model has never been downloaded to the cache directory.
        case notDownloaded
        /// Model is currently downloading. Progress is 0.0 to 1.0.
        case downloading(progress: Double)
        /// Model is downloaded and cached on disk, but not loaded into memory.
        case ready
        /// Model is being loaded into memory (e.g., CoreML compilation, MLX weight loading).
        case loading
        /// Model is loaded into memory and ready for inference.
        case loaded
        /// Model was previously loaded but has been released from memory (used for STT after transcription).
        case unloaded
        /// Model encountered an error during download or loading.
        case error(String)
    }

    // MARK: - Type Properties

    private static let logger = Log.logger(for: "ModelManager")

    // MARK: - Instance Properties

    /// Current state of each model, observable for Settings UI consumption.
    public private(set) var states: [ModelType: ModelState]

    /// Cache directory for downloaded models.
    public let cacheDirectory: URL

    /// Registered download handlers per model type, provided by engine implementations.
    private var downloadHandlers: [ModelType: DownloadHandler] = [:]

    /// In-flight download tasks, keyed by model type, to coalesce concurrent requests.
    private var activeDownloads: [ModelType: Task<URL, Error>] = [:]

    /// Continuation for broadcasting state changes to observers.
    private let stateChangedContinuation: AsyncStream<(ModelType, ModelState)>.Continuation

    /// Stream of state changes for Settings UI observation.
    ///
    /// Each element is a tuple of the model type and its new state. The stream
    /// is unbounded and never terminates during the application lifecycle.
    public let stateChanged: AsyncStream<(ModelType, ModelState)>

    // MARK: - Initialization

    /// Create a ModelManager with an optional custom cache directory.
    ///
    /// No models are downloaded or loaded at initialization. All model operations
    /// are lazy, triggered by explicit calls to `ensureDownloaded` or `ensureReady`.
    ///
    /// - Parameter cacheDirectory: Override the default cache directory. If nil, uses
    ///   `~/Library/Caches/com.operator.app/models/`.
    public init(cacheDirectory: URL? = nil) {
        let cacheDir =
            cacheDirectory
            ?? FileManager.default
            .urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("com.operator.app/models", isDirectory: true)
        self.cacheDirectory = cacheDir

        var initialStates: [ModelType: ModelState] = [:]
        for type in ModelType.allCases {
            initialStates[type] = .notDownloaded
        }
        self.states = initialStates

        let (stream, continuation) = AsyncStream<(ModelType, ModelState)>.makeStream()
        self.stateChanged = stream
        self.stateChangedContinuation = continuation

        Self.logger.info("ModelManager initialized with cache: \(cacheDir.path)")
    }

    // MARK: - Download Handler Registration

    /// Register a download handler for a specific model type.
    ///
    /// Engine implementations (ParakeetEngine, Qwen3TTSSpeechManager, MLXRoutingEngine)
    /// call this during their initialization to provide model-specific download logic.
    ///
    /// - Parameters:
    ///   - type: The model type this handler downloads.
    ///   - handler: The download handler closure.
    public func registerDownloadHandler(for type: ModelType, handler: @escaping DownloadHandler) {
        downloadHandlers[type] = handler
        Self.logger.debug("Download handler registered for \(type.displayName)")
    }

    // MARK: - Download & Cache Management

    /// Ensure a model is downloaded and return its cache path.
    ///
    /// If the model is already cached (state is `.ready`, `.loaded`, or `.unloaded`),
    /// returns the cache directory immediately. If a download is already in progress,
    /// awaits the existing download task rather than starting a new one.
    ///
    /// - Parameter type: The model type to ensure is downloaded.
    /// - Returns: The URL of the model within the cache directory.
    /// - Throws: If no download handler is registered, the download fails, or is cancelled.
    public func ensureDownloaded(_ type: ModelType) async throws -> URL {
        switch states[type] {
        case .ready, .loaded, .unloaded:
            return modelCacheDirectory(for: type)

        case .downloading:
            if let activeTask = activeDownloads[type] {
                return try await activeTask.value
            }
            throw ModelManagerError.noDownloadHandler(type)

        case .loading:
            return modelCacheDirectory(for: type)

        case .notDownloaded, .error, .none:
            break
        }

        guard let handler = downloadHandlers[type] else {
            throw ModelManagerError.noDownloadHandler(type)
        }

        let task = Task<URL, Error> { [cacheDirectory] in
            let modelDir = cacheDirectory.appendingPathComponent(type.rawValue, isDirectory: true)

            try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

            let result = try await handler(modelDir) { [weak self] progress in
                await self?.updateState(type, to: .downloading(progress: progress))
            }

            return result
        }

        activeDownloads[type] = task
        updateStateSync(type, to: .downloading(progress: 0.0))

        Self.logger.info("Starting download for \(type.displayName)")

        do {
            let url = try await task.value
            activeDownloads[type] = nil
            updateStateSync(type, to: .ready)
            Self.logger.info("Download complete for \(type.displayName)")
            return url
        } catch {
            activeDownloads[type] = nil
            let message = String(describing: error)
            updateStateSync(type, to: .error(message))
            Self.logger.error("Download failed for \(type.displayName): \(message)")
            throw error
        }
    }

    /// Check if a model is cached on disk without triggering a download.
    ///
    /// Useful for determining initial state on launch and for Settings UI
    /// to show whether models need downloading.
    ///
    /// - Parameter type: The model type to check.
    /// - Returns: `true` if the model directory exists in the cache.
    public func isCached(_ type: ModelType) -> Bool {
        let modelDir = modelCacheDirectory(for: type)
        return FileManager.default.fileExists(atPath: modelDir.path)
    }

    /// Scan the cache directory and update states for any previously downloaded models.
    ///
    /// Called once after initialization to detect models that were downloaded in
    /// a previous session. Does not load any models into memory.
    public func scanCachedModels() {
        for type in ModelType.allCases where isCached(type) {
            if case .notDownloaded = states[type] {
                updateStateSync(type, to: .ready)
                Self.logger.info("Found cached model: \(type.displayName)")
            }
        }
    }

    /// Get the cache directory path for a specific model type.
    ///
    /// - Parameter type: The model type.
    /// - Returns: The directory URL where this model type is stored.
    public func modelCacheDirectory(for type: ModelType) -> URL {
        cacheDirectory.appendingPathComponent(type.rawValue, isDirectory: true)
    }

    /// Calculate the total disk space used by cached models.
    ///
    /// - Returns: Total bytes used by all cached models, or 0 if calculation fails.
    public func totalCacheSize() -> UInt64 {
        var total: UInt64 = 0
        let fileManager = FileManager.default

        for type in ModelType.allCases {
            let modelDir = modelCacheDirectory(for: type)
            guard
                let enumerator = fileManager.enumerator(
                    at: modelDir,
                    includingPropertiesForKeys: [.fileSizeKey]
                )
            else {
                continue
            }
            for case let fileURL as URL in enumerator {
                if let size = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                    total += UInt64(size)
                }
            }
        }

        return total
    }

    // MARK: - Lifecycle State Transitions

    /// Mark a model as currently loading into memory.
    ///
    /// Called by engine implementations when they begin loading a model
    /// into memory for inference.
    ///
    /// - Parameter type: The model type being loaded.
    public func markLoading(_ type: ModelType) {
        updateStateSync(type, to: .loading)
        Self.logger.info("\(type.displayName) loading into memory")
    }

    /// Mark a model as loaded and ready for inference.
    ///
    /// Called by engine implementations after successful model loading.
    ///
    /// - Parameter type: The model type that is now loaded.
    public func markLoaded(_ type: ModelType) {
        updateStateSync(type, to: .loaded)
        Self.logger.info("\(type.displayName) loaded into memory")
    }

    /// Mark a model as unloaded, releasing memory.
    ///
    /// Called after STT transcription completes (STT model is transient) or
    /// when manually releasing a model. The model remains cached on disk
    /// (state becomes `.unloaded`, not `.notDownloaded`).
    ///
    /// - Parameter type: The model type to mark as unloaded.
    public func markUnloaded(_ type: ModelType) {
        updateStateSync(type, to: .unloaded)
        Self.logger.info("\(type.displayName) unloaded from memory")
    }

    /// Mark a model as having encountered an error.
    ///
    /// - Parameters:
    ///   - type: The model type.
    ///   - message: Human-readable error description.
    public func markError(_ type: ModelType, message: String) {
        updateStateSync(type, to: .error(message))
        Self.logger.error("\(type.displayName) error: \(message)")
    }

    /// Get the current state of a specific model.
    ///
    /// - Parameter type: The model type to query.
    /// - Returns: The current lifecycle state.
    public func state(for type: ModelType) -> ModelState {
        states[type] ?? .notDownloaded
    }

    // MARK: - Private Helpers

    /// Update model state synchronously within the actor and notify observers.
    private func updateStateSync(_ type: ModelType, to newState: ModelState) {
        states[type] = newState
        stateChangedContinuation.yield((type, newState))
    }

    /// Update model state asynchronously (for use from download handler callbacks).
    private func updateState(_ type: ModelType, to newState: ModelState) {
        updateStateSync(type, to: newState)
    }
}

// MARK: - Errors

/// Errors produced by ModelManager operations.
public enum ModelManagerError: Error, CustomStringConvertible {
    /// No download handler has been registered for the requested model type.
    case noDownloadHandler(ModelManager.ModelType)

    /// The model download failed.
    case downloadFailed(ModelManager.ModelType, String)

    /// Human-readable description of the error for logging and display.
    public var description: String {
        switch self {
        case .noDownloadHandler(let type):
            return "No download handler registered for \(type.displayName)"

        case .downloadFailed(let type, let reason):
            return "Download failed for \(type.displayName): \(reason)"
        }
    }
}

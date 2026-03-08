import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXStructured
import os

/// Errors produced by ``MLXRoutingEngine`` operations.
public enum MLXRoutingError: Error, CustomStringConvertible {
    /// The model failed to load into memory.
    case modelLoadFailed(String)

    /// The model is not loaded and cannot process requests.
    case modelNotLoaded

    /// The model output could not be parsed as valid JSON.
    case invalidOutput(String)

    /// The operation exceeded its timeout.
    case timeout(TimeInterval)

    /// A human-readable description of the routing error.
    public var description: String {
        switch self {
        case .modelLoadFailed(let reason):
            return "MLX routing model failed to load: \(reason)"

        case .modelNotLoaded:
            return "MLX routing model is not loaded"

        case .invalidOutput(let output):
            return "MLX routing model produced invalid output: \(output.prefix(200))"

        case .timeout(let seconds):
            return "MLX routing timed out after \(Int(seconds))s"
        }
    }
}

/// Local routing engine using Qwen 3.5 0.8B (4-bit quantized) via mlx-swift-lm
/// with grammar-constrained JSON decoding via mlx-swift-structured.
///
/// Conforms to ``RoutingEngine`` and produces the same JSON output format as
/// ``ClaudePipeRoutingEngine``, making it a transparent drop-in replacement.
/// The model loads lazily on first routing event and remains resident in memory
/// for fast repeated inference (~50-200ms per classification).
///
/// Grammar-constrained decoding via xgrammar guarantees that the model output
/// is always valid JSON matching the routing schema, eliminating parse failures.
/// Temperature 0 (ArgMaxSampler) ensures deterministic output.
///
/// Thread safety: all mutable state (``modelContainer``, ``isLoading``) is
/// accessed only from async methods called sequentially by ``MessageRouter``.
/// Marked `@unchecked Sendable` following the same pattern as ``ParakeetEngine``
/// and ``AdaptiveRoutingEngine``.
public final class MLXRoutingEngine: RoutingEngine, @unchecked Sendable {
    private static let logger = Log.logger(for: "MLXRoutingEngine")

    /// HuggingFace model identifier for the 4-bit quantized Qwen 3.5 0.8B.
    static let modelID = "mlx-community/Qwen3.5-0.8B-MLX-4bit"

    /// Maximum tokens to generate.
    ///
    /// Routing output is small (~50-80 tokens).
    private static let maxGenerationTokens = 150

    // MARK: - Properties

    private let modelManager: ModelManager
    private var modelContainer: ModelContainer?
    private var isLoading = false
    private let modelConfiguration: ModelConfiguration

    /// Cached grammar processor for the routing schema, reused across calls.
    private var grammarProcessor: GrammarMaskedLogitProcessor?

    // MARK: - Initialization

    /// Creates an MLX routing engine with the given model manager.
    ///
    /// Registers a download handler with ModelManager for Settings UI model downloads.
    /// No model loading occurs at construction time.
    ///
    /// - Parameter modelManager: The shared model manager for lifecycle tracking.
    public init(modelManager: ModelManager) {
        self.modelManager = modelManager
        self.modelConfiguration = ModelConfiguration(id: Self.modelID)
        registerDownloadHandler()
    }

    // MARK: - Generation

    /// Run grammar-constrained generation within a ModelContainer's perform block.
    private static func generateConstrained(
        prompt: String,
        context: ModelContext,
        configuration: ModelConfiguration,
        maxTokens: Int
    ) async throws -> String {
        let userInput = UserInput(
            chat: [
                .system(RoutingPrompt.systemInstruction),
                .user(prompt)
            ]
        )
        let input = try await context.processor.prepare(input: userInput)

        let grammar = Grammar.schema(RoutingPrompt.outputSchemaString)
        let parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)

        let result = try await MLXStructured.generate(
            input: input,
            parameters: parameters,
            context: context,
            grammar: grammar
        )

        return result.output
    }

    // MARK: - RoutingEngine

    /// Run a routing prompt through the local Qwen 3.5 0.8B model and return parsed JSON.
    ///
    /// Loads the model lazily on first call. Wraps the prompt with a system instruction
    /// optimized for the 0.8B model and enforces JSON output via grammar-constrained decoding.
    ///
    /// - Parameters:
    ///   - prompt: The routing prompt built by MessageRouter (contains session context).
    ///   - timeout: Maximum time to wait. Applied as a Task cancellation deadline.
    /// - Returns: A dictionary parsed from the constrained JSON output.
    /// - Throws: ``MLXRoutingError`` on model load failure, invalid output, or timeout.
    public func run(prompt: String, timeout: TimeInterval) async throws -> [String: Any] {
        try await ensureModelLoaded()

        guard let container = modelContainer else {
            throw MLXRoutingError.modelNotLoaded
        }

        let wrappedPrompt = RoutingPrompt.wrapForLocalModel(prompt)
        let configuration = modelConfiguration
        let maxTokens = Self.maxGenerationTokens

        Self.logger.info("Running local routing inference")

        let jsonString: String = try await withThrowingTaskGroup(of: String.self) { group in
            group.addTask {
                try await container.perform { [wrappedPrompt] context in
                    try await Self.generateConstrained(
                        prompt: wrappedPrompt,
                        context: context,
                        configuration: configuration,
                        maxTokens: maxTokens
                    )
                }
            }

            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
                throw MLXRoutingError.timeout(timeout)
            }

            guard let result = try await group.next() else {
                throw MLXRoutingError.modelNotLoaded
            }
            group.cancelAll()
            return result
        }

        Self.logger.debug("Local routing output: \(jsonString.prefix(200))")

        guard let data = jsonString.data(using: .utf8),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            throw MLXRoutingError.invalidOutput(jsonString)
        }

        return json
    }

    // MARK: - Model Lifecycle

    /// Load the routing model if not already in memory.
    ///
    /// Uses ``LLMModelFactory`` to download (if needed) and load the model from
    /// HuggingFace Hub. The model remains resident after loading for the application
    /// session (~500MB resident memory).
    private func ensureModelLoaded() async throws {
        guard modelContainer == nil else {
            return
        }
        guard !isLoading else {
            while isLoading {
                try await Task.sleep(nanoseconds: 50_000_000)
            }
            return
        }

        isLoading = true
        await modelManager.markLoading(.routing)
        Self.logger.info("Loading routing model: \(Self.modelID)")

        do {
            let container = try await LLMModelFactory.shared.loadContainer(
                configuration: modelConfiguration
            ) { progress in
                Self.logger.debug(
                    "Routing model download: \(Int(progress.fractionCompleted * 100))%"
                )
            }
            modelContainer = container
            isLoading = false
            await modelManager.markLoaded(.routing)
            Self.logger.info("Routing model loaded and ready")
        } catch {
            isLoading = false
            await modelManager.markError(.routing, message: error.localizedDescription)
            throw MLXRoutingError.modelLoadFailed(error.localizedDescription)
        }
    }

    /// Register the download handler with ModelManager for Settings UI downloads.
    private func registerDownloadHandler() {
        let manager = modelManager
        let modelID = Self.modelID
        // swiftlint:disable:next unhandled_throwing_task
        Task {
            await manager.registerDownloadHandler(for: .routing) { _, progressCallback in
                await progressCallback(0.0)
                let configuration = ModelConfiguration(id: modelID)
                _ = try await LLMModelFactory.shared.loadContainer(
                    configuration: configuration
                )
                await progressCallback(1.0)
                let cacheDir = FileManager.default
                    .urls(for: .cachesDirectory, in: .userDomainMask)[0]
                    .appendingPathComponent("huggingface/hub", isDirectory: true)
                return cacheDir
            }
        }
    }
}

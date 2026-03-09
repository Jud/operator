import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXStructured
import os

// swiftlint:disable type_contents_order

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
public final class MLXRoutingEngine: RoutingEngine, @unchecked Sendable {
    static let logger = Log.logger(for: "MLXRoutingEngine")
    static let modelID = "mlx-community/Qwen3.5-0.8B-MLX-4bit"
    static let maxGenerationTokens = 32
    static let prefillStepSize = 512

    let modelManager: ModelManager
    var modelContainer: ModelContainer?
    var isLoading = false
    let modelConfiguration: ModelConfiguration
    var grammarProcessor: GrammarMaskedLogitProcessor?
    var promptPrefixCache: MLXPromptPrefixCache?

    /// Creates an MLX routing engine with the given model manager.
    public init(modelManager: ModelManager) {
        self.modelManager = modelManager
        self.modelConfiguration = ModelConfiguration(id: Self.modelID)
        registerDownloadHandler()
    }

    /// Run a routing prompt through the local model and return parsed JSON.
    public func run(prompt: String, timeout: TimeInterval) async throws -> [String: Any] {
        try await runProfiled(prompt: prompt, timeout: timeout).json
    }

    /// Run a routing prompt and capture phase timings for benchmarking.
    public func runProfiled(
        prompt: String,
        timeout: TimeInterval
    ) async throws -> MLXRoutingProfiledResult {
        let totalStart = CFAbsoluteTimeGetCurrent()
        let promptWords = Self.wordCount(in: prompt)
        let modelWasLoadedBeforeRun = modelContainer != nil

        let modelLoadMs = try await measureAsyncMilliseconds {
            try await ensureModelLoaded()
        }
        guard let container = modelContainer else {
            throw MLXRoutingError.modelNotLoaded
        }

        let grammarProfile = try await loadGrammarProcessor(using: container)
        let promptRuntime = try await resolvePromptRuntime(for: prompt, using: container)
        Self.logger.info("Running local routing inference")

        let inferenceProfile = try await runTimedInference(
            container: container,
            runtimePrompt: promptRuntime.runtimePrompt,
            processor: grammarProfile.processor,
            prefixBaseOffset: promptRuntime.prefixProfile?.baseOffset,
            timeout: timeout
        )
        let jsonProfile = try parseRoutingJSON(from: inferenceProfile.generation.output)
        let totalMs = (CFAbsoluteTimeGetCurrent() - totalStart) * 1_000

        let metrics = makeRunMetrics(
            prompt: prompt,
            promptWords: promptWords,
            runtime: promptRuntime,
            modelLoadMs: modelLoadMs,
            grammarProfile: grammarProfile,
            inferenceProfile: inferenceProfile,
            jsonParseMs: jsonProfile.jsonParseMs,
            totalMs: totalMs,
            modelWasLoadedBeforeRun: modelWasLoadedBeforeRun
        )

        return MLXRoutingProfiledResult(
            json: jsonProfile.json,
            rawOutput: inferenceProfile.generation.output,
            metrics: metrics
        )
    }

    // swiftlint:disable:next function_body_length function_parameter_count
    func makeRunMetrics(
        prompt: String,
        promptWords: Int,
        runtime: MLXPromptRuntime,
        modelLoadMs: Double,
        grammarProfile: MLXGrammarProfile,
        inferenceProfile: MLXInferenceProfile,
        jsonParseMs: Double,
        totalMs: Double,
        modelWasLoadedBeforeRun: Bool
    ) -> MLXRoutingRunMetrics {
        let generation = inferenceProfile.generation
        let cachedContext = runtime.cachedContextPrompt ?? ""
        let taskGroupOverheadMs = max(
            0,
            inferenceProfile.inferenceRoundTripMs - inferenceProfile.containerPerformRoundTripMs
        )
        let containerPerformSchedulingMs = max(
            0,
            inferenceProfile.containerPerformRoundTripMs - generation.closureMs
        )
        let inferenceOverheadMs = max(
            0,
            inferenceProfile.inferenceRoundTripMs
                - generation.promptPreparationMs
                - generation.tokenIteratorInitMs
                - generation.generationMs
        )

        return MLXRoutingRunMetrics(
            promptCharacters: prompt.count,
            promptWords: promptWords,
            cachedContextCharacters: cachedContext.count,
            cachedContextWords: Self.wordCount(in: cachedContext),
            runtimePromptCharacters: runtime.runtimePrompt.count,
            runtimePromptWords: Self.wordCount(in: runtime.runtimePrompt),
            modelLoadMs: modelLoadMs,
            grammarSetupMs: grammarProfile.setupMs,
            promptPrefixCacheMs: runtime.prefixProfile?.buildMs ?? 0,
            inferenceRoundTripMs: inferenceProfile.inferenceRoundTripMs,
            taskGroupOverheadMs: taskGroupOverheadMs,
            containerPerformRoundTripMs: inferenceProfile.containerPerformRoundTripMs,
            containerPerformSchedulingMs: containerPerformSchedulingMs,
            containerClosureMs: generation.closureMs,
            inferenceOverheadMs: inferenceOverheadMs,
            promptPreparationMs: generation.promptPreparationMs,
            tokenIteratorInitMs: generation.tokenIteratorInitMs,
            generationMs: generation.generationMs,
            jsonParseMs: jsonParseMs,
            totalMs: totalMs,
            usedCachedGrammar: grammarProfile.usedCachedGrammar,
            usedPromptPrefixCache: runtime.prefixProfile != nil,
            promptPrefixCacheHit: runtime.prefixProfile?.cacheHit ?? false,
            modelWasLoadedBeforeRun: modelWasLoadedBeforeRun
        )
    }

    static func wordCount(in text: String) -> Int {
        text.split { $0.isWhitespace || $0.isNewline }.count
    }
}

// swiftlint:enable type_contents_order

import Foundation

/// Phase timings captured for a single MLX routing inference.
public struct MLXRoutingRunMetrics {
    /// Character count of the full prompt provided to the routing engine.
    public let promptCharacters: Int

    /// Word count of the full prompt provided to the routing engine.
    public let promptWords: Int

    /// Character count of the reusable cached session-context prefix.
    public let cachedContextCharacters: Int

    /// Word count of the reusable cached session-context prefix.
    public let cachedContextWords: Int

    /// Character count of the dynamic prompt built for the current routing turn.
    public let runtimePromptCharacters: Int

    /// Word count of the dynamic prompt built for the current routing turn.
    public let runtimePromptWords: Int

    /// Time spent loading the MLX model for this run.
    public let modelLoadMs: Double

    /// Time spent creating or retrieving the grammar processor.
    public let grammarSetupMs: Double

    /// Time spent building the reusable prompt-prefix KV cache.
    public let promptPrefixCacheMs: Double

    /// End-to-end time spent waiting for the inference task to complete.
    public let inferenceRoundTripMs: Double

    /// Overhead introduced by the timeout task group around inference.
    public let taskGroupOverheadMs: Double

    /// End-to-end time spent inside `ModelContainer.perform`.
    public let containerPerformRoundTripMs: Double

    /// Time between entering `ModelContainer.perform` and entering the closure body.
    public let containerPerformSchedulingMs: Double

    /// Total time spent inside the `ModelContainer.perform` closure body.
    public let containerClosureMs: Double

    /// Inference round-trip time not explained by prompt prep, iterator init, or generation.
    public let inferenceOverheadMs: Double

    /// Time spent converting the chat prompt into model input tensors.
    public let promptPreparationMs: Double

    /// Time spent constructing the `TokenIterator`, including prompt prefill.
    public let tokenIteratorInitMs: Double

    /// Time spent generating constrained output tokens.
    public let generationMs: Double

    /// Time spent parsing the constrained JSON output.
    public let jsonParseMs: Double

    /// Total end-to-end time for the profiled routing call.
    public let totalMs: Double

    /// Whether the grammar processor was already cached before this run.
    public let usedCachedGrammar: Bool

    /// Whether this run attempted to use the prompt-prefix cache.
    public let usedPromptPrefixCache: Bool

    /// Whether the prompt-prefix cache lookup hit an existing prefetched context.
    public let promptPrefixCacheHit: Bool

    /// Whether the routing model was already loaded before this run started.
    public let modelWasLoadedBeforeRun: Bool

    /// Create a fully populated routing metrics snapshot.
    public init(
        promptCharacters: Int,
        promptWords: Int,
        cachedContextCharacters: Int,
        cachedContextWords: Int,
        runtimePromptCharacters: Int,
        runtimePromptWords: Int,
        modelLoadMs: Double,
        grammarSetupMs: Double,
        promptPrefixCacheMs: Double,
        inferenceRoundTripMs: Double,
        taskGroupOverheadMs: Double,
        containerPerformRoundTripMs: Double,
        containerPerformSchedulingMs: Double,
        containerClosureMs: Double,
        inferenceOverheadMs: Double,
        promptPreparationMs: Double,
        tokenIteratorInitMs: Double,
        generationMs: Double,
        jsonParseMs: Double,
        totalMs: Double,
        usedCachedGrammar: Bool,
        usedPromptPrefixCache: Bool,
        promptPrefixCacheHit: Bool,
        modelWasLoadedBeforeRun: Bool
    ) {
        self.promptCharacters = promptCharacters
        self.promptWords = promptWords
        self.cachedContextCharacters = cachedContextCharacters
        self.cachedContextWords = cachedContextWords
        self.runtimePromptCharacters = runtimePromptCharacters
        self.runtimePromptWords = runtimePromptWords
        self.modelLoadMs = modelLoadMs
        self.grammarSetupMs = grammarSetupMs
        self.promptPrefixCacheMs = promptPrefixCacheMs
        self.inferenceRoundTripMs = inferenceRoundTripMs
        self.taskGroupOverheadMs = taskGroupOverheadMs
        self.containerPerformRoundTripMs = containerPerformRoundTripMs
        self.containerPerformSchedulingMs = containerPerformSchedulingMs
        self.containerClosureMs = containerClosureMs
        self.inferenceOverheadMs = inferenceOverheadMs
        self.promptPreparationMs = promptPreparationMs
        self.tokenIteratorInitMs = tokenIteratorInitMs
        self.generationMs = generationMs
        self.jsonParseMs = jsonParseMs
        self.totalMs = totalMs
        self.usedCachedGrammar = usedCachedGrammar
        self.usedPromptPrefixCache = usedPromptPrefixCache
        self.promptPrefixCacheHit = promptPrefixCacheHit
        self.modelWasLoadedBeforeRun = modelWasLoadedBeforeRun
    }
}

/// Parsed routing output plus phase timings for a single inference.
public struct MLXRoutingProfiledResult {
    /// Parsed JSON routing payload.
    public let json: [String: Any]

    /// Raw constrained model output before JSON parsing.
    public let rawOutput: String

    /// Detailed timing data for the routing call.
    public let metrics: MLXRoutingRunMetrics

    /// Create a profiled routing result.
    public init(json: [String: Any], rawOutput: String, metrics: MLXRoutingRunMetrics) {
        self.json = json
        self.rawOutput = rawOutput
        self.metrics = metrics
    }
}

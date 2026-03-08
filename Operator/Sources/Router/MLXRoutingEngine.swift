import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXStructured
import os

/// Phase timings captured for a single MLX routing inference.
public struct MLXRoutingRunMetrics {
    public let promptCharacters: Int
    public let promptWords: Int
    public let cachedContextCharacters: Int
    public let cachedContextWords: Int
    public let runtimePromptCharacters: Int
    public let runtimePromptWords: Int
    public let modelLoadMs: Double
    public let grammarSetupMs: Double
    public let promptPrefixCacheMs: Double
    public let inferenceRoundTripMs: Double
    public let taskGroupOverheadMs: Double
    public let containerPerformRoundTripMs: Double
    public let containerPerformSchedulingMs: Double
    public let containerClosureMs: Double
    public let inferenceOverheadMs: Double
    public let promptPreparationMs: Double
    public let tokenIteratorInitMs: Double
    public let generationMs: Double
    public let jsonParseMs: Double
    public let totalMs: Double
    public let usedCachedGrammar: Bool
    public let usedPromptPrefixCache: Bool
    public let promptPrefixCacheHit: Bool
    public let modelWasLoadedBeforeRun: Bool
}

/// Parsed routing output plus phase timings for a single inference.
public struct MLXRoutingProfiledResult {
    public let json: [String: Any]
    public let rawOutput: String
    public let metrics: MLXRoutingRunMetrics
}

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
    private static let maxGenerationTokens = 32
    private static let prefillStepSize = 512

    // MARK: - Properties

    private let modelManager: ModelManager
    private var modelContainer: ModelContainer?
    private var isLoading = false
    private let modelConfiguration: ModelConfiguration

    /// Cached grammar processor for the routing schema, reused across calls.
    private var grammarProcessor: GrammarMaskedLogitProcessor?

    private struct PromptPrefixCache {
        let key: String
        let baseOffset: Int
        let cache: [KVCache]
    }

    private var promptPrefixCache: PromptPrefixCache?

    private struct GenerationProfile {
        let output: String
        let closureMs: Double
        let promptPreparationMs: Double
        let tokenIteratorInitMs: Double
        let generationMs: Double
    }

    private struct InferenceProfile {
        let generation: GenerationProfile
        let containerPerformRoundTripMs: Double
    }

    private struct PrefixCacheProfile: Sendable {
        let baseOffset: Int
        let buildMs: Double
        let cacheHit: Bool
    }

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
        messages: [Chat.Message],
        context: ModelContext,
        processor: GrammarMaskedLogitProcessor,
        maxTokens: Int,
        prefixCache: [KVCache]? = nil,
        prefixBaseOffset: Int? = nil
    ) async throws -> GenerationProfile {
        let closureStart = CFAbsoluteTimeGetCurrent()
        let prepareStart = CFAbsoluteTimeGetCurrent()
        let userInput = UserInput(chat: messages)
        let input = try await context.processor.prepare(input: userInput)
        let promptPreparationMs = (CFAbsoluteTimeGetCurrent() - prepareStart) * 1000

        let sampler = GenerateParameters(temperature: 0).sampler()
        processor.grammarMatcher.reset()

        let prefixOffset = prefixBaseOffset ?? prefixCache?.first?.offset ?? 0
        defer {
            if let prefixCache {
                let currentOffset = prefixCache.first?.offset ?? prefixOffset
                let delta = max(0, currentOffset - prefixOffset)
                if delta > 0 {
                    trimPromptCache(prefixCache, numTokens: delta)
                }
            }
        }

        let iteratorStart = CFAbsoluteTimeGetCurrent()
        let iterator = try TokenIterator(
            input: input,
            model: context.model,
            cache: prefixCache,
            processor: processor,
            sampler: sampler,
            prefillStepSize: Self.prefillStepSize,
            maxTokens: maxTokens
        )
        let tokenIteratorInitMs = (CFAbsoluteTimeGetCurrent() - iteratorStart) * 1000

        let didGenerate: ([Int]) -> GenerateDisposition = { _ in
            if processor.grammarMatcher.isTerminated() {
                return .stop
            }
            return .more
        }
        let generationStart = CFAbsoluteTimeGetCurrent()
        let result: GenerateResult = MLXLMCommon.generate(
            input: input,
            context: context,
            iterator: iterator,
            didGenerate: didGenerate
        )
        let generationMs = (CFAbsoluteTimeGetCurrent() - generationStart) * 1000
        let closureMs = (CFAbsoluteTimeGetCurrent() - closureStart) * 1000

        return GenerationProfile(
            output: result.output,
            closureMs: closureMs,
            promptPreparationMs: promptPreparationMs,
            tokenIteratorInitMs: tokenIteratorInitMs,
            generationMs: generationMs
        )
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

        let modelLoadStart = CFAbsoluteTimeGetCurrent()
        try await ensureModelLoaded()
        let modelLoadMs = (CFAbsoluteTimeGetCurrent() - modelLoadStart) * 1000

        guard let container = modelContainer else {
            throw MLXRoutingError.modelNotLoaded
        }

        let maxTokens = Self.maxGenerationTokens
        let usedCachedGrammar = grammarProcessor != nil

        // Build/cache the grammar processor (first call loads tokenizer, ~1s; subsequent calls instant)
        let grammarStart = CFAbsoluteTimeGetCurrent()
        let processor = try await container.perform { [self] context in
            try await self.ensureGrammarProcessor(context: context)
        }
        let grammarSetupMs = (CFAbsoluteTimeGetCurrent() - grammarStart) * 1000

        let promptParts = RoutingPrompt.splitForLocalModel(prompt)
        let cachedContextPrompt = promptParts?.context
        let runtimePrompt: String
        let promptPrefixProfile: PrefixCacheProfile?

        if let promptParts {
            runtimePrompt = RoutingPrompt.buildLocalModelCurrentTurn(promptParts.currentMessage)
            let prefixProfile = try await container.perform { [self] context in
                try await self.ensurePromptPrefixCache(
                    contextPrompt: promptParts.context,
                    context: context
                )
            }
            promptPrefixProfile = prefixProfile
        } else {
            runtimePrompt = RoutingPrompt.wrapForLocalModel(prompt)
            promptPrefixProfile = nil
        }

        let cachedContextWords = Self.wordCount(in: cachedContextPrompt ?? "")
        let runtimePromptWords = Self.wordCount(in: runtimePrompt)

        Self.logger.info("Running local routing inference")

        let inferenceStart = CFAbsoluteTimeGetCurrent()
        let inferenceProfile: InferenceProfile = try await withThrowingTaskGroup(
            of: InferenceProfile.self
        ) { group in
            let prefixBaseOffset = promptPrefixProfile?.baseOffset
            group.addTask { [self, runtimePrompt, prefixBaseOffset] in
                let performStart = CFAbsoluteTimeGetCurrent()
                let profile = try await container.perform { [self] context in
                    try await Self.generateConstrained(
                        messages: prefixBaseOffset == nil
                            ? [
                                .system(RoutingPrompt.systemInstruction),
                                .user(runtimePrompt)
                            ]
                            : [.user(runtimePrompt)],
                        context: context,
                        processor: processor,
                        maxTokens: maxTokens,
                        prefixCache: self.promptPrefixCache?.cache,
                        prefixBaseOffset: prefixBaseOffset
                    )
                }
                return InferenceProfile(
                    generation: profile,
                    containerPerformRoundTripMs: (CFAbsoluteTimeGetCurrent() - performStart) * 1000
                )
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
        let inferenceRoundTripMs = (CFAbsoluteTimeGetCurrent() - inferenceStart) * 1000
        let generationProfile = inferenceProfile.generation
        let containerPerformRoundTripMs = inferenceProfile.containerPerformRoundTripMs
        let jsonString = generationProfile.output

        Self.logger.debug("Local routing output: \(jsonString.prefix(200))")

        let parseStart = CFAbsoluteTimeGetCurrent()
        guard let data = jsonString.data(using: .utf8),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            throw MLXRoutingError.invalidOutput(jsonString)
        }
        let jsonParseMs = (CFAbsoluteTimeGetCurrent() - parseStart) * 1000
        let totalMs = (CFAbsoluteTimeGetCurrent() - totalStart) * 1000
        let taskGroupOverheadMs = max(0, inferenceRoundTripMs - containerPerformRoundTripMs)
        let containerPerformSchedulingMs = max(
            0,
            containerPerformRoundTripMs - generationProfile.closureMs
        )
        let inferenceOverheadMs = max(
            0,
            inferenceRoundTripMs - generationProfile.promptPreparationMs
                - generationProfile.tokenIteratorInitMs - generationProfile.generationMs
        )

        let metrics = MLXRoutingRunMetrics(
            promptCharacters: prompt.count,
            promptWords: promptWords,
            cachedContextCharacters: cachedContextPrompt?.count ?? 0,
            cachedContextWords: cachedContextWords,
            runtimePromptCharacters: runtimePrompt.count,
            runtimePromptWords: runtimePromptWords,
            modelLoadMs: modelLoadMs,
            grammarSetupMs: grammarSetupMs,
            promptPrefixCacheMs: promptPrefixProfile?.buildMs ?? 0,
            inferenceRoundTripMs: inferenceRoundTripMs,
            taskGroupOverheadMs: taskGroupOverheadMs,
            containerPerformRoundTripMs: containerPerformRoundTripMs,
            containerPerformSchedulingMs: containerPerformSchedulingMs,
            containerClosureMs: generationProfile.closureMs,
            inferenceOverheadMs: inferenceOverheadMs,
            promptPreparationMs: generationProfile.promptPreparationMs,
            tokenIteratorInitMs: generationProfile.tokenIteratorInitMs,
            generationMs: generationProfile.generationMs,
            jsonParseMs: jsonParseMs,
            totalMs: totalMs,
            usedCachedGrammar: usedCachedGrammar,
            usedPromptPrefixCache: promptPrefixProfile != nil,
            promptPrefixCacheHit: promptPrefixProfile?.cacheHit ?? false,
            modelWasLoadedBeforeRun: modelWasLoadedBeforeRun
        )

        return MLXRoutingProfiledResult(json: json, rawOutput: jsonString, metrics: metrics)
    }

    // MARK: - Model Lifecycle

    /// Build or return the cached grammar processor for the routing schema.
    ///
    /// The processor loads tokenizer vocab from Hub configuration, which takes ~1s.
    /// Caching it eliminates this overhead on subsequent calls.
    private func ensureGrammarProcessor(context: ModelContext) async throws -> GrammarMaskedLogitProcessor {
        if let cached = grammarProcessor {
            return cached
        }
        let grammar = Grammar.schema(RoutingPrompt.outputSchemaString)
        let processor = try await GrammarMaskedLogitProcessor.from(
            configuration: context.configuration,
            grammar: grammar
        )
        grammarProcessor = processor
        return processor
    }

    /// Build or return a cached prompt-prefix KV cache for the current session snapshot.
    private func ensurePromptPrefixCache(
        contextPrompt: String,
        context: ModelContext
    ) async throws -> PrefixCacheProfile {
        if let cached = promptPrefixCache, cached.key == contextPrompt {
            return PrefixCacheProfile(
                baseOffset: cached.baseOffset,
                buildMs: 0,
                cacheHit: true
            )
        }

        let buildStart = CFAbsoluteTimeGetCurrent()
        let input = try await context.processor.prepare(
            input: UserInput(
                chat: [
                    .system(RoutingPrompt.systemInstruction),
                    .user(contextPrompt)
                ]
            )
        )
        let cache = try Self.prefillPromptCache(input: input, context: context)
        let baseOffset = cache.first?.offset ?? 0
        let profile = PrefixCacheProfile(
            baseOffset: baseOffset,
            buildMs: (CFAbsoluteTimeGetCurrent() - buildStart) * 1000,
            cacheHit: false
        )
        promptPrefixCache = PromptPrefixCache(
            key: contextPrompt,
            baseOffset: baseOffset,
            cache: cache
        )
        return profile
    }

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

    /// Prefill a prompt into a reusable KV cache without generating output tokens.
    private static func prefillPromptCache(
        input: LMInput,
        context: ModelContext
    ) throws -> [KVCache] {
        let cache = context.model.newCache(parameters: nil)

        switch try context.model.prepare(input, cache: cache, windowSize: Self.prefillStepSize) {
        case .tokens(let tokens):
            let result = context.model(
                tokens[text: .newAxis],
                cache: cache.isEmpty ? nil : cache,
                state: nil
            )
            eval(result.logits)

        case .logits(let result):
            eval(result.logits)
        }

        return cache
    }

    private static func wordCount(in text: String) -> Int {
        text.split { $0.isWhitespace || $0.isNewline }.count
    }
}

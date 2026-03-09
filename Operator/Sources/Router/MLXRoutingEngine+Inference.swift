import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXStructured

// swiftlint:disable type_contents_order

internal struct MLXPromptPrefixCache {
    let key: String
    let baseOffset: Int
    let cache: [KVCache]
}

internal struct MLXGenerationProfile {
    let output: String
    let closureMs: Double
    let promptPreparationMs: Double
    let tokenIteratorInitMs: Double
    let generationMs: Double
}

internal struct MLXTimedGeneration {
    let generation: MLXGenerationProfile
    let containerPerformRoundTripMs: Double
}

internal struct MLXInferenceProfile {
    let generation: MLXGenerationProfile
    let inferenceRoundTripMs: Double
    let containerPerformRoundTripMs: Double
}

internal struct MLXPrefixCacheProfile: Sendable {
    let baseOffset: Int
    let buildMs: Double
    let cacheHit: Bool
}

internal struct MLXGrammarProfile {
    let processor: GrammarMaskedLogitProcessor
    let setupMs: Double
    let usedCachedGrammar: Bool
}

internal struct MLXPromptRuntime {
    let cachedContextPrompt: String?
    let runtimePrompt: String
    let prefixProfile: MLXPrefixCacheProfile?
}

internal struct MLXJSONParseProfile {
    let json: [String: Any]
    let jsonParseMs: Double
}

extension MLXRoutingEngine {
    func loadGrammarProcessor(using container: ModelContainer) async throws -> MLXGrammarProfile {
        let usedCachedGrammar = grammarProcessor != nil
        let setupStart = CFAbsoluteTimeGetCurrent()
        let processor = try await container.perform { [self] context in
            try await self.ensureGrammarProcessor(context: context)
        }
        return MLXGrammarProfile(
            processor: processor,
            setupMs: (CFAbsoluteTimeGetCurrent() - setupStart) * 1_000,
            usedCachedGrammar: usedCachedGrammar
        )
    }

    func resolvePromptRuntime(
        for prompt: String,
        using container: ModelContainer
    ) async throws -> MLXPromptRuntime {
        guard let promptParts = RoutingPrompt.splitForLocalModel(prompt) else {
            return MLXPromptRuntime(
                cachedContextPrompt: nil,
                runtimePrompt: RoutingPrompt.wrapForLocalModel(prompt),
                prefixProfile: nil
            )
        }

        let prefixProfile = try await container.perform { [self] context in
            try await self.ensurePromptPrefixCache(
                contextPrompt: promptParts.context,
                context: context
            )
        }

        return MLXPromptRuntime(
            cachedContextPrompt: promptParts.context,
            runtimePrompt: RoutingPrompt.buildLocalModelCurrentTurn(promptParts.currentMessage),
            prefixProfile: prefixProfile
        )
    }

    func runTimedInference(
        container: ModelContainer,
        runtimePrompt: String,
        processor: GrammarMaskedLogitProcessor,
        prefixBaseOffset: Int?,
        timeout: TimeInterval
    ) async throws -> MLXInferenceProfile {
        let inferenceStart = CFAbsoluteTimeGetCurrent()
        let timedGeneration = try await withThrowingTaskGroup(of: MLXTimedGeneration.self) { group in
            group.addTask { [self, runtimePrompt, prefixBaseOffset] in
                try await self.performGeneration(
                    in: container,
                    runtimePrompt: runtimePrompt,
                    processor: processor,
                    prefixBaseOffset: prefixBaseOffset
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
        let inferenceRoundTripMs = (CFAbsoluteTimeGetCurrent() - inferenceStart) * 1_000

        Self.logger.debug("Local routing output: \(timedGeneration.generation.output.prefix(200))")

        return MLXInferenceProfile(
            generation: timedGeneration.generation,
            inferenceRoundTripMs: inferenceRoundTripMs,
            containerPerformRoundTripMs: timedGeneration.containerPerformRoundTripMs
        )
    }

    func performGeneration(
        in container: ModelContainer,
        runtimePrompt: String,
        processor: GrammarMaskedLogitProcessor,
        prefixBaseOffset: Int?
    ) async throws -> MLXTimedGeneration {
        let performStart = CFAbsoluteTimeGetCurrent()
        let generation = try await container.perform { [self] context in
            try await Self.generateConstrained(
                messages: promptMessages(
                    for: runtimePrompt,
                    prefixBaseOffset: prefixBaseOffset
                ),
                context: context,
                processor: processor,
                maxTokens: Self.maxGenerationTokens,
                prefixCache: self.promptPrefixCache?.cache,
                prefixBaseOffset: prefixBaseOffset
            )
        }
        let roundTripMs = (CFAbsoluteTimeGetCurrent() - performStart) * 1_000
        return MLXTimedGeneration(
            generation: generation,
            containerPerformRoundTripMs: roundTripMs
        )
    }

    func ensureGrammarProcessor(context: ModelContext) async throws -> GrammarMaskedLogitProcessor {
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

    func ensurePromptPrefixCache(
        contextPrompt: String,
        context: ModelContext
    ) async throws -> MLXPrefixCacheProfile {
        if let cached = promptPrefixCache, cached.key == contextPrompt {
            return MLXPrefixCacheProfile(
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
        promptPrefixCache = MLXPromptPrefixCache(
            key: contextPrompt,
            baseOffset: baseOffset,
            cache: cache
        )
        return MLXPrefixCacheProfile(
            baseOffset: baseOffset,
            buildMs: (CFAbsoluteTimeGetCurrent() - buildStart) * 1_000,
            cacheHit: false
        )
    }

    func parseRoutingJSON(from jsonString: String) throws -> MLXJSONParseProfile {
        let parseStart = CFAbsoluteTimeGetCurrent()
        guard let data = jsonString.data(using: .utf8),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            throw MLXRoutingError.invalidOutput(jsonString)
        }
        return MLXJSONParseProfile(
            json: json,
            jsonParseMs: (CFAbsoluteTimeGetCurrent() - parseStart) * 1_000
        )
    }

    // swiftlint:disable:next function_body_length
    static func generateConstrained(
        messages: [Chat.Message],
        context: ModelContext,
        processor: GrammarMaskedLogitProcessor,
        maxTokens: Int,
        prefixCache: [KVCache]? = nil,
        prefixBaseOffset: Int? = nil
    ) async throws -> MLXGenerationProfile {
        let closureStart = CFAbsoluteTimeGetCurrent()
        let userInput = UserInput(chat: messages)

        let prepareStart = CFAbsoluteTimeGetCurrent()
        let input = try await context.processor.prepare(input: userInput)
        let promptPreparationMs = (CFAbsoluteTimeGetCurrent() - prepareStart) * 1_000

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
        let tokenIteratorInitMs = (CFAbsoluteTimeGetCurrent() - iteratorStart) * 1_000

        let generationStart = CFAbsoluteTimeGetCurrent()
        let result: GenerateResult = MLXLMCommon.generate(
            input: input,
            context: context,
            iterator: iterator,
            didGenerate: stopWhenGrammarTerminated(processor)
        )
        let generationMs = (CFAbsoluteTimeGetCurrent() - generationStart) * 1_000

        return MLXGenerationProfile(
            output: result.output,
            closureMs: (CFAbsoluteTimeGetCurrent() - closureStart) * 1_000,
            promptPreparationMs: promptPreparationMs,
            tokenIteratorInitMs: tokenIteratorInitMs,
            generationMs: generationMs
        )
    }

    static func prefillPromptCache(
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
}

extension MLXRoutingEngine {
    fileprivate func promptMessages(
        for runtimePrompt: String,
        prefixBaseOffset: Int?
    ) -> [Chat.Message] {
        if prefixBaseOffset == nil {
            return [
                .system(RoutingPrompt.systemInstruction),
                .user(runtimePrompt)
            ]
        }
        return [.user(runtimePrompt)]
    }

    fileprivate static func stopWhenGrammarTerminated(
        _ processor: GrammarMaskedLogitProcessor
    ) -> ([Int]) -> GenerateDisposition {
        { _ in
            if processor.grammarMatcher.isTerminated() {
                return .stop
            }
            return .more
        }
    }
}

internal func measureAsyncMilliseconds<T>(
    _ operation: () async throws -> T
) async rethrows -> Double {
    let start = CFAbsoluteTimeGetCurrent()
    _ = try await operation()
    return (CFAbsoluteTimeGetCurrent() - start) * 1_000
}

// swiftlint:enable type_contents_order

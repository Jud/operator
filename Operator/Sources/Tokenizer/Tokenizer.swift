//
//  Tokenizer.swift
//
//  Vendored from huggingface/swift-transformers (Apache 2.0).
//  Stripped to: BPE tokenizer loading from local tokenizer.json files.
//  No Hub API, no chat templates, no Jinja dependency.
//

import Foundation

// MARK: - Errors

public enum TokenizerError: LocalizedError {
    case missingConfig
    case missingTokenizerClassInConfig
    case unsupportedTokenizer(String)
    case missingVocab
    case malformedVocab
    case tooLong(String)

    public var errorDescription: String? {
        switch self {
        case .missingConfig: "Tokenizer configuration is missing."
        case .missingTokenizerClassInConfig: "The tokenizer class is not specified in the configuration."
        case let .unsupportedTokenizer(name): "The tokenizer type '\(name)' is not supported."
        case .missingVocab: "Vocabulary file is missing."
        case .malformedVocab: "The vocabulary file is malformed."
        case let .tooLong(message): "Input is too long: \(message)"
        }
    }
}

// MARK: - Protocols

public protocol TokenizingModel {
    func tokenize(text: String) -> [String]
    func callAsFunction(_ text: String) -> [String]
    func convertTokenToId(_ token: String) -> Int?
    func convertTokensToIds(_ tokens: [String]) -> [Int?]
    func convertIdToToken(_ id: Int) -> String?
    func convertIdsToTokens(_ ids: [Int]) -> [String?]

    var bosToken: String? { get }
    var bosTokenId: Int? { get }
    var eosToken: String? { get }
    var eosTokenId: Int? { get }
    var unknownToken: String? { get }
    var unknownTokenId: Int? { get }
    var fuseUnknownTokens: Bool { get }
}

public extension TokenizingModel {
    func callAsFunction(_ text: String) -> [String] { tokenize(text: text) }
    func convertTokensToIds(_ tokens: [String]) -> [Int?] { tokens.map { convertTokenToId($0) } }
    func convertIdsToTokens(_ ids: [Int]) -> [String?] { ids.map { convertIdToToken($0) } }
}

public protocol PreTrainedTokenizerModel: TokenizingModel {
    init(tokenizerConfig: Config, tokenizerData: Config, addedTokens: [String: Int]) throws
}

// MARK: - Helpers

func addedTokenAsString(_ addedToken: Config?) -> String? {
    guard let addedToken else { return nil }
    if let stringValue = addedToken.string() { return stringValue }
    return addedToken.content.string()
}

// MARK: - Token decoding protocol

public protocol TokenDecoding {
    func decode(tokens: [String]) -> [String]
    func callAsFunction(_ tokens: [String]) -> [String]
}

public extension TokenDecoding {
    func callAsFunction(_ tokens: [String]) -> [String] { decode(tokens: tokens) }
}

// MARK: - Tokenizer (wraps a model with pre/post processing)

public protocol Tokenizer {
    func tokenize(text: String) -> [String]
    func encode(text: String) -> [Int]
    func decode(tokens: [Int]) -> String

    var bosToken: String? { get }
    var bosTokenId: Int? { get }
    var eosToken: String? { get }
    var eosTokenId: Int? { get }
    var unknownToken: String? { get }
    var unknownTokenId: Int? { get }
}

// MARK: - PreTrainedTokenizer (concrete implementation)

public class PreTrainedTokenizer: Tokenizer {
    let model: TokenizingModel
    let preTokenizer: PreTokenizer?
    let normalizer: Normalizer?
    let postProcessor: PostProcessor?
    let decoder: TokenDecoder?
    let cleanUpTokenizationSpaces: Bool

    public var bosToken: String? { model.bosToken }
    public var bosTokenId: Int? { model.bosTokenId }
    public var eosToken: String? { model.eosToken }
    public var eosTokenId: Int? { model.eosTokenId }
    public var unknownToken: String? { model.unknownToken }
    public var unknownTokenId: Int? { model.unknownTokenId }

    init(
        model: TokenizingModel,
        preTokenizer: PreTokenizer? = nil,
        normalizer: Normalizer? = nil,
        postProcessor: PostProcessor? = nil,
        decoder: TokenDecoder? = nil,
        cleanUpTokenizationSpaces: Bool = false
    ) {
        self.model = model
        self.preTokenizer = preTokenizer
        self.normalizer = normalizer
        self.postProcessor = postProcessor
        self.decoder = decoder
        self.cleanUpTokenizationSpaces = cleanUpTokenizationSpaces
    }

    public func tokenize(text: String) -> [String] {
        var input = text
        if let normalizer { input = normalizer(text: input) }
        if let preTokenizer {
            return preTokenizer(text: input).flatMap { model.tokenize(text: $0) }
        }
        return model.tokenize(text: input)
    }

    public func encode(text: String) -> [Int] {
        var tokens = tokenize(text: text)
        if let postProcessor {
            tokens = postProcessor(tokens: tokens)
        }
        return tokens.compactMap { model.convertTokenToId($0) }
    }

    public func decode(tokens: [Int]) -> String {
        let tokenStrings = tokens.compactMap { model.convertIdToToken($0) }
        if let decoder {
            return decoder(tokens: tokenStrings).joined()
        }
        return tokenStrings.joined()
    }
}

// MARK: - Factory: load from local directory

enum TokenizerModel {
    static func unknownToken(from tokenizerConfig: Config) -> String? {
        tokenizerConfig.unkToken.content.string() ?? tokenizerConfig.unkToken.string()
    }
}

public enum AutoTokenizer {
    /// Load a tokenizer from a local directory containing tokenizer.json and tokenizer_config.json.
    public static func from(modelFolder: URL) throws -> Tokenizer {
        let tokenizerDataURL = modelFolder.appendingPathComponent("tokenizer.json")
        let tokenizerConfigURL = modelFolder.appendingPathComponent("tokenizer_config.json")

        let tokenizerData = try Config(fileURL: tokenizerDataURL)
        let tokenizerConfig = try Config(fileURL: tokenizerConfigURL)

        // Extract added tokens
        var addedTokens: [String: Int] = [:]
        if let tokens = tokenizerData.addedTokens.array() {
            for token in tokens {
                if let id = token.id.integer(), let content = token.content.string() {
                    addedTokens[content] = id
                }
            }
        }

        // Create BPE model
        let model = try BPETokenizer(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData, addedTokens: addedTokens)

        // Create processing pipeline
        let preTokenizer = PreTokenizerFactory.fromConfig(config: tokenizerData.preTokenizer)
        let normalizer = NormalizerFactory.fromConfig(config: tokenizerData.normalizer)
        let postProcessor = PostProcessorFactory.fromConfig(config: tokenizerData.postProcessor)
        let decoder = DecoderFactory.fromConfig(config: tokenizerData.decoder)

        let cleanUp = tokenizerConfig.cleanUpTokenizationSpaces.boolean(or: false)

        return PreTrainedTokenizer(
            model: model,
            preTokenizer: preTokenizer,
            normalizer: normalizer,
            postProcessor: postProcessor,
            decoder: decoder,
            cleanUpTokenizationSpaces: cleanUp
        )
    }
}

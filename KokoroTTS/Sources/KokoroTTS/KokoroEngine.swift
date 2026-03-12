import CoreML
import Foundation
import NaturalLanguage

/// High-quality text-to-speech engine using Kokoro-82M CoreML models.
///
/// Uses unified end-to-end models (StyleTTS2 + iSTFTNet vocoder) with one
/// CoreML model per bucket size. Runs on the Apple Neural Engine for 3.5x
/// faster inference vs GPU.
///
/// ```swift
/// let engine = try KokoroEngine(modelDirectory: ModelManager.defaultDirectory)
/// try engine.warmUp()
/// let result = try engine.synthesize(text: "Hello world", voice: "af_heart")
/// ```
public final class KokoroEngine: @unchecked Sendable {
    // All mutable state is set exclusively during init and never modified after,
    // making concurrent reads safe despite @unchecked Sendable.

    /// Output sample rate in Hz.
    public static let sampleRate = 24_000

    /// Number of random phases for iSTFTNet vocoder.
    static let numPhases = 9

    private let phonemizer: Phonemizer
    private let tokenizer: Tokenizer
    private let voiceStore: VoiceStore
    private let modelDirectory: URL
    private var unifiedModels: [UnifiedBucket: MLModel] = [:]

    /// Maximum tokens that can be processed in a single model call.
    private var maxTokensPerChunk: Int {
        UnifiedBucket.allCases.map(\.maxTokens).max() ?? 242
    }

    /// Creates a KokoroEngine from cached models.
    ///
    /// - Parameters:
    ///   - modelDirectory: Path containing `.mlmodelc` model directories and voice embeddings.
    ///   - allowLowPrecisionGPU: Enable FP16 accumulation on GPU for potentially faster inference.
    /// - Throws: ``KokoroError/modelsNotAvailable(_:)`` if no models found.
    public init(modelDirectory: URL, allowLowPrecisionGPU: Bool = true) throws {
        self.modelDirectory = modelDirectory

        guard ModelManager.modelsAvailable(at: modelDirectory) else {
            throw KokoroError.modelsNotAvailable(modelDirectory)
        }

        // Load phonemizer: prefer model directory (runtime download), fall back to bundle
        let phonemizer = Phonemizer()
        let hasRuntimeDicts = FileManager.default.fileExists(
            atPath: modelDirectory.appendingPathComponent("us_gold.json").path)
        if hasRuntimeDicts {
            try phonemizer.loadDictionaries(from: modelDirectory)
        } else {
            phonemizer.loadDictionariesFromBundle()
        }
        if let g2p = try? G2PModel.load(from: modelDirectory) {
            phonemizer.attachG2P(g2p)
        }
        self.phonemizer = phonemizer

        // Load tokenizer: try model directory first, then bundle
        let vocabURL = modelDirectory.appendingPathComponent("vocab_index.json")
        if FileManager.default.fileExists(atPath: vocabURL.path) {
            self.tokenizer = try Tokenizer.load(from: vocabURL)
        } else {
            self.tokenizer = try Tokenizer.loadFromBundle()
        }

        self.voiceStore = try VoiceStore(directory: modelDirectory.appendingPathComponent("voices"))

        // Load unified models
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine  // ANE: 3.5x faster than GPU (425ms vs 1490ms)
        config.allowLowPrecisionAccumulationOnGPU = allowLowPrecisionGPU
        for bucket in UnifiedBucket.allCases {
            let url = modelDirectory.appendingPathComponent(bucket.modelName + ".mlmodelc")
            if FileManager.default.fileExists(atPath: url.path) {
                self.unifiedModels[bucket] = try MLModel(contentsOf: url, configuration: config)
            }
        }

        guard !unifiedModels.isEmpty else {
            throw KokoroError.modelsNotAvailable(modelDirectory)
        }
    }

    /// Synthesize text to PCM audio samples.
    ///
    /// Automatically splits long text into sentences when it exceeds the model's
    /// token limit. Each sentence is synthesized separately and the results are
    /// concatenated.
    ///
    /// - Parameters:
    ///   - text: Text to speak (any length).
    ///   - voice: Voice preset name (e.g. "af_heart", "am_adam").
    ///   - speed: Playback speed multiplier (0.5–2.0). Native duration scaling, not post-processing.
    /// - Returns: Synthesis result with PCM samples and metadata.
    public func synthesize(text: String, voice: String, speed: Float = 1.0, verbose: Bool = false) throws -> SynthesisResult {
        let t0 = CFAbsoluteTimeGetCurrent()
        let styleVector = try voiceStore.embedding(for: voice)

        let phonemes = phonemizer.textToPhonemes(text)
        let tokenIds = tokenizer.encode(phonemes)

        // If it fits in a single model call, synthesize directly
        if tokenIds.count <= maxTokensPerChunk {
            let (samples, durations) = try synthesizeUnified(
                tokenIds: tokenIds, styleVector: styleVector, speed: speed)
            let elapsed = CFAbsoluteTimeGetCurrent() - t0
            if verbose {
                print("  inference: \(String(format: "%.2f", elapsed * 1_000))ms  tokens: \(tokenIds.count)")
            }
            return SynthesisResult(
                samples: samples, phonemeDurations: durations,
                tokenCount: tokenIds.count, synthesisTime: elapsed)
        }

        // Auto-chunk by sentence for long text
        let sentences = splitSentences(text)
        var allSamples: [Float] = []
        var allDurations: [Int] = []
        var totalTokens = 0

        for sentence in sentences {
            let sp = phonemizer.textToPhonemes(sentence)
            let st = tokenizer.encode(sp)
            totalTokens += st.count
            let (samples, durations) = try synthesizeUnified(
                tokenIds: st, styleVector: styleVector, speed: speed)
            allSamples.append(contentsOf: samples)
            allDurations.append(contentsOf: durations)
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        if verbose {
            print("  total: \(String(format: "%.2f", elapsed * 1_000))ms (\(sentences.count) chunks, \(totalTokens) tokens)")
        }

        return SynthesisResult(
            samples: allSamples, phonemeDurations: allDurations,
            tokenCount: totalTokens, synthesisTime: elapsed)
    }

    /// Split text into sentences using NLTokenizer.
    private func splitSentences(_ text: String) -> [String] {
        var sentences: [String] = []
        let nlTokenizer = NLTokenizer(unit: .sentence)
        nlTokenizer.string = text
        nlTokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let s = String(text[range]).trimmingCharacters(in: .whitespaces)
            if !s.isEmpty { sentences.append(s) }
            return true
        }
        return sentences.isEmpty ? [text] : sentences
    }

    /// Convert text to IPA phonemes and token IDs (for debugging).
    public func phonemize(_ text: String) -> (phonemes: String, tokenIds: [Int]) {
        let phonemes = phonemizer.textToPhonemes(text)
        let tokenIds = tokenizer.encode(phonemes)
        return (phonemes, tokenIds)
    }

    /// Pre-warm CoreML compilation cache. Call once at daemon startup.
    public func warmUp() throws {
        _ = try? synthesize(text: "hello", voice: availableVoices.first ?? "af_heart")
    }

    /// Available voice preset names.
    public var availableVoices: [String] {
        voiceStore.availableVoices
    }

    // MARK: - Unified Pipeline

    private func synthesizeUnified(
        tokenIds: [Int],
        styleVector: [Float],
        speed: Float
    ) throws -> (samples: [Float], durations: [Int]) {
        guard let bucket = UnifiedBucket.select(forTokenCount: tokenIds.count) else {
            let maxAvailable = UnifiedBucket.allCases.map(\.maxTokens).max() ?? 0
            throw KokoroError.textTooLong(tokenCount: tokenIds.count, maxTokens: maxAvailable)
        }

        // Find a loaded model: preferred bucket first, then any loaded model that fits
        let resolved: (bucket: UnifiedBucket, model: MLModel)
        if let m = unifiedModels[bucket] {
            resolved = (bucket, m)
        } else if let fallback = unifiedModels.sorted(by: { $0.key.maxTokens < $1.key.maxTokens })
            .first(where: { $0.key.maxTokens >= tokenIds.count })
        {
            resolved = (fallback.key, fallback.value)
        } else {
            throw KokoroError.inferenceFailed("No suitable model loaded")
        }

        let model = resolved.model
        let maxTokens = resolved.bucket.maxTokens
        let (inputIds, maskArray) = try MLArrayHelpers.makeTokenInputs(
            tokenIds: tokenizer.pad(tokenIds, to: maxTokens), maxLength: maxTokens)
        // Fix mask: pad() fills with pad tokens, but mask should reflect real token count
        let maskPtr = maskArray.dataPointer.assumingMemoryBound(to: Int32.self)
        for i in tokenIds.count..<maxTokens { maskPtr[i] = 0 }

        let refS = try MLArrayHelpers.makeStyleArray(styleVector)

        let randomPhases = try MLMultiArray(shape: [1, Self.numPhases as NSNumber], dataType: .float32)
        let phasePtr = randomPhases.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<Self.numPhases {
            phasePtr[i] = Float.random(in: 0..<(2 * .pi))
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: inputIds),
            "attention_mask": MLFeatureValue(multiArray: maskArray),
            "ref_s": MLFeatureValue(multiArray: refS),
            "random_phases": MLFeatureValue(multiArray: randomPhases),
        ])

        let output = try model.prediction(from: input)

        guard let audio = output.featureValue(for: "audio")?.multiArrayValue else {
            throw KokoroError.inferenceFailed("Missing audio output")
        }

        let validSamples: Int
        if let lengthArray = output.featureValue(for: "audio_length_samples")?.multiArrayValue {
            validSamples = lengthArray[0].intValue
        } else {
            validSamples = audio.count
        }

        let samples = MLArrayHelpers.extractFloats(from: audio, maxCount: validSamples)

        let durations: [Int]
        if let predDur = output.featureValue(for: "pred_dur")?.multiArrayValue {
            durations = (0..<min(predDur.count, tokenIds.count)).map { predDur[$0].intValue }
        } else {
            durations = []
        }

        return (samples, durations)
    }
}

/// Result from a synthesis call.
public struct SynthesisResult: Sendable {
    /// 24kHz mono PCM float samples.
    public let samples: [Float]
    /// Per-phoneme frame counts for interruption tracking.
    public let phonemeDurations: [Int]
    /// Number of input tokens.
    public let tokenCount: Int
    /// Wall-clock synthesis time.
    public let synthesisTime: TimeInterval
    /// Audio duration in seconds.
    public var duration: TimeInterval { Double(samples.count) / Double(KokoroEngine.sampleRate) }
}

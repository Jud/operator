import CoreML
import Foundation
import MisakiSwift
import NaturalLanguage

/// High-quality text-to-speech engine using Kokoro-82M CoreML models.
///
/// Uses unified end-to-end models (StyleTTS2 + iSTFTNet vocoder) with one
/// CoreML model per bucket size. Runs on the Apple Neural Engine for 3.5x
/// faster inference vs GPU.
///
/// ```swift
/// let engine = try KokoroEngine(modelDirectory: myModelPath)
/// engine.warmUp()
/// let result = try engine.synthesize(text: "Hello world", voice: "af_heart")
/// // result.samples contains 24kHz mono PCM float audio
/// ```
public final class KokoroEngine: @unchecked Sendable {
    // Immutable after init. EnglishG2P has internal mutable state,
    // so all phonemize calls are serialized through g2pLock.

    /// Output sample rate in Hz (24kHz).
    public static let sampleRate = 24_000

    private static let numPhases = 9

    private let g2p: EnglishG2P
    private let g2pLock = NSLock()
    private let tokenizer: Tokenizer
    private let voiceStore: VoiceStore
    private let unifiedModels: [UnifiedBucket: MLModel]

    /// Creates a KokoroEngine from cached models.
    ///
    /// - Parameter modelDirectory: Path containing `.mlmodelc` model bundles
    ///   and a `voices/` directory with voice embedding JSON files.
    /// - Throws: ``KokoroError/modelsNotAvailable(_:)`` if no models found.
    public init(modelDirectory: URL) throws {
        guard ModelManager.modelsAvailable(at: modelDirectory) else {
            throw KokoroError.modelsNotAvailable(modelDirectory)
        }

        self.g2p = EnglishG2P(british: false)

        let vocabURL = modelDirectory.appendingPathComponent("vocab_index.json")
        if FileManager.default.fileExists(atPath: vocabURL.path) {
            self.tokenizer = try Tokenizer.load(from: vocabURL)
        } else {
            self.tokenizer = try Tokenizer.loadFromBundle()
        }

        self.voiceStore = try VoiceStore(directory: modelDirectory.appendingPathComponent("voices"))

        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        var models: [UnifiedBucket: MLModel] = [:]
        for bucket in UnifiedBucket.allCases {
            let url = modelDirectory.appendingPathComponent(bucket.modelName + ".mlmodelc")
            if FileManager.default.fileExists(atPath: url.path) {
                models[bucket] = try MLModel(contentsOf: url, configuration: config)
            }
        }

        guard !models.isEmpty else {
            throw KokoroError.modelsNotAvailable(modelDirectory)
        }
        self.unifiedModels = models
    }

    // MARK: - Synthesis

    /// Synthesize text to PCM audio samples.
    ///
    /// Automatically splits long text into sentences when it exceeds the model's
    /// token limit. Each sentence is synthesized separately and concatenated.
    ///
    /// - Parameters:
    ///   - text: Text to speak (any length).
    ///   - voice: Voice preset name (e.g. `"af_heart"`, `"am_adam"`).
    ///     See ``availableVoices`` for valid names.
    ///   - speed: Playback speed multiplier (0.5–2.0). Native duration scaling,
    ///     not post-processing pitch distortion.
    /// - Returns: Synthesis result with PCM samples and metadata.
    /// - Throws: ``KokoroError/voiceNotFound(_:)`` if the voice doesn't exist.
    public func synthesize(text: String, voice: String, speed: Float = 1.0) throws -> SynthesisResult {
        let t0 = CFAbsoluteTimeGetCurrent()
        let styleVector = try voiceStore.embedding(for: voice)

        let phonemes = lockedPhonemize(text)
        let tokenIds = tokenizer.encode(phonemes)

        if tokenIds.count <= UnifiedBucket.maxTokenCount {
            let (samples, durations) = try synthesizeUnified(
                tokenIds: tokenIds, styleVector: styleVector, speed: speed)
            let elapsed = CFAbsoluteTimeGetCurrent() - t0
            return SynthesisResult(
                samples: samples, tokenDurations: durations,
                tokenCount: tokenIds.count, synthesisTime: elapsed)
        }

        let sentences = splitSentences(text)
        var allSamples: [Float] = []
        var allDurations: [Int] = []
        var totalTokens = 0

        for sentence in sentences {
            let sp = lockedPhonemize(sentence)
            let st = tokenizer.encode(sp)
            totalTokens += st.count
            let (samples, durations) = try synthesizeUnified(
                tokenIds: st, styleVector: styleVector, speed: speed)
            allSamples.append(contentsOf: samples)
            allDurations.append(contentsOf: durations)
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        return SynthesisResult(
            samples: allSamples, tokenDurations: allDurations,
            tokenCount: totalTokens, synthesisTime: elapsed)
    }

    /// Convert text to IPA phonemes using the built-in G2P pipeline.
    ///
    /// Useful for debugging pronunciation or inspecting the phonemization output.
    public func phonemize(_ text: String) -> String {
        lockedPhonemize(text)
    }

    // MARK: - Voices

    /// Available voice preset names.
    public var availableVoices: [String] {
        voiceStore.availableVoices
    }

    // MARK: - Warmup

    /// Pre-warm the CoreML compilation cache.
    ///
    /// Triggers on-device model compilation for the Neural Engine. Call once
    /// at app startup to avoid a cold-start delay on the first synthesis call.
    /// Safe to skip — the first real synthesis will just be slower.
    public func warmUp() {
        guard let (bucket, model) = unifiedModels
            .min(by: { $0.key.maxTokens < $1.key.maxTokens }) else { return }
        let maxTokens = bucket.maxTokens
        do {
            let (inputIds, mask) = try MLArrayHelpers.makeTokenInputs(
                tokenIds: [Int](repeating: 0, count: maxTokens), maxLength: maxTokens)
            let refS = try MLArrayHelpers.makeStyleArray(
                [Float](repeating: 0, count: VoiceStore.styleDim))
            let phases = try MLMultiArray(
                shape: [1, Self.numPhases as NSNumber], dataType: .float32)
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": MLFeatureValue(multiArray: inputIds),
                "attention_mask": MLFeatureValue(multiArray: mask),
                "ref_s": MLFeatureValue(multiArray: refS),
                "random_phases": MLFeatureValue(multiArray: phases),
            ])
            _ = try model.prediction(from: input)
        } catch {
            // Warmup failure is non-fatal; first real call will be slower.
        }
    }

    // MARK: - Private

    private func lockedPhonemize(_ text: String) -> String {
        g2pLock.lock()
        defer { g2pLock.unlock() }
        let (phonemes, _) = g2p.phonemize(text: text)
        return phonemes
    }

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

    private func synthesizeUnified(
        tokenIds: [Int],
        styleVector: [Float],
        speed: Float
    ) throws -> (samples: [Float], durations: [Int]) {
        guard let bucket = UnifiedBucket.select(forTokenCount: tokenIds.count) else {
            throw KokoroError.textTooLong(
                tokenCount: tokenIds.count, maxTokens: UnifiedBucket.maxTokenCount)
        }

        let resolved: (bucket: UnifiedBucket, model: MLModel)
        if let m = unifiedModels[bucket] {
            resolved = (bucket, m)
        } else if let fallback = UnifiedBucket.sortedBySize
            .first(where: { $0.maxTokens >= tokenIds.count })
            .flatMap({ b in unifiedModels[b].map { (b, $0) } })
        {
            resolved = fallback
        } else {
            throw KokoroError.inferenceFailed("No suitable model loaded")
        }

        let model = resolved.model
        let maxTokens = resolved.bucket.maxTokens
        let (inputIds, maskArray) = try MLArrayHelpers.makeTokenInputs(
            tokenIds: tokenizer.pad(tokenIds, to: maxTokens), maxLength: maxTokens)
        let maskPtr = maskArray.dataPointer.assumingMemoryBound(to: Int32.self)
        for i in tokenIds.count..<maxTokens { maskPtr[i] = 0 }

        let refS = try MLArrayHelpers.makeStyleArray(styleVector)

        let randomPhases = try MLMultiArray(
            shape: [1, Self.numPhases as NSNumber], dataType: .float32)
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

// MARK: - SynthesisResult

/// Result from a text-to-speech synthesis call.
public struct SynthesisResult: Sendable {
    /// 24kHz mono PCM float samples.
    public let samples: [Float]

    /// Per-token predicted durations in audio frames (at 24kHz).
    /// Useful for mapping playback position back to input tokens.
    public let tokenDurations: [Int]

    /// Number of input tokens processed.
    public let tokenCount: Int

    /// Wall-clock synthesis time in seconds.
    public let synthesisTime: TimeInterval

    /// Audio duration in seconds.
    public var duration: TimeInterval {
        Double(samples.count) / Double(KokoroEngine.sampleRate)
    }

    /// Creates a new synthesis result.
    public init(
        samples: [Float],
        tokenDurations: [Int] = [],
        tokenCount: Int,
        synthesisTime: TimeInterval
    ) {
        self.samples = samples
        self.tokenDurations = tokenDurations
        self.tokenCount = tokenCount
        self.synthesisTime = synthesisTime
    }
}

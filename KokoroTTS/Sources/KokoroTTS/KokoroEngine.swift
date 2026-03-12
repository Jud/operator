import CoreML
import Foundation

/// High-quality text-to-speech engine using Kokoro-82M CoreML models.
///
/// Two pipeline modes:
/// - **Unified** (primary): Single end-to-end model per bucket
/// - **Two-stage** (future): Duration Model (CPU/GPU) → Alignment (Accelerate) → HAR Decoder (ANE)
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

    /// Number of random phases for iSTFTNet vocoder (unified models only).
    static let numPhases = 9

    private let phonemizer: Phonemizer
    private let tokenizer: Tokenizer
    private let voiceStore: VoiceStore
    private let modelDirectory: URL

    // Two-stage pipeline
    private var durationModel: DurationModel?
    private var harDecoders: [Bucket: HARDecoder] = [:]

    // Unified pipeline (primary)
    private var unifiedModels: [UnifiedBucket: MLModel] = [:]

    /// Whether the engine uses the two-stage pipeline (true) or unified (false).
    private(set) var isTwoStage: Bool = false

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

        // Try unified models first (primary path)
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU  // ANE compilation takes 18s+ and doesn't cache
        config.allowLowPrecisionAccumulationOnGPU = allowLowPrecisionGPU
        for bucket in UnifiedBucket.allCases {
            let url = modelDirectory.appendingPathComponent(bucket.modelName + ".mlmodelc")
            if FileManager.default.fileExists(atPath: url.path) {
                self.unifiedModels[bucket] = try MLModel(contentsOf: url, configuration: config)
            }
        }

        // Fall back to two-stage if no unified models
        if unifiedModels.isEmpty {
            let durationURL = modelDirectory.appendingPathComponent("kokoro_duration.mlmodelc")
            if FileManager.default.fileExists(atPath: durationURL.path) {
                self.durationModel = try DurationModel(modelURL: durationURL)
                for bucket in Bucket.allCases {
                    let decoderURL = modelDirectory.appendingPathComponent(
                        bucket.decoderModelName + ".mlmodelc")
                    if FileManager.default.fileExists(atPath: decoderURL.path) {
                        self.harDecoders[bucket] = try HARDecoder(
                            modelURL: decoderURL, bucket: bucket)
                    }
                }
                if !harDecoders.isEmpty {
                    self.isTwoStage = true
                }
            }

            if !isTwoStage {
                throw KokoroError.modelsNotAvailable(modelDirectory)
            }
        }
    }

    /// Synthesize text to PCM audio samples.
    ///
    /// - Parameters:
    ///   - text: Text to speak (1-3 sentences recommended).
    ///   - voice: Voice preset name (e.g. "af_heart", "am_adam").
    ///   - speed: Playback speed multiplier (0.5–2.0). Native duration scaling, not post-processing.
    /// - Returns: Synthesis result with PCM samples and metadata.
    public func synthesize(text: String, voice: String, speed: Float = 1.0, verbose: Bool = false) throws -> SynthesisResult {
        let t0 = CFAbsoluteTimeGetCurrent()

        let styleVector = try voiceStore.embedding(for: voice)
        let tVoice = CFAbsoluteTimeGetCurrent()

        let phonemes = phonemizer.textToPhonemes(text)
        let tPhonemes = CFAbsoluteTimeGetCurrent()

        let tokenIds = tokenizer.encode(phonemes)
        let tTokens = CFAbsoluteTimeGetCurrent()

        let samples: [Float]
        let phonemeDurations: [Int]

        if isTwoStage {
            (samples, phonemeDurations) = try synthesizeTwoStage(
                tokenIds: tokenIds, styleVector: styleVector, speed: speed
            )
        } else {
            (samples, phonemeDurations) = try synthesizeUnified(
                tokenIds: tokenIds, styleVector: styleVector, speed: speed
            )
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - t0

        if verbose {
            let ms = { (t: Double) in String(format: "%.2f", t * 1_000) }
            print("  voice lookup:  \(ms(tVoice - t0))ms")
            print("  phonemize:     \(ms(tPhonemes - tVoice))ms")
            print("  tokenize:      \(ms(tTokens - tPhonemes))ms")
            print("  inference:     \(ms(elapsed - (tTokens - t0)))ms")
            print("  total:         \(ms(elapsed))ms")
        }

        return SynthesisResult(
            samples: samples,
            phonemeDurations: phonemeDurations,
            tokenCount: tokenIds.count,
            synthesisTime: elapsed
        )
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

    // MARK: - Two-Stage Pipeline

    private func synthesizeTwoStage(
        tokenIds: [Int],
        styleVector: [Float],
        speed: Float
    ) throws -> (samples: [Float], durations: [Int]) {
        guard let durationModel else {
            throw KokoroError.inferenceFailed("Duration model not loaded")
        }

        let bucket = Bucket.select(forTokenCount: tokenIds.count)
        guard let decoder = harDecoders[bucket] else {
            throw KokoroError.inferenceFailed("No HAR decoder for bucket \(bucket)")
        }

        let durOutput = try durationModel.predict(
            tokenIds: tokenIds, styleVector: styleVector, speed: speed)

        let alignment = try Alignment.compute(
            predDur: durOutput.predDur, tEn: durOutput.tEn,
            tokenCount: tokenIds.count, targetFrames: bucket.maxFrames)

        let waveform = try decoder.predict(
            asr: alignment.asr, f0Pred: alignment.f0Pred,
            nPred: alignment.nPred, styleVector: styleVector)

        let durations = durOutput.predDur.map { Int(roundf($0 * (1.0 / speed))) }
        return (waveform, durations)
    }

    // MARK: - Unified Pipeline

    private func synthesizeUnified(
        tokenIds: [Int],
        styleVector: [Float],
        speed: Float
    ) throws -> (samples: [Float], durations: [Int]) {
        guard let bucket = UnifiedBucket.select(forTokenCount: tokenIds.count) else {
            throw KokoroError.textTooLong(tokenCount: tokenIds.count, maxTokens: 249)
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

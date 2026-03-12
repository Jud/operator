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

    /// Number of random phase channels for the iSTFTNet vocoder.
    private static let numPhases = 9

    // CoreML model input/output feature names.
    private enum Feature {
        static let inputIds = "input_ids"
        static let attentionMask = "attention_mask"
        static let refS = "ref_s"
        static let randomPhases = "random_phases"
        static let audio = "audio"
        static let audioLength = "audio_length_samples"
        static let predDur = "pred_dur_clamped"
        static let speed = "speed"
    }

    private let g2p: EnglishG2P
    private let g2pLock = NSLock()
    private let tokenizer: Tokenizer
    private let voiceStore: VoiceStore
    private let model: MLModel

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
        let bucket = UnifiedBucket.v24_10s
        let url = modelDirectory.appendingPathComponent(bucket.modelName + ".mlmodelc")
        self.model = try MLModel(contentsOf: url, configuration: config)
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
    ///   - speed: Speech rate multiplier (0.5 = half speed, 2.0 = double speed).
    ///     Clamped to 0.5...2.0. Default is 1.0.
    /// - Returns: Synthesis result with PCM samples and metadata.
    /// - Throws: ``KokoroError/voiceNotFound(_:)`` if the voice doesn't exist.
    public func synthesize(text: String, voice: String, speed: Float = 1.0) throws -> SynthesisResult {
        let t0 = CFAbsoluteTimeGetCurrent()
        let styleVector = try voiceStore.embedding(for: voice)
        let clampedSpeed = min(max(speed, 0.5), 2.0)

        var allSamples: [Float] = []
        var allDurations: [Int] = []
        var totalTokens = 0

        for sentence in splitSentences(text) {
            let phonemes = lockedPhonemize(sentence)
            let tokenIds = tokenizer.encode(phonemes)
            totalTokens += tokenIds.count
            let (samples, durations) = try synthesizeUnified(
                tokenIds: tokenIds, styleVector: styleVector, speed: clampedSpeed)
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
        let maxTokens = UnifiedBucket.maxTokenCount
        do {
            let (inputIds, mask) = try MLArrayHelpers.makeTokenInputs(
                tokenIds: [Int](repeating: 0, count: maxTokens), maxLength: maxTokens)
            let refS = try MLArrayHelpers.makeStyleArray(
                [Float](repeating: 0, count: VoiceStore.styleDim))
            let phases = try MLMultiArray(
                shape: [1, Self.numPhases as NSNumber], dataType: .float32)
            let speed = try MLMultiArray(shape: [1], dataType: .float32)
            speed[0] = 1.0 as NSNumber
            let input = try MLDictionaryFeatureProvider(dictionary: [
                Feature.inputIds: MLFeatureValue(multiArray: inputIds),
                Feature.attentionMask: MLFeatureValue(multiArray: mask),
                Feature.refS: MLFeatureValue(multiArray: refS),
                Feature.randomPhases: MLFeatureValue(multiArray: phases),
                Feature.speed: MLFeatureValue(multiArray: speed),
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
        guard tokenIds.count <= UnifiedBucket.maxTokenCount else {
            throw KokoroError.textTooLong(
                tokenCount: tokenIds.count, maxTokens: UnifiedBucket.maxTokenCount)
        }

        let maxTokens = UnifiedBucket.maxTokenCount
        let (inputIds, maskArray) = try MLArrayHelpers.makeTokenInputs(
            tokenIds: tokenIds, maxLength: maxTokens)

        let refS = try MLArrayHelpers.makeStyleArray(styleVector)

        let randomPhases = try MLMultiArray(
            shape: [1, Self.numPhases as NSNumber], dataType: .float32)
        let phasePtr = randomPhases.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<Self.numPhases {
            phasePtr[i] = Float.random(in: 0..<(2 * .pi))
        }

        let speedArray = try MLMultiArray(shape: [1], dataType: .float32)
        speedArray[0] = speed as NSNumber

        let input = try MLDictionaryFeatureProvider(dictionary: [
            Feature.inputIds: MLFeatureValue(multiArray: inputIds),
            Feature.attentionMask: MLFeatureValue(multiArray: maskArray),
            Feature.refS: MLFeatureValue(multiArray: refS),
            Feature.randomPhases: MLFeatureValue(multiArray: randomPhases),
            Feature.speed: MLFeatureValue(multiArray: speedArray),
        ])

        let output = try model.prediction(from: input)

        guard let audio = output.featureValue(for: Feature.audio)?.multiArrayValue else {
            throw KokoroError.inferenceFailed("Missing audio output")
        }

        let validSamples: Int
        if let lengthArray = output.featureValue(for: Feature.audioLength)?.multiArrayValue {
            validSamples = lengthArray[0].intValue
        } else {
            validSamples = audio.count
        }

        let samples = MLArrayHelpers.extractFloats(from: audio, maxCount: validSamples)

        let durations: [Int]
        if let predDur = output.featureValue(for: Feature.predDur)?.multiArrayValue {
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

    /// Real-time factor (audio duration / synthesis time).
    /// Values above 1.0 mean faster than real-time.
    public var realTimeFactor: Double {
        synthesisTime > 0 ? duration / synthesisTime : 0
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

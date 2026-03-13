import Accelerate
import CoreML
import Foundation
import NaturalLanguage
import os

/// Model size bucket for automatic selection based on token count.
public enum ModelBucket: CaseIterable, Sendable, Comparable {
    /// Short utterances — 124 tokens max (~5s audio).
    case small
    /// Medium utterances — 242 tokens max (~10s audio).
    case medium

    /// CoreML model bundle name (without .mlmodelc extension).
    public var modelName: String {
        switch self {
        case .small: "kokoro_21_5s"
        case .medium: "kokoro_24_10s"
        }
    }

    /// Maximum input token count for this bucket.
    public var maxTokens: Int {
        switch self {
        case .small: 124
        case .medium: 242
        }
    }

    /// Select the smallest bucket that fits the token count.
    /// Returns nil if no bucket is large enough.
    ///
    /// Assumes `available` is sorted ascending (which `activeBuckets` always is).
    public static func select(forTokenCount count: Int, available: [ModelBucket]) -> ModelBucket? {
        assert(available == available.sorted(), "available must be sorted ascending")
        return available.first { $0.maxTokens >= count }
    }
}

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

    /// Audio samples per duration frame (sampleRate / vocoder frame rate).
    public static let hopSize = sampleRate / 24

    /// Valid speed range for synthesis.
    public static let speedRange: ClosedRange<Float> = 0.5...2.0

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
        static let predDurClamped = "pred_dur_clamped"
        static let predDur = "pred_dur"
        static let speed = "speed"
    }

    /// Per-bucket CoreML model and pre-allocated buffers.
    private struct BucketResources {
        let model: MLModel
        let inputIds: MLMultiArray
        let mask: MLMultiArray
        let refS: MLMultiArray
        let randomPhases: MLMultiArray
        let speed: MLMultiArray
    }

    private static let logger = Logger(
        subsystem: "com.kokorotts", category: "KokoroEngine")

    /// Threshold in seconds above which model.prediction() is considered slow
    /// (likely ANE fallback to CPU/GPU).
    private static let slowPredictionThreshold: TimeInterval = 0.150

    private let g2p: EnglishG2P
    private let g2pLock = NSLock()
    private let synthesizeLock = NSLock()
    private let tokenizer: Tokenizer
    private let voiceStore: VoiceStore
    private let bucketResources: [ModelBucket: BucketResources]

    /// Which buckets are loaded and available for synthesis (sorted ascending).
    public let activeBuckets: [ModelBucket]

    /// Creates a KokoroEngine from cached models.
    ///
    /// Loads all available model buckets from the directory. At least one
    /// `.mlmodelc` model bundle and a `voices/` directory must be present.
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

        var resources: [ModelBucket: BucketResources] = [:]
        for bucket in ModelBucket.allCases {
            let url = modelDirectory.appendingPathComponent(bucket.modelName + ".mlmodelc")
            guard FileManager.default.fileExists(atPath: url.path) else { continue }
            do {
                let model = try MLModel(contentsOf: url, configuration: config)
                let maxTokens = bucket.maxTokens
                let res = BucketResources(
                    model: model,
                    inputIds: try MLMultiArray(
                        shape: [1, maxTokens as NSNumber], dataType: .int32),
                    mask: try MLMultiArray(
                        shape: [1, maxTokens as NSNumber], dataType: .int32),
                    refS: try MLMultiArray(
                        shape: [1, VoiceStore.styleDim as NSNumber], dataType: .float32),
                    randomPhases: try MLMultiArray(
                        shape: [1, Self.numPhases as NSNumber], dataType: .float32),
                    speed: try MLMultiArray(shape: [1], dataType: .float32)
                )
                resources[bucket] = res
                Self.logger.info("Loaded \(bucket.modelName) (\(maxTokens) tokens)")
            } catch {
                Self.logger.warning("Failed to load \(bucket.modelName): \(error)")
            }
        }

        guard !resources.isEmpty else {
            throw KokoroError.modelsNotAvailable(modelDirectory)
        }

        self.bucketResources = resources
        self.activeBuckets = ModelBucket.allCases.filter { resources[$0] != nil }
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
    public func synthesize(
        text: String, voice: String, speed: Float = 1.0, bucket: ModelBucket? = nil
    ) throws -> SynthesisResult {
        let t0 = CFAbsoluteTimeGetCurrent()
        let styleVector = try voiceStore.embedding(for: voice)
        let clampedSpeed = min(max(speed, Self.speedRange.lowerBound), Self.speedRange.upperBound)

        var allSamples: [Float] = []
        var allDurations: [Int] = []
        var totalTokens = 0
        var selectedBucket: ModelBucket?

        for sentence in splitSentences(text) {
            let phonemes = lockedPhonemize(sentence)
            let tokenIds = tokenizer.encode(phonemes)
            totalTokens += tokenIds.count

            let useBucket: ModelBucket
            if let forced = bucket {
                useBucket = forced
            } else if let auto = ModelBucket.select(
                forTokenCount: tokenIds.count, available: activeBuckets)
            {
                useBucket = auto
            } else {
                // Token count exceeds all buckets — use largest available.
                // activeBuckets is guaranteed non-empty by init.
                useBucket = activeBuckets.last!
            }
            selectedBucket = useBucket

            var (samples, durations) = try synthesizeUnified(
                tokenIds: tokenIds, styleVector: styleVector, speed: clampedSpeed,
                bucket: useBucket)
            trimTrailingArtifacts(&samples)
            if allSamples.isEmpty {
                allSamples = samples
            } else {
                allSamples.append(contentsOf: samples)
            }
            allDurations.append(contentsOf: durations)
        }

        postProcess(&allSamples)

        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        return SynthesisResult(
            samples: allSamples, tokenDurations: allDurations,
            tokenCount: totalTokens, synthesisTime: elapsed,
            bucket: selectedBucket)
    }

    // MARK: - Voices

    /// Available voice preset names.
    public var availableVoices: [String] {
        voiceStore.availableVoices
    }

    // MARK: - Warmup

    /// Pre-warm the CoreML compilation cache for all loaded buckets.
    ///
    /// Triggers on-device model compilation for the Neural Engine. Call once
    /// at app startup to avoid a cold-start delay on the first synthesis call.
    /// Safe to skip — the first real synthesis will just be slower.
    public func warmUp() {
        for bucket in activeBuckets {
            do {
                let dummyTokens = [Int](repeating: 0, count: bucket.maxTokens)
                let dummyStyle = [Float](repeating: 0, count: VoiceStore.styleDim)
                _ = try synthesizeUnified(
                    tokenIds: dummyTokens, styleVector: dummyStyle, speed: 1.0,
                    bucket: bucket)
            } catch {
                Self.logger.warning(
                    "Warmup failed for \(bucket.modelName) (non-fatal): \(error.localizedDescription)"
                )
            }
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
        speed: Float,
        bucket: ModelBucket
    ) throws -> (samples: [Float], durations: [Int]) {
        guard let res = bucketResources[bucket] else {
            throw KokoroError.inferenceFailed("Bucket \(bucket.modelName) not loaded")
        }
        guard tokenIds.count <= bucket.maxTokens else {
            throw KokoroError.textTooLong(
                tokenCount: tokenIds.count, maxTokens: bucket.maxTokens)
        }

        synthesizeLock.lock()
        defer { synthesizeLock.unlock() }

        MLArrayHelpers.fillTokenInputs(
            from: tokenIds, into: res.inputIds, mask: res.mask, maxLength: bucket.maxTokens)
        MLArrayHelpers.fillStyleArray(from: styleVector, into: res.refS)

        let phasePtr = res.randomPhases.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<Self.numPhases {
            phasePtr[i] = Float.random(in: 0..<(2 * .pi))
        }

        res.speed[0] = speed as NSNumber

        let input = try MLDictionaryFeatureProvider(dictionary: [
            Feature.inputIds: MLFeatureValue(multiArray: res.inputIds),
            Feature.attentionMask: MLFeatureValue(multiArray: res.mask),
            Feature.refS: MLFeatureValue(multiArray: res.refS),
            Feature.randomPhases: MLFeatureValue(multiArray: res.randomPhases),
            Feature.speed: MLFeatureValue(multiArray: res.speed),
        ])

        let predictionStart = CFAbsoluteTimeGetCurrent()
        let output = try res.model.prediction(from: input)
        let predictionElapsed = CFAbsoluteTimeGetCurrent() - predictionStart

        if predictionElapsed > Self.slowPredictionThreshold {
            let ms = String(format: "%.0f", predictionElapsed * 1_000)
            Self.logger.warning(
                "\(bucket.modelName) prediction took \(ms)ms (possible ANE fallback to CPU/GPU)")
        }

        guard let audio = output.featureValue(for: Feature.audio)?.multiArrayValue else {
            throw KokoroError.inferenceFailed("Missing audio output")
        }

        // Models with speed output pred_dur_clamped; without speed output pred_dur.
        let durations: [Int]
        let predDur = output.featureValue(for: Feature.predDurClamped)?.multiArrayValue
            ?? output.featureValue(for: Feature.predDur)?.multiArrayValue
        if let predDur {
            durations = (0..<min(predDur.count, tokenIds.count)).map { predDur[$0].intValue }
        } else {
            durations = []
        }

        // Use pred_dur to compute expected audio length.
        // Each duration unit = sampleRate/24 audio samples (hop size).
        let validSamples: Int
        if !durations.isEmpty {
            let totalFrames = durations.reduce(0, +)
            validSamples = min(totalFrames * Self.hopSize, audio.count)
        } else if let lengthArray = output.featureValue(for: Feature.audioLength)?.multiArrayValue,
            lengthArray[0].intValue > 0, lengthArray[0].intValue < audio.count
        {
            validSamples = lengthArray[0].intValue
        } else {
            validSamples = audio.count
        }

        let samples = MLArrayHelpers.extractFloats(from: audio, maxCount: validSamples)
        return (samples, durations)
    }

    /// Trim trailing silence and artifacts in-place.
    ///
    /// Uses relative thresholds (percentage of peak amplitude) so it works
    /// on raw model output before normalization. Two passes:
    /// 1. Find the click (highest delta window followed by amplitude drop)
    /// 2. Find the first post-speech silence gap preceded by real speech
    private func trimTrailingArtifacts(_ samples: inout [Float]) {
        let windowSize = 2400  // 100ms at 24kHz
        var validCount = samples.count
        guard validCount > windowSize * 3 else { return }

        // Compute peak amplitude for relative thresholds.
        var peak: Float = 0
        for s in samples { peak = max(peak, abs(s)) }
        guard peak > 1e-6 else { return }
        let silenceThreshold = peak * 0.02  // 2% of peak = silence
        let speechThreshold = peak * 0.15   // 15% of peak = speech

        // --- Pass 1: Click detection (scan back third of audio) ---
        let searchFrom = validCount / 3
        let windowCount = (validCount - searchFrom) / windowSize
        if windowCount > 2 {
            struct WindowStats {
                let start: Int
                var maxDelta: Float = 0
                var maxAmp: Float = 0
            }
            let windows = (0..<windowCount).map { w -> WindowStats in
                let wStart = searchFrom + w * windowSize
                var stats = WindowStats(start: wStart)
                for j in wStart..<(wStart + windowSize) {
                    stats.maxAmp = max(stats.maxAmp, abs(samples[j]))
                    if j > wStart {
                        stats.maxDelta = max(stats.maxDelta, abs(samples[j] - samples[j - 1]))
                    }
                }
                return stats
            }

            var bestClickIdx = -1
            var bestDelta: Float = 0
            for w in 0..<(windowCount - 2) {
                let delta = windows[w].maxDelta
                guard delta > peak * 0.1 else { continue }
                let amp = windows[w].maxAmp
                let nextAmp = max(windows[w + 1].maxAmp, windows[w + 2].maxAmp)
                if nextAmp < amp * 0.5 && delta > bestDelta {
                    bestDelta = delta
                    bestClickIdx = w
                }
            }

            if bestClickIdx >= 0 {
                validCount = windows[bestClickIdx].start
            }
        }

        // --- Pass 2: End-of-speech silence gap ---
        let baseLookback = max(5, 12000 / windowSize)
        let lookback = min(baseLookback, validCount / windowSize / 3)
        let pass2Count = validCount / windowSize
        if pass2Count > lookback {
            let pass2Stats = (0..<pass2Count).map { w -> (start: Int, maxAmp: Float) in
                let wStart = w * windowSize
                var ma: Float = 0
                for j in wStart..<min(wStart + windowSize, validCount) {
                    ma = max(ma, abs(samples[j]))
                }
                return (wStart, ma)
            }
            for w in lookback..<pass2Count {
                if pass2Stats[w].maxAmp < silenceThreshold {
                    let hasSpeech = (max(0, w - lookback)..<w).contains {
                        pass2Stats[$0].maxAmp > speechThreshold
                    }
                    if hasSpeech {
                        validCount = pass2Stats[w].start
                        break
                    }
                }
            }
        }

        if validCount < samples.count {
            samples.removeSubrange(validCount..<samples.count)
        }

        // Fade-out: 20ms (always applied)
        let fadeOut = min(480, samples.count)
        let fadeStart = samples.count - fadeOut
        for i in 0..<fadeOut { samples[fadeStart + i] *= Float(fadeOut - i) / Float(fadeOut) }
    }

    // Precomputed biquad coefficients for presence boost EQ.
    // Peaking EQ: center=2.5kHz, Q=0.8, gain=+2dB at 24kHz sample rate.
    private static let eqCoeffs: (nb0: Float, nb1: Float, nb2: Float, na1: Float, na2: Float) = {
        let fc: Float = 2500.0
        let fs = Float(sampleRate)
        let A = powf(10.0, 2.0 / 40.0)  // +2dB
        let w0 = 2.0 * Float.pi * fc / fs
        let sinW0 = sinf(w0)
        let cosW0 = cosf(w0)
        let alpha = sinW0 / (2.0 * 0.8)  // Q=0.8
        let a0 = 1.0 + alpha / A
        return (
            nb0: (1.0 + alpha * A) / a0,
            nb1: (-2.0 * cosW0) / a0,
            nb2: (1.0 - alpha * A) / a0,
            na1: (-2.0 * cosW0) / a0,
            na2: (1.0 - alpha / A) / a0
        )
    }()

    // Precomputed high-pass filter coefficient: alpha = 1 - (2π * 80Hz / sampleRate)
    private static let hpAlpha: Float = 1.0 - (2.0 * .pi * 80.0 / Float(sampleRate))

    /// Post-process audio in-place: high-pass filter, presence boost, peak normalize.
    private func postProcess(_ samples: inout [Float]) {
        guard samples.count > 1 else { return }

        // 1. DC offset removal + high-pass at ~80Hz (single-pole IIR)
        let alpha = Self.hpAlpha
        let prechargeCount = min(240, samples.count)  // 10ms at 24kHz

        // Pre-charge: run filter backwards on first 10ms to warm up state
        var prev: Float = 0
        var prevOut: Float = 0
        for i in stride(from: prechargeCount - 1, through: 0, by: -1) {
            let x = samples[i]
            prevOut = (x - prev) + alpha * prevOut
            prev = x
        }

        // Forward pass with warmed-up state
        for i in 0..<samples.count {
            let x = samples[i]
            prevOut = (x - prev) + alpha * prevOut
            prev = x
            samples[i] = prevOut
        }

        // 2. Presence boost biquad
        let c = Self.eqCoeffs

        // Pre-charge biquad on reversed first 10ms
        var x1: Float = 0, x2: Float = 0, y1: Float = 0, y2: Float = 0
        for i in stride(from: prechargeCount - 1, through: 0, by: -1) {
            let x = samples[i]
            let y = c.nb0 * x + c.nb1 * x1 + c.nb2 * x2 - c.na1 * y1 - c.na2 * y2
            x2 = x1; x1 = x
            y2 = y1; y1 = y
        }

        // Forward pass with warmed-up state
        for i in 0..<samples.count {
            let x = samples[i]
            let y = c.nb0 * x + c.nb1 * x1 + c.nb2 * x2 - c.na1 * y1 - c.na2 * y2
            x2 = x1; x1 = x
            y2 = y1; y1 = y
            samples[i] = y
        }

        // 3. Peak normalize to 0.95 (-0.5dBFS)
        var peak: Float = 0
        vDSP_maxmgv(samples, 1, &peak, vDSP_Length(samples.count))
        if peak > 0.001 {
            var scale = Float(0.95) / peak
            vDSP_vsmul(samples, 1, &scale, &samples, 1, vDSP_Length(samples.count))
        }
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

    /// Which model bucket was used (last sentence's bucket if multi-sentence).
    public let bucket: ModelBucket?

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
        synthesisTime: TimeInterval,
        bucket: ModelBucket? = nil
    ) {
        self.samples = samples
        self.tokenDurations = tokenDurations
        self.tokenCount = tokenCount
        self.synthesisTime = synthesisTime
        self.bucket = bucket
    }
}

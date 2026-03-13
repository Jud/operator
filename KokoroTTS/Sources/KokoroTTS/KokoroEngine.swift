import Accelerate
import CoreML
import Foundation
import NaturalLanguage
import os

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

    /// Maximum token count for the unified model.
    static let maxTokenCount = 242

    /// CoreML model bundle name (without .mlmodelc extension).
    static let modelName = "kokoro_24_10s"

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
        static let predDur = "pred_dur_clamped"
        static let speed = "speed"
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
    private let model: MLModel

    // Pre-allocated MLMultiArray buffers reused across synthesis calls.
    private let preInputIds: MLMultiArray
    private let preMask: MLMultiArray
    private let preRefS: MLMultiArray
    private let preRandomPhases: MLMultiArray
    private let preSpeed: MLMultiArray

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
        let url = modelDirectory.appendingPathComponent(Self.modelName + ".mlmodelc")
        self.model = try MLModel(contentsOf: url, configuration: config)

        let maxTokens = Self.maxTokenCount
        self.preInputIds = try MLMultiArray(shape: [1, maxTokens as NSNumber], dataType: .int32)
        self.preMask = try MLMultiArray(shape: [1, maxTokens as NSNumber], dataType: .int32)
        self.preRefS = try MLMultiArray(
            shape: [1, VoiceStore.styleDim as NSNumber], dataType: .float32)
        self.preRandomPhases = try MLMultiArray(
            shape: [1, Self.numPhases as NSNumber], dataType: .float32)
        self.preSpeed = try MLMultiArray(shape: [1], dataType: .float32)
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
        let clampedSpeed = min(max(speed, Self.speedRange.lowerBound), Self.speedRange.upperBound)

        var allSamples: [Float] = []
        var allDurations: [Int] = []
        var totalTokens = 0

        for sentence in splitSentences(text) {
            let phonemes = lockedPhonemize(sentence)
            let tokenIds = tokenizer.encode(phonemes)
            totalTokens += tokenIds.count
            let (samples, durations) = try synthesizeUnified(
                tokenIds: tokenIds, styleVector: styleVector, speed: clampedSpeed)
            if allSamples.isEmpty {
                allSamples = samples
            } else {
                allSamples.append(contentsOf: samples)
            }
            allDurations.append(contentsOf: durations)
        }

        trimTrailingArtifacts(&allSamples)
        postProcess(&allSamples)

        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        return SynthesisResult(
            samples: allSamples, tokenDurations: allDurations,
            tokenCount: totalTokens, synthesisTime: elapsed)
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
        do {
            let dummyTokens = [Int](repeating: 0, count: Self.maxTokenCount)
            let dummyStyle = [Float](repeating: 0, count: VoiceStore.styleDim)
            _ = try synthesizeUnified(
                tokenIds: dummyTokens, styleVector: dummyStyle, speed: 1.0)
        } catch {
            Self.logger.warning("Warmup failed (non-fatal): \(error.localizedDescription)")
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
        guard tokenIds.count <= Self.maxTokenCount else {
            throw KokoroError.textTooLong(
                tokenCount: tokenIds.count, maxTokens: Self.maxTokenCount)
        }

        synthesizeLock.lock()
        defer { synthesizeLock.unlock() }

        let maxTokens = Self.maxTokenCount
        MLArrayHelpers.fillTokenInputs(
            from: tokenIds, into: preInputIds, mask: preMask, maxLength: maxTokens)
        MLArrayHelpers.fillStyleArray(from: styleVector, into: preRefS)

        let phasePtr = preRandomPhases.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<Self.numPhases {
            phasePtr[i] = Float.random(in: 0..<(2 * .pi))
        }

        preSpeed[0] = speed as NSNumber

        let input = try MLDictionaryFeatureProvider(dictionary: [
            Feature.inputIds: MLFeatureValue(multiArray: preInputIds),
            Feature.attentionMask: MLFeatureValue(multiArray: preMask),
            Feature.refS: MLFeatureValue(multiArray: preRefS),
            Feature.randomPhases: MLFeatureValue(multiArray: preRandomPhases),
            Feature.speed: MLFeatureValue(multiArray: preSpeed),
        ])

        let predictionStart = CFAbsoluteTimeGetCurrent()
        let output = try model.prediction(from: input)
        let predictionElapsed = CFAbsoluteTimeGetCurrent() - predictionStart

        if predictionElapsed > Self.slowPredictionThreshold {
            let ms = String(format: "%.0f", predictionElapsed * 1_000)
            Self.logger.warning(
                "model.prediction() took \(ms)ms (possible ANE fallback to CPU/GPU)")
        }

        guard let audio = output.featureValue(for: Feature.audio)?.multiArrayValue else {
            throw KokoroError.inferenceFailed("Missing audio output")
        }

        let durations: [Int]
        if let predDur = output.featureValue(for: Feature.predDur)?.multiArrayValue {
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

    /// Trim trailing click + artifacts in-place.
    ///
    /// The end-to-end model often produces a click (sharp transient) after
    /// speech, followed by repeating low-level artifacts. Two passes:
    /// 1. Find the click (highest delta window followed by amplitude drop)
    /// 2. Find the first post-speech silence gap preceded by real speech
    private func trimTrailingArtifacts(_ samples: inout [Float]) {
        let windowSize = 2400  // 100ms at 24kHz
        var validCount = samples.count

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
                guard delta > 0.2 else { continue }
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
        let lookback = max(5, 12000 / windowSize)
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
                if pass2Stats[w].maxAmp < 0.01 {
                    let hasSpeech = (max(0, w - lookback)..<w).contains { pass2Stats[$0].maxAmp > 0.3 }
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

import ArgumentParser
import AVFoundation
import CoreML
import Foundation
import KokoroTTS
import NaturalLanguage

@main
struct KokoroTwoStageSpike: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "spike",
        abstract: "Kokoro two-stage pipeline hypothesis tests",
        subcommands: [
            Say.self, Voices.self, Phonemize.self, Profile.self,
            H1ModelLoad.self, H2Alignment.self, H3Performance.self, H4Caching.self,
            E2EValidation.self, BucketBenchmark.self, ConfigBench.self,
            VocosLoad.self,
            VocosRun.self,
        ]
    )
}

// MARK: - Say: Synthesize and play text

struct Say: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "say",
        abstract: "Speak text using Kokoro TTS (streams sentence-by-sentence)"
    )

    @Argument(help: "Text to speak")
    var text: String

    @Option(name: .shortAndLong, help: "Voice preset name")
    var voice: String = "af_heart"

    @Option(name: .shortAndLong, help: "Speech speed (0.5–2.0)")
    var speed: Float = 1.0

    @Option(name: .shortAndLong, help: "Output WAV path (skips streaming playback)")
    var output: String?

    @Flag(name: .long, help: "Don't play audio, just write WAV")
    var noPlay: Bool = false

    @Flag(name: .long, help: "Disable streaming (synthesize all at once)")
    var noStream: Bool = false

    @Flag(name: .long, help: "Print per-stage timing breakdown")
    var verbose: Bool = false

    func run() async throws {
        let modelDir = ModelManager.defaultDirectory
        let engine = try KokoroEngine(modelDirectory: modelDir)

        if noStream || output != nil {
            try runBatch(engine: engine)
        } else {
            try await runStreaming(engine: engine)
        }
    }

    private func runBatch(engine: KokoroEngine) throws {
        let result = try engine.synthesize(text: text, voice: voice, speed: speed, verbose: verbose)
        let synthMs = String(format: "%.0f", result.synthesisTime * 1_000)
        let audioDur = String(format: "%.2f", result.duration)
        let rtfx = String(format: "%.1f", result.duration / result.synthesisTime)
        print("\(synthMs)ms synth | \(audioDur)s audio | \(rtfx)x RTFx | \(result.tokenCount) tokens | voice: \(voice)")

        let wavPath = output ?? "/tmp/kokoro_say.wav"
        try E2EValidation().writeWAV(
            samples: result.samples, sampleRate: KokoroEngine.sampleRate, to: wavPath)

        if noPlay {
            print(wavPath)
        } else {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/afplay")
            process.arguments = [wavPath]
            try process.run()
            process.waitUntilExit()
        }
    }

    private func runStreaming(engine: KokoroEngine) async throws {
        let sentences = splitSentences(text)
        if sentences.count <= 1 {
            try runBatch(engine: engine)
            return
        }

        let format = AVAudioFormat(
            standardFormatWithSampleRate: Double(KokoroEngine.sampleRate), channels: 1)!
        let audioEngine = AVAudioEngine()
        let playerNode = AVAudioPlayerNode()
        audioEngine.attach(playerNode)
        audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: format)
        try audioEngine.start()
        playerNode.play()

        let overallStart = CFAbsoluteTimeGetCurrent()
        var totalAudioDuration = 0.0
        let completionStream = AsyncStream<Void> { continuation in
            var scheduled = 0

            for (i, sentence) in sentences.enumerated() {
                let trimmed = sentence.trimmingCharacters(in: .whitespacesAndNewlines)
                if trimmed.isEmpty { continue }

                guard let result = try? engine.synthesize(text: trimmed, voice: voice, speed: speed) else {
                    continue
                }
                totalAudioDuration += result.duration

                let synthMs = String(format: "%.0f", result.synthesisTime * 1_000)
                let elapsed = String(format: "%.0f", (CFAbsoluteTimeGetCurrent() - overallStart) * 1_000)
                print("[\(elapsed)ms] chunk \(i + 1)/\(sentences.count): \(synthMs)ms synth → \(String(format: "%.1f", result.duration))s audio | \"\(trimmed.prefix(50))\"")

                let frameCount = AVAudioFrameCount(result.samples.count)
                guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount),
                      let channelData = buffer.floatChannelData?[0] else { continue }
                buffer.frameLength = frameCount
                result.samples.withUnsafeBufferPointer { src in
                    guard let base = src.baseAddress else { return }
                    channelData.update(from: base, count: result.samples.count)
                }

                scheduled += 1
                let isLast = (i == sentences.count - 1)
                playerNode.scheduleBuffer(buffer) {
                    if isLast { continuation.finish() }
                }
            }

            if scheduled == 0 { continuation.finish() }
        }

        let totalSynth = CFAbsoluteTimeGetCurrent() - overallStart
        print(String(format: "\nTotal: %.0fms synth for %.1fs audio (%.1fx RTFx) | %d chunks | voice: %@",
                     totalSynth * 1_000, totalAudioDuration,
                     totalAudioDuration / totalSynth, sentences.count, voice))

        // Wait for all buffers to finish playing
        for await _ in completionStream {}

        playerNode.stop()
        audioEngine.stop()
    }

    private func splitSentences(_ text: String) -> [String] {
        var sentences: [String] = []
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = text
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            sentences.append(String(text[range]))
            return true
        }
        return sentences.isEmpty ? [text] : sentences
    }
}

// MARK: - Profile: Instrument synthesis pipeline stages

struct Profile: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "profile",
        abstract: "Profile synthesis pipeline — time each stage"
    )

    @Option(name: .shortAndLong, help: "Voice preset name")
    var voice: String = "af_heart"

    @Option(name: .long, help: "Number of iterations")
    var iterations: Int = 10

    func run() async throws {
        let modelDir = ModelManager.defaultDirectory
        print("=== Synthesis Pipeline Profile ===\n")

        // Phase 0: Engine init (one-time)
        let t0 = CFAbsoluteTimeGetCurrent()
        let engine = try KokoroEngine(modelDirectory: modelDir)
        print(String(format: "Engine init: %.1fms", (CFAbsoluteTimeGetCurrent() - t0) * 1_000))

        // Warm up
        _ = try engine.synthesize(text: "warmup", voice: voice)
        _ = try engine.synthesize(text: "warmup", voice: voice)
        print("Warm-up done.\n")

        let testCases: [(label: String, text: String)] = [
            ("1-word", "Hello."),
            ("short", "The quick brown fox jumps over the lazy dog."),
            ("medium", "Operator uses neural speech synthesis for natural voice output. Each agent gets a distinct voice."),
            ("long", "Here's the thing about streaming synthesis. The first sentence starts playing while I'm still working on the next one. So you hear audio much sooner. That means lower latency for the user."),
        ]

        // Phase 1: Phonemizer + Tokenizer timing (isolated)
        print("--- Phase 1: Phonemizer + Tokenizer ---")
        for (label, text) in testCases {
            var phonemizeTimes = [Double]()
            for _ in 0..<iterations {
                let tp = CFAbsoluteTimeGetCurrent()
                let (_, _) = engine.phonemize(text)
                phonemizeTimes.append(CFAbsoluteTimeGetCurrent() - tp)
            }
            let avg = phonemizeTimes.reduce(0, +) / Double(phonemizeTimes.count) * 1_000
            let min_ = phonemizeTimes.min()! * 1_000
            print(String(format: "  %-8s  avg: %6.2fms  min: %6.2fms", label, avg, min_))
        }

        // Phase 2: Input preparation timing (MLMultiArray creation)
        print("\n--- Phase 2: MLMultiArray Input Prep ---")
        let (_, tokenIds) = engine.phonemize(testCases[2].text)
        let styleVector = try getStyleVector(modelDir: modelDir, voice: voice)
        let maxTokens = 124  // v21_5s bucket
        var prepTimes = [Double]()
        for _ in 0..<iterations {
            let tp = CFAbsoluteTimeGetCurrent()
            let padded = tokenIds + [Int](repeating: 0, count: max(0, maxTokens - tokenIds.count))
            let _ = try makeTokenInputs(tokenIds: padded, maxLength: maxTokens)
            let _ = try makeStyleArray(styleVector)
            let randomPhases = try MLMultiArray(shape: [1, 9], dataType: .float32)
            let phasePtr = randomPhases.dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0..<9 { phasePtr[i] = Float.random(in: 0..<(2 * .pi)) }
            prepTimes.append(CFAbsoluteTimeGetCurrent() - tp)
        }
        let avgPrep = prepTimes.reduce(0, +) / Double(prepTimes.count) * 1_000
        print(String(format: "  avg: %.3fms  min: %.3fms", avgPrep, prepTimes.min()! * 1_000))

        // Phase 3: CoreML prediction timing (the big one)
        print("\n--- Phase 3: CoreML model.prediction() ---")
        for (label, text) in testCases {
            var predTimes = [Double]()
            for _ in 0..<iterations {
                // Full synthesize call — but we already know phases 1+2 are negligible
                let tp = CFAbsoluteTimeGetCurrent()
                let _ = try engine.synthesize(text: text, voice: voice)
                predTimes.append(CFAbsoluteTimeGetCurrent() - tp)
            }
            let avg = predTimes.reduce(0, +) / Double(predTimes.count) * 1_000
            let min_ = predTimes.min()! * 1_000
            let max_ = predTimes.max()! * 1_000
            print(String(format: "  %-8s  avg: %7.1fms  min: %7.1fms  max: %7.1fms", label, avg, min_, max_))
        }

        // Phase 4: Output extraction timing
        print("\n--- Phase 4: Audio Sample Extraction ---")
        // Simulate extracting ~120000 samples (5s audio)
        let dummyArray = try MLMultiArray(shape: [1, 1, 120000], dataType: .float32)
        let dPtr = dummyArray.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<120000 { dPtr[i] = Float.random(in: -1...1) }
        var extractTimes = [Double]()
        for _ in 0..<iterations {
            let tp = CFAbsoluteTimeGetCurrent()
            let _ = extractFloats(from: dummyArray, maxCount: 120000)
            extractTimes.append(CFAbsoluteTimeGetCurrent() - tp)
        }
        let avgExtract = extractTimes.reduce(0, +) / Double(extractTimes.count) * 1_000
        print(String(format: "  120k samples: avg: %.3fms  min: %.3fms", avgExtract, extractTimes.min()! * 1_000))

        // Phase 5: Summary
        print("\n--- Phase 5: Summary ---")
        print("If prediction dominates (>95%), the only levers are:")
        print("  1. Quantized model (INT8/FP16 weights) — re-export via coremltools")
        print("  2. Smaller bucket (v21_5s vs v24_10s) for short text")
        print("  3. Different compute unit (ANE vs GPU)")
        print("  4. Pre-allocated MLMultiArray reuse (avoids alloc per call)")
        print("  5. Async prediction pipeline (predict next while playing current)")
    }

    private func makeTokenInputs(tokenIds: [Int], maxLength: Int) throws -> (MLMultiArray, MLMultiArray) {
        let inputIds = try MLMultiArray(shape: [1, maxLength as NSNumber], dataType: .int32)
        let mask = try MLMultiArray(shape: [1, maxLength as NSNumber], dataType: .int32)
        let iPtr = inputIds.dataPointer.assumingMemoryBound(to: Int32.self)
        let mPtr = mask.dataPointer.assumingMemoryBound(to: Int32.self)
        for i in 0..<maxLength {
            if i < tokenIds.count {
                iPtr[i] = Int32(tokenIds[i]); mPtr[i] = 1
            } else {
                iPtr[i] = 0; mPtr[i] = 0
            }
        }
        return (inputIds, mask)
    }

    private func makeStyleArray(_ sv: [Float]) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: [1, 256], dataType: .float32)
        let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<min(sv.count, 256) { ptr[i] = sv[i] }
        return arr
    }

    private func extractFloats(from array: MLMultiArray, maxCount: Int) -> [Float] {
        let count = min(array.count, maxCount)
        let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
        return [Float](UnsafeBufferPointer(start: ptr, count: count))
    }

    private func getStyleVector(modelDir: URL, voice: String) throws -> [Float] {
        let voiceURL = modelDir.appendingPathComponent("voices/\(voice).json")
        let data = try Data(contentsOf: voiceURL)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let embedding = json["embedding"] as? [Double] else {
            throw KokoroError.voiceNotFound(voice)
        }
        return embedding.prefix(256).map { Float($0) }
    }
}

// MARK: - Voices: List available voices

struct Voices: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "voices",
        abstract: "List available Kokoro voice presets"
    )

    func run() async throws {
        let modelDir = ModelManager.defaultDirectory
        let engine = try KokoroEngine(modelDirectory: modelDir)
        let voices = engine.availableVoices.sorted()

        print("\(voices.count) voices available:\n")
        for voice in voices {
            print("  \(voice)")
        }
        print("\nUsage: spike say \"Hello world\" --voice \(voices.first ?? "af_heart")")
    }
}

// MARK: - Phonemize: Debug phoneme output

struct Phonemize: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "phonemize",
        abstract: "Show IPA phonemes and token IDs for text (debug)"
    )

    @Argument(help: "Text to phonemize")
    var text: String

    func run() async throws {
        let modelDir = ModelManager.defaultDirectory
        let engine = try KokoroEngine(modelDirectory: modelDir)

        let (phonemes, tokenIds) = engine.phonemize(text)
        print("Text:     \(text)")
        print("IPA:      \(phonemes)")
        print("Tokens:   \(tokenIds.count) IDs")
        print("Token IDs: \(tokenIds)")
    }
}

// MARK: - E2E: Full KokoroTTS engine validation

struct E2EValidation: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "e2e",
        abstract: "End-to-end KokoroTTS engine: init → synthesize → benchmark"
    )

    @Option(name: .long, help: "Number of benchmark iterations")
    var iterations: Int = 10

    func run() async throws {
        print("=== E2E: KokoroTTS Engine Validation ===\n")

        let modelDir = ModelManager.defaultDirectory
        print("Model directory: \(modelDir.path)")
        print("Models available: \(ModelManager.modelsAvailable(at: modelDir))\n")

        // Phase 1: Engine initialization
        print("--- Phase 1: Engine Init ---")
        let t0 = CFAbsoluteTimeGetCurrent()
        let engine = try KokoroEngine(modelDirectory: modelDir)
        let initTime = CFAbsoluteTimeGetCurrent() - t0
        print("Engine init: \(String(format: "%.2f", initTime))s")
        print("Pipeline: unified (end-to-end)")
        print("Voices: \(engine.availableVoices.count) (\(engine.availableVoices.prefix(5).joined(separator: ", "))...)")

        // Phase 2: Warm-up
        print("\n--- Phase 2: Warm-up ---")
        let tw = CFAbsoluteTimeGetCurrent()
        try engine.warmUp()
        let warmTime = CFAbsoluteTimeGetCurrent() - tw
        print("Warm-up: \(String(format: "%.2f", warmTime))s")

        // Phase 3: Synthesis validation
        print("\n--- Phase 3: Synthesis Validation ---")
        let testCases: [(text: String, voice: String)] = [
            ("Hello world.", "af_heart"),
            ("The quick brown fox jumps over the lazy dog.", "am_adam"),
            ("I can speak with natural intonation and clear pronunciation.", "af_bella"),
        ]

        for (text, voice) in testCases {
            let result = try engine.synthesize(text: text, voice: voice)
            let synthMs = String(format: "%.0f", result.synthesisTime * 1000)
            let audioDur = String(format: "%.2f", result.duration)
            let rtfx = String(format: "%.1f", result.duration / result.synthesisTime)

            // Check audio has energy
            var rms: Float = 0
            for s in result.samples { rms += s * s }
            rms = sqrt(rms / Float(max(result.samples.count, 1)))

            let status = rms > 0.001 ? "PASS" : "FAIL"
            print("  \(status): \"\(text.prefix(40))...\" → \(synthMs)ms synth, \(audioDur)s audio, \(rtfx)x RTFx, RMS=\(String(format: "%.4f", rms))")
        }

        // Phase 4: Write a test WAV to verify audio quality
        print("\n--- Phase 4: Audio Quality Check ---")
        let qualityResult = try engine.synthesize(
            text: "This is a test of the Kokoro text to speech engine integrated into Operator.",
            voice: "af_heart"
        )
        let wavPath = "/tmp/kokoro_e2e_test.wav"
        try writeWAV(samples: qualityResult.samples, sampleRate: KokoroEngine.sampleRate, to: wavPath)
        print("WAV written to \(wavPath) (\(String(format: "%.2f", qualityResult.duration))s)")
        print("Listen: open \(wavPath)")

        // Phase 5: Bucket comparison
        print("\n--- Phase 5: Bucket Comparison ---")
        let bucketCases: [(label: String, text: String)] = [
            ("1-word v21", "Hello."),
            ("1-sentence v21", "The quick brown fox jumps over the lazy dog."),
            ("2-sentence v21", "Operator uses neural speech. Each agent gets a distinct voice."),
            ("long v24", "Operator is a voice orchestration layer that manages concurrent Claude Code sessions and provides natural speech for each agent."),
        ]

        for (label, text) in bucketCases {
            // Warm-up
            _ = try engine.synthesize(text: text, voice: "af_heart")
            _ = try engine.synthesize(text: text, voice: "af_heart")

            var synthTimes = [Double]()
            var lastResult: SynthesisResult?
            for _ in 0..<iterations {
                let r = try engine.synthesize(text: text, voice: "af_heart")
                synthTimes.append(r.synthesisTime)
                lastResult = r
            }
            guard let result = lastResult else { continue }
            let avgSynth = synthTimes.reduce(0, +) / Double(synthTimes.count)
            let rtfx = result.duration / avgSynth
            print(String(format: "  %-20s  %7.0fms synth, %5.2fs audio, %5.1fx RTFx, %d tokens",
                         label, avgSynth * 1000, result.duration, rtfx, result.tokenCount))
        }

        // Phase 6: Full benchmark with 10s model
        print("\n--- Phase 6: Benchmark (\(iterations) iterations) ---")
        let benchText = "Operator uses neural speech synthesis for natural voice output. Each agent gets a distinct voice."
        let benchVoice = "af_heart"

        print("  Run    Synth(ms)    Audio(s)    RTFx")
        print("  ---    ---------    --------    ----")

        var rtfxValues = [Double]()
        for i in 1...iterations {
            let r = try engine.synthesize(text: benchText, voice: benchVoice)
            let rtfx = r.duration / r.synthesisTime
            let warmup = i <= 2 ? " (warmup)" : ""
            print(String(format: "  %3d    %8.1f    %7.2f    %5.1f%@",
                         i, r.synthesisTime * 1000, r.duration, rtfx, warmup))
            if i > 2 { rtfxValues.append(rtfx) }
        }

        let avgRTFx = rtfxValues.reduce(0, +) / Double(rtfxValues.count)
        let minRTFx = rtfxValues.min() ?? 0
        let maxRTFx = rtfxValues.max() ?? 0
        print(String(format: "\n  Avg RTFx: %.1f (min: %.1f, max: %.1f)", avgRTFx, minRTFx, maxRTFx))

        // Phase 6: Summary
        print("\n=== E2E SUMMARY ===")
        print("Init time:     \(String(format: "%.2f", initTime))s")
        print("Warm-up time:  \(String(format: "%.2f", warmTime))s")
        print("Pipeline:      unified (end-to-end)")
        print("Avg RTFx:      \(String(format: "%.1f", avgRTFx))x")
        print("Voices:        \(engine.availableVoices.count)")

        if avgRTFx >= 2.0 {
            print("\nVERDICT: SHIP IT — unified model meets performance bar for v1")
        } else {
            print("\nVERDICT: NEEDS OPTIMIZATION — RTFx below 2x")
        }
    }

    func writeWAV(samples: [Float], sampleRate: Int, to path: String) throws {
        let numSamples = samples.count
        let dataSize = numSamples * 2  // 16-bit PCM
        let fileSize = 36 + dataSize

        var data = Data()
        // RIFF header
        data.append(contentsOf: "RIFF".utf8)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(fileSize).littleEndian) { Array($0) })
        data.append(contentsOf: "WAVE".utf8)
        // fmt chunk
        data.append(contentsOf: "fmt ".utf8)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })  // PCM
        data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })  // mono
        data.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate * 2).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(2).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(16).littleEndian) { Array($0) })
        // data chunk
        data.append(contentsOf: "data".utf8)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(dataSize).littleEndian) { Array($0) })

        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            let int16 = Int16(clamped * 32767)
            data.append(contentsOf: withUnsafeBytes(of: int16.littleEndian) { Array($0) })
        }

        try data.write(to: URL(fileURLWithPath: path))
    }
}

// MARK: - Bucket Benchmark: Compare v21_5s vs v24_10s

struct BucketBenchmark: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "bucket-bench",
        abstract: "Compare synthesis speed across unified model buckets (v21_5s vs v24_10s)"
    )

    @Option(name: .long, help: "Number of benchmark iterations per case")
    var iterations: Int = 10

    func run() async throws {
        print("=== Bucket Benchmark: v21_5s vs v24_10s ===\n")

        let modelDir = ModelManager.defaultDirectory
        let engine = try KokoroEngine(modelDirectory: modelDir)
        try engine.warmUp()

        let voice = "af_heart"

        let testCases: [(label: String, text: String)] = [
            ("1-word", "Hello."),
            ("short-sentence", "The quick brown fox jumps over the lazy dog."),
            ("2-sentences", "Operator uses neural speech synthesis. Each agent gets a distinct voice."),
            ("3-sentences", "Operator uses neural speech. Each agent gets a distinct voice. Messages are queued and played in order."),
        ]

        print(String(format: "%-16s  %8s  %8s  %6s  %6s  %s",
                      "Case", "Synth", "Audio", "RTFx", "Tokens", "Bucket"))
        print(String(repeating: "-", count: 72))

        for (label, text) in testCases {
            // Run warm-up iterations
            for _ in 0..<2 {
                _ = try engine.synthesize(text: text, voice: voice)
            }

            // Benchmark iterations
            var synthTimes = [Double]()
            var lastResult: SynthesisResult?
            for _ in 0..<iterations {
                let result = try engine.synthesize(text: text, voice: voice)
                synthTimes.append(result.synthesisTime)
                lastResult = result
            }

            guard let result = lastResult else { continue }
            let avgSynth = synthTimes.reduce(0, +) / Double(synthTimes.count)
            let rtfx = result.duration / avgSynth

            print(String(format: "%-16s  %7.0fms  %6.2fs  %5.1fx  %5d  %s",
                         label,
                         avgSynth * 1000,
                         result.duration,
                         rtfx,
                         result.tokenCount,
                         result.tokenCount <= 124 ? "v21_5s" : "v24_10s"))
        }

        // WAV quality comparison: write both short and long
        print("\n--- Audio Quality Comparison ---")
        let shortText = "Hello, this is a short test."
        let longText = "This is a longer test sentence to verify that the ten second model still produces natural sounding speech output."

        let shortResult = try engine.synthesize(text: shortText, voice: voice)
        let longResult = try engine.synthesize(text: longText, voice: voice)

        try E2EValidation().writeWAV(
            samples: shortResult.samples,
            sampleRate: KokoroEngine.sampleRate,
            to: "/tmp/kokoro_bucket_short.wav"
        )
        try E2EValidation().writeWAV(
            samples: longResult.samples,
            sampleRate: KokoroEngine.sampleRate,
            to: "/tmp/kokoro_bucket_long.wav"
        )
        print("Short (v21_5s): /tmp/kokoro_bucket_short.wav (\(String(format: "%.2f", shortResult.duration))s)")
        print("Long  (v24_10s): /tmp/kokoro_bucket_long.wav (\(String(format: "%.2f", longResult.duration))s)")
        print("\nListen to both to verify quality parity.")
    }
}

// MARK: - H1: Two-stage models exist and are loadable

struct H1ModelLoad: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "h1",
        abstract: "Download Duration Model + HAR Decoder, load each, print tensor specs"
    )

    @Option(name: .long, help: "Path to model directory (or will download)")
    var modelDir: String?

    func run() async throws {
        print("=== H1: Two-stage models exist and are loadable ===\n")

        let dir: URL
        if let path = modelDir {
            dir = URL(fileURLWithPath: path)
        } else {
            let appSupport = FileManager.default.urls(
                for: .applicationSupportDirectory, in: .userDomainMask).first!
            dir = appSupport
                .appendingPathComponent("com.operator")
                .appendingPathComponent("models")
                .appendingPathComponent("kokoro")
        }

        print("Model directory: \(dir.path)\n")

        // Check for two-stage models
        let durationURL = dir.appendingPathComponent("kokoro_duration.mlmodelc")
        let synth3sURL = dir.appendingPathComponent("kokoro_synthesizer_3s.mlmodelc")

        print("Checking two-stage models:")
        print("  kokoro_duration.mlmodelc:       \(FileManager.default.fileExists(atPath: durationURL.path) ? "FOUND" : "MISSING")")
        print("  kokoro_synthesizer_3s.mlmodelc:  \(FileManager.default.fileExists(atPath: synth3sURL.path) ? "FOUND" : "MISSING")")

        // Check for unified models (fallback)
        print("\nChecking unified models (fallback):")
        for name in ["kokoro_21_5s", "kokoro_21_10s", "kokoro_24_10s"] {
            let url = dir.appendingPathComponent("\(name).mlmodelc")
            let exists = FileManager.default.fileExists(atPath: url.path)
            print("  \(name).mlmodelc: \(exists ? "FOUND" : "MISSING")")

            if exists {
                print("  Loading \(name)...")
                let config = MLModelConfiguration()
                config.computeUnits = .all
                let t0 = CFAbsoluteTimeGetCurrent()
                do {
                    let model = try MLModel(contentsOf: url, configuration: config)
                    let elapsed = CFAbsoluteTimeGetCurrent() - t0
                    print("  Loaded in \(String(format: "%.2f", elapsed))s")
                    printModelSpec(model, name: name)
                } catch {
                    print("  FAILED to load: \(error)")
                }
            }
        }

        // Try loading two-stage models
        var twoStageOK = true
        for (url, name, units) in [
            (durationURL, "kokoro_duration", MLComputeUnits.cpuAndGPU),
            (synth3sURL, "kokoro_synthesizer_3s", MLComputeUnits.all),
        ] {
            if FileManager.default.fileExists(atPath: url.path) {
                print("\n--- \(name) ---")
                let config = MLModelConfiguration()
                config.computeUnits = units
                let t = CFAbsoluteTimeGetCurrent()
                do {
                    let model = try MLModel(contentsOf: url, configuration: config)
                    let elapsed = CFAbsoluteTimeGetCurrent() - t
                    print("  Loaded in \(String(format: "%.2f", elapsed))s")
                    printModelSpec(model, name: name)
                    print("  PASS: \(name) loads successfully")
                } catch {
                    print("  FAIL: \(error)")
                    twoStageOK = false
                }
            } else {
                twoStageOK = false
            }
        }

        print("\n=== H1 RESULT ===")
        if twoStageOK {
            print("PASS: Two-stage models found and loadable")
        } else {
            print("FAIL: Two-stage models not found or failed to load.")
            print("      Download from philippdxx/Kokoro-CoreML-Experimental-Models on HuggingFace.")
        }
    }

    func printModelSpec(_ model: MLModel, name: String) {
        print("  Inputs:")
        for input in model.modelDescription.inputDescriptionsByName {
            let desc = input.value
            let shape = desc.multiArrayConstraint?.shape ?? []
            print("    \(input.key): \(desc.type) shape=\(shape)")
        }
        print("  Outputs:")
        for output in model.modelDescription.outputDescriptionsByName {
            let desc = output.value
            let shape = desc.multiArrayConstraint?.shape ?? []
            print("    \(output.key): \(desc.type) shape=\(shape)")
        }
    }
}

// MARK: - H2: Alignment math produces valid audio

struct H2Alignment: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "h2",
        abstract: "Run full pipeline: Duration → Alignment → HAR → WAV output"
    )

    @Option(name: .long, help: "Path to model directory")
    var modelDir: String?

    @Option(name: .long, help: "Output WAV path")
    var output: String = "/tmp/kokoro_h2_test.wav"

    func run() async throws {
        print("=== H2: Alignment math produces valid audio ===\n")
        print("This test requires two-stage models (kokoro_duration + kokoro_har_*).")
        print("If not available, testing with unified model instead.\n")

        let dir: URL
        if let path = modelDir {
            dir = URL(fileURLWithPath: path)
        } else {
            let appSupport = FileManager.default.urls(
                for: .applicationSupportDirectory, in: .userDomainMask).first!
            dir = appSupport
                .appendingPathComponent("com.operator")
                .appendingPathComponent("models")
                .appendingPathComponent("kokoro")
        }

        // Test with unified model as proof of concept
        var modelURL: URL?
        for name in ["kokoro_24_10s", "kokoro_21_10s", "kokoro_21_5s"] {
            let url = dir.appendingPathComponent("\(name).mlmodelc")
            if FileManager.default.fileExists(atPath: url.path) {
                modelURL = url
                print("Using unified model: \(name)")
                break
            }
        }

        guard let url = modelURL else {
            print("SKIP: No models found at \(dir.path)")
            print("Run `spike h1` first to check model availability.")
            return
        }

        print("Loading model...")
        let config = MLModelConfiguration()
        config.computeUnits = .all
        let model = try MLModel(contentsOf: url, configuration: config)
        print("Model loaded.\n")

        // Get a voice embedding
        let voicesDir = dir.appendingPathComponent("voices")
        guard FileManager.default.fileExists(atPath: voicesDir.path) else {
            print("SKIP: No voices directory found")
            return
        }

        let files = try FileManager.default.contentsOfDirectory(
            at: voicesDir, includingPropertiesForKeys: nil)
        guard let firstVoice = files.first(where: { $0.pathExtension == "json" }) else {
            print("SKIP: No voice JSON files found")
            return
        }

        let voiceData = try Data(contentsOf: firstVoice)
        guard let voiceJSON = try JSONSerialization.jsonObject(with: voiceData) as? [String: Any],
              let embedding = voiceJSON["embedding"] as? [Double] else {
            print("SKIP: Could not parse voice embedding")
            return
        }
        let styleVector = embedding.prefix(256).map { Float($0) }
        print("Using voice: \(firstVoice.deletingPathExtension().lastPathComponent)")

        // Dummy token IDs (a simple test phrase)
        let tokenIds: [Int32] = [1, 26, 32, 61, 12, 22, 36, 12, 61, 5, 37, 29, 41, 15, 61, 8, 31, 12, 19, 2]
        let maxTokens = 242
        let paddedIds = tokenIds + [Int32](repeating: 0, count: maxTokens - tokenIds.count)

        let inputArray = try MLMultiArray(shape: [1, maxTokens as NSNumber], dataType: .int32)
        let maskArray = try MLMultiArray(shape: [1, maxTokens as NSNumber], dataType: .int32)
        let ptr = inputArray.dataPointer.assumingMemoryBound(to: Int32.self)
        let mptr = maskArray.dataPointer.assumingMemoryBound(to: Int32.self)
        for i in 0..<maxTokens {
            ptr[i] = paddedIds[i]
            mptr[i] = (i < tokenIds.count) ? 1 : 0
        }

        let refS = try MLMultiArray(shape: [1, 256 as NSNumber], dataType: .float32)
        let refPtr = refS.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<256 { refPtr[i] = styleVector[i] }

        let phases = try MLMultiArray(shape: [1, 9 as NSNumber], dataType: .float32)
        let pPtr = phases.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<9 { pPtr[i] = Float.random(in: 0..<(2 * .pi)) }

        print("Running inference...")
        let t0 = CFAbsoluteTimeGetCurrent()
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: inputArray),
            "attention_mask": MLFeatureValue(multiArray: maskArray),
            "ref_s": MLFeatureValue(multiArray: refS),
            "random_phases": MLFeatureValue(multiArray: phases),
        ])
        let result = try await model.prediction(from: input)
        let elapsed = CFAbsoluteTimeGetCurrent() - t0

        guard let audio = result.featureValue(for: "audio")?.multiArrayValue else {
            print("FAIL: No audio output")
            return
        }

        let sampleCount: Int
        if let length = result.featureValue(for: "audio_length_samples")?.multiArrayValue {
            sampleCount = length[0].intValue
        } else {
            sampleCount = audio.count
        }

        print("Inference: \(String(format: "%.1f", elapsed * 1000))ms")
        print("Audio samples: \(sampleCount)")
        print("Duration: \(String(format: "%.2f", Double(sampleCount) / 24000.0))s")

        // Check audio has energy
        var rms: Float = 0
        let aPtr = audio.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<min(sampleCount, audio.count) {
            rms += aPtr[i] * aPtr[i]
        }
        rms = sqrt(rms / Float(sampleCount))
        print("RMS energy: \(String(format: "%.6f", rms))")

        if rms > 0.001 {
            print("\nPASS: Audio has energy (RMS > 0.001)")
        } else {
            print("\nFAIL: Audio appears silent (RMS \(rms))")
        }
    }
}

// MARK: - H3: ANE performance target

struct H3Performance: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "h3",
        abstract: "Benchmark RTFx over 20 inferences (skip first 2 for warmup)"
    )

    @Option(name: .long, help: "Path to model directory")
    var modelDir: String?

    @Option(name: .long, help: "Number of inferences")
    var iterations: Int = 20

    func run() async throws {
        print("=== H3: Two-stage synthesizer performance (CPU+GPU) ===\n")

        let dir: URL
        if let path = modelDir {
            dir = URL(fileURLWithPath: path)
        } else {
            let appSupport = FileManager.default.urls(
                for: .applicationSupportDirectory, in: .userDomainMask).first!
            dir = appSupport
                .appendingPathComponent("com.operator")
                .appendingPathComponent("models")
                .appendingPathComponent("kokoro")
        }

        // Synthesizer 5s: d[1,640,20], t_en[1,512,20], s[1,128], ref_s[1,256], pred_aln_trg[20,200]
        // Output: waveform[1,120000] = 5s at 24kHz
        let synthURL = dir.appendingPathComponent("kokoro_synthesizer_5s.mlmodelc")
        guard FileManager.default.fileExists(atPath: synthURL.path) else {
            print("SKIP: kokoro_synthesizer_5s.mlmodelc not found at \(dir.path)")
            return
        }

        print("Loading kokoro_synthesizer_5s with .cpuAndGPU...")
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU
        let t0 = CFAbsoluteTimeGetCurrent()
        let model = try MLModel(contentsOf: synthURL, configuration: config)
        let loadTime = CFAbsoluteTimeGetCurrent() - t0
        print("Loaded in \(String(format: "%.2f", loadTime))s\n")

        // Prepare dummy inputs matching synthesizer specs
        // In production these come from the duration model + alignment
        let traceLen = 20
        let frames = 200
        let outputSamples = 120000
        let audioDuration = Double(outputSamples) / 24000.0  // 5s

        let d = try MLMultiArray(shape: [1, 640, traceLen as NSNumber], dataType: .float32)
        let tEn = try MLMultiArray(shape: [1, 512, traceLen as NSNumber], dataType: .float32)
        let s = try MLMultiArray(shape: [1, 128], dataType: .float32)
        let refS = try MLMultiArray(shape: [1, 256], dataType: .float32)
        let predAlnTrg = try MLMultiArray(shape: [traceLen as NSNumber, frames as NSNumber], dataType: .float32)

        // Fill with small random values to simulate real data
        func fillRandom(_ arr: MLMultiArray) {
            let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0..<arr.count { ptr[i] = Float.random(in: -0.1...0.1) }
        }
        fillRandom(d); fillRandom(tEn); fillRandom(s); fillRandom(refS); fillRandom(predAlnTrg)

        print("Running \(iterations) inferences (first 2 are warmup)...\n")
        print("  Run    Synth(ms)    Audio(s)    RTFx")
        print("  ---    ---------    --------    ----")

        var rtfxValues = [Double]()

        for i in 1...iterations {
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "d": MLFeatureValue(multiArray: d),
                "t_en": MLFeatureValue(multiArray: tEn),
                "s": MLFeatureValue(multiArray: s),
                "ref_s": MLFeatureValue(multiArray: refS),
                "pred_aln_trg": MLFeatureValue(multiArray: predAlnTrg),
            ])

            let start = CFAbsoluteTimeGetCurrent()
            let result = try await model.prediction(from: input)
            let elapsed = CFAbsoluteTimeGetCurrent() - start

            let waveform = result.featureValue(for: "waveform")!.multiArrayValue!
            let rtfx = audioDuration / elapsed

            let warmup = i <= 2 ? " (warmup)" : ""
            print(String(format: "  %3d    %8.1f    %7.2f    %5.1f%@",
                         i, elapsed * 1000, audioDuration, rtfx, warmup))

            if i > 2 { rtfxValues.append(rtfx) }

            // On first run, check output has data
            if i == 1 {
                let ptr = waveform.dataPointer.assumingMemoryBound(to: Float.self)
                var rms: Float = 0
                for j in 0..<min(10000, waveform.count) { rms += ptr[j] * ptr[j] }
                rms = sqrt(rms / 10000)
                print("    (output RMS: \(String(format: "%.6f", rms)), samples: \(waveform.count))")
            }
        }

        let avgRTFx = rtfxValues.reduce(0, +) / Double(rtfxValues.count)
        let minRTFx = rtfxValues.min() ?? 0
        let maxRTFx = rtfxValues.max() ?? 0

        print(String(format: "\n  Avg RTFx: %.1f (min: %.1f, max: %.1f)", avgRTFx, minRTFx, maxRTFx))

        print("\n=== H3 RESULT ===")
        if avgRTFx >= 10 {
            print("PASS: RTFx \(String(format: "%.1f", avgRTFx)) >= 10x target")
        } else if avgRTFx >= 5 {
            print("PARTIAL: RTFx \(String(format: "%.1f", avgRTFx)) — good enough for shipping")
        } else {
            print("NEEDS WORK: RTFx \(String(format: "%.1f", avgRTFx)) < 5x — consider HAR decoder path")
        }
    }
}

// MARK: - H4: Fixed-path caching fixes load time

struct H4Caching: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "h4",
        abstract: "Compare load time: fixed path vs fresh path"
    )

    @Option(name: .long, help: "Path to model directory")
    var modelDir: String?

    func run() async throws {
        print("=== H4: Fixed-path caching fixes load time ===\n")

        let dir: URL
        if let path = modelDir {
            dir = URL(fileURLWithPath: path)
        } else {
            let appSupport = FileManager.default.urls(
                for: .applicationSupportDirectory, in: .userDomainMask).first!
            dir = appSupport
                .appendingPathComponent("com.operator")
                .appendingPathComponent("models")
                .appendingPathComponent("kokoro")
        }

        // Find a model to test — prefer two-stage synthesizer (heaviest), fall back to unified
        let candidates = [
            ("kokoro_synthesizer_5s", MLComputeUnits.all),
            ("kokoro_duration", MLComputeUnits.cpuAndGPU),
            ("kokoro_24_10s", MLComputeUnits.all),
            ("kokoro_21_10s", MLComputeUnits.all),
            ("kokoro_21_5s", MLComputeUnits.all),
        ]

        var modelURL: URL?
        var modelName = ""
        var computeUnits = MLComputeUnits.all
        for (name, units) in candidates {
            let url = dir.appendingPathComponent("\(name).mlmodelc")
            if FileManager.default.fileExists(atPath: url.path) {
                modelURL = url
                modelName = name
                computeUnits = units
                break
            }
        }

        guard let url = modelURL else {
            print("SKIP: No models found at \(dir.path)")
            return
        }

        print("Model: \(modelName)")
        print("Path: \(url.path)")
        print("Compute units: \(computeUnits == .all ? ".all (ANE)" : ".cpuAndGPU")\n")

        // Three loads to measure compilation vs cached performance
        var loadTimes = [Double]()
        for i in 1...3 {
            let label = i == 1 ? "(should use ANE cache from H1)" : "(cached)"
            print("Load \(i) \(label)...")
            let config = MLModelConfiguration()
            config.computeUnits = computeUnits
            let t = CFAbsoluteTimeGetCurrent()
            _ = try MLModel(contentsOf: url, configuration: config)
            let elapsed = CFAbsoluteTimeGetCurrent() - t
            loadTimes.append(elapsed)
            print("  Time: \(String(format: "%.2f", elapsed))s")
        }

        let fastest = loadTimes.min()!
        print(String(format: "\nFastest load: %.2fs", fastest))

        print("\n=== H4 RESULT ===")
        if fastest <= 2.0 {
            print("PASS: Cached load (\(String(format: "%.2f", fastest))s) <= 2s target")
        } else if fastest <= 5.0 {
            print("PARTIAL: Cached load (\(String(format: "%.2f", fastest))s) > 2s but < 5s")
        } else {
            print("FAIL: Cached load (\(String(format: "%.2f", fastest))s) > 5s")
        }
    }
}

// MARK: - ConfigBench: A/B test MLModelConfiguration options

struct ConfigBench: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "config-bench",
        abstract: "A/B test MLModelConfiguration options (low-precision GPU, compute units)"
    )

    @Option(name: .shortAndLong, help: "Voice preset name")
    var voice: String = "af_heart"

    @Option(name: .long, help: "Iterations per config (after 2 warmup)")
    var iterations: Int = 10

    func run() async throws {
        print("=== MLModelConfiguration A/B Benchmark ===\n")

        let modelDir = ModelManager.defaultDirectory

        // Test each config sequentially — load model, benchmark, release before next
        let testText = "Operator uses neural speech synthesis for natural voice output. Each agent gets a distinct voice."

        // Configs to test (sequentially — only one model loaded at a time)
        let configSpecs: [(label: String, lowPrec: Bool, units: MLComputeUnits)] = [
            ("cpuAndGPU (baseline)", false, .cpuAndGPU),
            ("cpuAndGPU + lowPrecision", true, .cpuAndGPU),
            ("cpuOnly", false, .cpuOnly),
        ]

        print("Text: \"\(testText.prefix(60))...\"")
        print("Iterations: \(iterations) (+ 2 warmup)\n")

        print(String(format: "%-30s  %8s  %8s  %8s  %6s",
                      "Config", "Avg(ms)", "Min(ms)", "Max(ms)", "RTFx"))
        print(String(repeating: "-", count: 72))

        for (label, lowPrec, _) in configSpecs {
            // Fresh engine per config to avoid multi-model SIGSEGV
            do {
                let engine = try KokoroEngine(
                    modelDirectory: modelDir, allowLowPrecisionGPU: lowPrec)

                // Warmup
                _ = try engine.synthesize(text: "warmup", voice: voice)
                _ = try engine.synthesize(text: "warmup", voice: voice)

                // Benchmark
                var times = [Double]()
                var audioDur = 0.0
                for _ in 0..<iterations {
                    let t = CFAbsoluteTimeGetCurrent()
                    let result = try engine.synthesize(text: testText, voice: voice)
                    times.append(CFAbsoluteTimeGetCurrent() - t)
                    if audioDur == 0 { audioDur = result.duration }
                }

                let avg = times.reduce(0, +) / Double(times.count)
                let minT = times.min()!
                let maxT = times.max()!
                let rtfx = audioDur / avg

                print(String(format: "%-30s  %7.0f  %7.0f  %7.0f  %5.1fx",
                             label, avg * 1000, minT * 1000, maxT * 1000, rtfx))
            } catch {
                print(String(format: "%-30s  FAILED: %@", label, "\(error)"))
            }
        }

        print("\nNote: 'lowPrecision' uses FP16 accumulation on GPU — may slightly affect audio quality.")
    }
}

// MARK: - VocosLoad: Load FluidInference CoreML models and inspect tensor specs

struct VocosLoad: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "vocos-load",
        abstract: "Load FluidInference .mlmodelc files, test compute unit configs, print tensor specs"
    )

    func run() async throws {
        let modelDir = ModelManager.defaultDirectory

        // FluidInference model files to test
        let modelFiles: [(name: String, description: String)] = [
            ("kokoro_21_5s.mlmodelc", "Kokoro v0.21 (5s capacity, spec v7 / iOS 16+)"),
            ("kokoro_24_10s.mlmodelc", "Kokoro v0.24 (10s capacity, spec v8 / iOS 17+)"),
        ]

        let computeConfigs: [(units: MLComputeUnits, label: String)] = [
            (.cpuAndGPU, "cpuAndGPU"),
            (.cpuAndNeuralEngine, "cpuAndNeuralEngine"),
            (.all, "all"),
        ]

        print("=" * 80)
        print("FluidInference Kokoro CoreML Model Loader")
        print("Model directory: \(modelDir.path)")
        print("=" * 80)

        var results: [(model: String, config: String, loadTime: Double, success: Bool)] = []

        for modelFile in modelFiles {
            let modelURL = modelDir.appendingPathComponent(modelFile.name)

            print("\n" + "-" * 80)
            print("MODEL: \(modelFile.name)")
            print("Description: \(modelFile.description)")
            print("Path: \(modelURL.path)")

            // Check file exists
            guard FileManager.default.fileExists(atPath: modelURL.path) else {
                print("ERROR: Model file not found at \(modelURL.path)")
                continue
            }

            // Print file size
            if let attrs = try? FileManager.default.attributesOfItem(atPath: modelURL.path),
               let size = attrs[.size] as? UInt64 {
                print("Size: \(size / 1_000_000) MB")
            }

            // Test each compute unit configuration
            for config in computeConfigs {
                let mlConfig = MLModelConfiguration()
                mlConfig.computeUnits = config.units

                print("\n  --- Compute Units: \(config.label) ---")
                let startTime = CFAbsoluteTimeGetCurrent()

                do {
                    let model = try MLModel(contentsOf: modelURL, configuration: mlConfig)
                    let loadTime = CFAbsoluteTimeGetCurrent() - startTime

                    results.append((modelFile.name, config.label, loadTime, true))
                    print("  Load time: \(String(format: "%.3f", loadTime))s")

                    // Print input feature descriptions
                    let inputDesc = model.modelDescription.inputDescriptionsByName
                    print("  INPUTS (\(inputDesc.count)):")
                    for (name, desc) in inputDesc.sorted(by: { $0.key < $1.key }) {
                        print("    - \(name): \(formatFeatureDescription(desc))")
                    }

                    // Print output feature descriptions
                    let outputDesc = model.modelDescription.outputDescriptionsByName
                    print("  OUTPUTS (\(outputDesc.count)):")
                    for (name, desc) in outputDesc.sorted(by: { $0.key < $1.key }) {
                        print("    - \(name): \(formatFeatureDescription(desc))")
                    }

                    // Print metadata if available
                    if let metadata = model.modelDescription.metadata[.description] as? String,
                       !metadata.isEmpty {
                        print("  Metadata description: \(metadata)")
                    }
                    if let author = model.modelDescription.metadata[.author] as? String,
                       !author.isEmpty {
                        print("  Metadata author: \(author)")
                    }

                } catch {
                    let loadTime = CFAbsoluteTimeGetCurrent() - startTime
                    results.append((modelFile.name, config.label, loadTime, false))
                    print("  FAILED after \(String(format: "%.3f", loadTime))s")
                    print("  Error: \(error)")
                }
            }
        }

        // Build summary using string interpolation (String(format: %s) crashes with Swift Strings)
        func pad(_ s: String, _ width: Int) -> String {
            s.count >= width ? s : s + String(repeating: " ", count: width - s.count)
        }

        print("\n" + String(repeating: "=", count: 80))
        print("SUMMARY")
        print(String(repeating: "=", count: 80))
        print("\(pad("Model", 30))  \(pad("Compute Units", 22))  \(pad("Load (s)", 10))  Status")
        print(String(repeating: "-", count: 80))

        for r in results {
            let status = r.success ? "OK" : "FAILED"
            let loadStr = String(format: "%10.3f", r.loadTime)
            print("\(pad(r.model, 30))  \(pad(r.config, 22))  \(loadStr)  \(status)")
        }

        // Find fastest config per model
        let grouped = Dictionary(grouping: results.filter(\.success), by: \.model)
        print("")
        for (model, configs) in grouped.sorted(by: { $0.key < $1.key }) {
            if let fastest = configs.min(by: { $0.loadTime < $1.loadTime }) {
                print("Fastest for \(model): \(fastest.config) (\(String(format: "%.3f", fastest.loadTime))s)")
            }
        }

        // ANE availability check
        let aneConfigs = results.filter { ($0.config == "cpuAndNeuralEngine" || $0.config == "all") && $0.success }
        if aneConfigs.isEmpty {
            print("\nWARNING: No ANE-capable configurations loaded successfully!")
        } else {
            print("\nANE-capable configs loaded: \(aneConfigs.count) / \(results.filter { $0.config == "cpuAndNeuralEngine" || $0.config == "all" }.count)")
        }
        print("\n[Done]")
        fflush(stdout)
    }

    private func formatFeatureDescription(_ desc: MLFeatureDescription) -> String {
        var parts: [String] = []
        parts.append("type=\(featureTypeName(desc.type))")

        if let constraint = desc.multiArrayConstraint {
            let shape = constraint.shape.map { $0.intValue }
            parts.append("shape=\(shape)")
            parts.append("dtype=\(multiArrayDataTypeName(constraint.dataType))")
        }

        if desc.isOptional {
            parts.append("optional")
        }

        return parts.joined(separator: ", ")
    }

    private func featureTypeName(_ type: MLFeatureType) -> String {
        switch type {
        case .multiArray: return "MultiArray"
        case .double: return "Double"
        case .int64: return "Int64"
        case .string: return "String"
        case .image: return "Image"
        case .dictionary: return "Dictionary"
        case .sequence: return "Sequence"
        @unknown default: return "Unknown(\(type.rawValue))"
        }
    }

    private func multiArrayDataTypeName(_ type: MLMultiArrayDataType) -> String {
        switch type {
        case .float16: return "Float16"
        case .float32: return "Float32"
        case .float64: return "Float64"
        case .int32: return "Int32"
        @unknown default: return "Unknown(\(type.rawValue))"
        }
    }
}

// MARK: - VocosRun: End-to-end two-stage pipeline benchmark

struct VocosRun: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "vocos-run",
        abstract: "Run FluidInference unified models end-to-end, measure per-stage timing and RTFx"
    )

    @Option(name: .shortAndLong, help: "Voice preset name")
    var voice: String = "af_heart"

    @Option(name: .long, help: "Number of benchmark iterations (after warmup)")
    var iterations: Int = 5

    @Flag(name: .long, help: "Skip audio output")
    var noAudio: Bool = false

    func run() async throws {
        let modelDir = ModelManager.defaultDirectory
        let sampleRate = KokoroEngine.sampleRate

        print(String(repeating: "=", count: 80))
        print("Vocos/FluidInference Two-Stage Pipeline Benchmark")
        print(String(repeating: "=", count: 80))
        print("Model directory: \(modelDir.path)")
        print("Voice: \(voice)")
        print("Iterations: \(iterations)")
        print()

        // -- Step 1: Load both FluidInference CoreML models --
        print("--- Step 1: Load FluidInference CoreML models ---")

        let modelFiles: [(name: String, label: String)] = [
            ("kokoro_21_5s.mlmodelc", "v0.21-5s"),
            ("kokoro_24_10s.mlmodelc", "v0.24-10s"),
        ]

        var loadedModels: [(label: String, model: MLModel, loadTime: Double)] = []

        for (name, label) in modelFiles {
            let url = modelDir.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.path) else {
                print("  SKIP \(name) -- not found")
                continue
            }

            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndNeuralEngine

            let t0 = CFAbsoluteTimeGetCurrent()
            do {
                let model = try MLModel(contentsOf: url, configuration: config)
                let elapsed = CFAbsoluteTimeGetCurrent() - t0
                loadedModels.append((label, model, elapsed))
                print("  Loaded \(label): \(String(format: "%.3f", elapsed))s")

                let inputs = model.modelDescription.inputDescriptionsByName
                print("    Inputs:")
                for (iname, desc) in inputs.sorted(by: { $0.key < $1.key }) {
                    if let c = desc.multiArrayConstraint {
                        let shape = c.shape.map { $0.intValue }
                        print("      \(iname): \(shape) \(vocosTypeName(c.dataType))")
                    }
                }
                let outputs = model.modelDescription.outputDescriptionsByName
                print("    Outputs:")
                for (oname, desc) in outputs.sorted(by: { $0.key < $1.key }) {
                    if let c = desc.multiArrayConstraint {
                        let shape = c.shape.map { $0.intValue }
                        print("      \(oname): \(shape) \(vocosTypeName(c.dataType))")
                    }
                }
            } catch {
                let elapsed = CFAbsoluteTimeGetCurrent() - t0
                print("  FAILED \(label) after \(String(format: "%.3f", elapsed))s: \(error)")
            }
        }

        guard let primary = loadedModels.first else {
            print("\nERROR: No models loaded.")
            return
        }
        print()

        // -- Step 2: Phonemize test text --
        print("--- Step 2: Phonemize test text ---")

        let engine = try KokoroEngine(modelDirectory: modelDir)
        let testText = "The quick brown fox jumps over the lazy dog."

        let t1 = CFAbsoluteTimeGetCurrent()
        let (phonemes, tokenIds) = engine.phonemize(testText)
        let phonemizeTime = CFAbsoluteTimeGetCurrent() - t1

        print("  Text: \"\(testText)\"")
        print("  Phonemes: \(phonemes)")
        print("  Token IDs (\(tokenIds.count)): \(tokenIds.prefix(20))...")
        print("  Phonemize time: \(String(format: "%.2f", phonemizeTime * 1000))ms")
        print()

        // -- Step 3: Prepare MLMultiArray inputs --
        print("--- Step 3: Prepare model inputs ---")

        let model = primary.model
        let inputSpecs = model.modelDescription.inputDescriptionsByName

        var maxTokens = 124
        if let inputIdSpec = inputSpecs["input_ids"],
           let constraint = inputIdSpec.multiArrayConstraint
        {
            let shape = constraint.shape.map { $0.intValue }
            if shape.count >= 2 {
                maxTokens = shape[1]
            }
        }
        print("  Max tokens for model: \(maxTokens)")

        let realTokenCount = min(tokenIds.count, maxTokens)

        let inputIds = try MLMultiArray(shape: [1, maxTokens as NSNumber], dataType: .int32)
        let maskArray = try MLMultiArray(shape: [1, maxTokens as NSNumber], dataType: .int32)
        let idPtr = inputIds.dataPointer.assumingMemoryBound(to: Int32.self)
        let mskPtr = maskArray.dataPointer.assumingMemoryBound(to: Int32.self)
        for i in 0..<maxTokens {
            idPtr[i] = i < tokenIds.count ? Int32(tokenIds[i]) : 0
            mskPtr[i] = i < realTokenCount ? 1 : 0
        }

        // Load voice embedding directly from JSON
        let voiceURL = modelDir
            .appendingPathComponent("voices")
            .appendingPathComponent("\(voice).json")
        let voiceData = try Data(contentsOf: voiceURL)
        guard let voiceJSON = try JSONSerialization.jsonObject(with: voiceData) as? [String: Any],
              let embeddingArray = voiceJSON["embedding"] as? [Double]
        else {
            print("  ERROR: Could not load voice embedding from \(voiceURL.path)")
            return
        }
        let styleVector = embeddingArray.prefix(256).map { Float($0) }
        let refS = try MLMultiArray(shape: [1, 256], dataType: .float32)
        let refPtr = refS.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<min(styleVector.count, 256) {
            refPtr[i] = styleVector[i]
        }

        let needsPhases = inputSpecs["random_phases"] != nil
        var randomPhases: MLMultiArray?
        if needsPhases {
            let numPhases = 9
            randomPhases = try MLMultiArray(shape: [1, numPhases as NSNumber], dataType: .float32)
            let phasePtr = randomPhases!.dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0..<numPhases {
                phasePtr[i] = Float.random(in: 0..<(2 * .pi))
            }
        }

        print("  input_ids: [1, \(maxTokens)] int32 (real tokens: \(realTokenCount))")
        print("  attention_mask: [1, \(maxTokens)] int32")
        print("  ref_s: [1, 256] float32")
        if needsPhases { print("  random_phases: [1, 9] float32") }
        print()

        // -- Step 4: Run inference --
        print("--- Step 4: Run inference ---")

        var inputDict: [String: MLFeatureValue] = [
            "input_ids": MLFeatureValue(multiArray: inputIds),
            "attention_mask": MLFeatureValue(multiArray: maskArray),
            "ref_s": MLFeatureValue(multiArray: refS),
        ]
        if let phases = randomPhases {
            inputDict["random_phases"] = MLFeatureValue(multiArray: phases)
        }

        for (name, _) in inputSpecs where !inputDict.keys.contains(name) {
            print("  WARNING: Model expects input '\(name)' which is not provided")
        }

        let input = try MLDictionaryFeatureProvider(dictionary: inputDict)

        print("  Running 2 warmup iterations...")
        for _ in 0..<2 {
            if let phases = randomPhases {
                let phasePtr = phases.dataPointer.assumingMemoryBound(to: Float.self)
                for i in 0..<9 { phasePtr[i] = Float.random(in: 0..<(2 * .pi)) }
            }
            _ = try await model.prediction(from: input)
        }
        print("  Warmup done.")

        var inferTimes: [Double] = []
        var lastOutput: MLFeatureProvider?

        print("  Running \(iterations) benchmark iterations...")
        for i in 0..<iterations {
            if let phases = randomPhases {
                let phasePtr = phases.dataPointer.assumingMemoryBound(to: Float.self)
                for j in 0..<9 { phasePtr[j] = Float.random(in: 0..<(2 * .pi)) }
            }

            let t = CFAbsoluteTimeGetCurrent()
            let out = try await model.prediction(from: input)
            let elapsed = CFAbsoluteTimeGetCurrent() - t
            inferTimes.append(elapsed)
            lastOutput = out

            print("    iter \(i + 1): \(String(format: "%.1f", elapsed * 1000))ms")
        }

        print()

        // -- Step 5: Extract output --
        print("--- Step 5: Extract and validate output ---")

        guard let finalOutput = lastOutput else {
            print("  ERROR: No output from inference")
            return
        }

        let outputNames = model.modelDescription.outputDescriptionsByName.keys.sorted()
        print("  Output feature names: \(outputNames)")

        var samples: [Float] = []
        if let audioArray = finalOutput.featureValue(for: "audio")?.multiArrayValue {
            var validCount = audioArray.count
            if let lengthArray = finalOutput.featureValue(for: "audio_length_samples")?.multiArrayValue {
                validCount = lengthArray[0].intValue
                print("  audio_length_samples: \(validCount)")
            }
            samples = extractFloatsVocos(from: audioArray, maxCount: validCount)
            print("  Audio array shape: \(audioArray.shape.map { $0.intValue })")
            print("  Valid samples: \(samples.count)")
        } else if let waveArray = finalOutput.featureValue(for: "waveform")?.multiArrayValue {
            samples = extractFloatsVocos(from: waveArray)
            print("  Waveform array shape: \(waveArray.shape.map { $0.intValue })")
            print("  Samples: \(samples.count)")
        } else {
            print("  ERROR: No 'audio' or 'waveform' output found")
            for name in outputNames {
                if let arr = finalOutput.featureValue(for: name)?.multiArrayValue {
                    print("    \(name): shape=\(arr.shape.map { $0.intValue })")
                }
            }
            return
        }

        if let predDur = finalOutput.featureValue(for: "pred_dur")?.multiArrayValue {
            let durCount = min(predDur.count, realTokenCount)
            var totalFrames: Float = 0
            for i in 0..<durCount {
                totalFrames += predDur[i].floatValue
            }
            print("  pred_dur: \(durCount) values, total frames: \(String(format: "%.0f", totalFrames))")
        }

        let audioDuration = Double(samples.count) / Double(sampleRate)
        print("  Audio duration: \(String(format: "%.2f", audioDuration))s")

        var rms: Float = 0
        for s in samples { rms += s * s }
        rms = sqrt(rms / Float(max(samples.count, 1)))
        print("  RMS: \(String(format: "%.6f", rms))")
        print("  RMS > 0.001: \(rms > 0.001 ? "PASS" : "FAIL")")
        print()

        // -- Step 6: Write WAV --
        if !noAudio {
            print("--- Step 6: Write WAV output ---")
            let wavPath = "/tmp/kokoro_vocos_test.wav"
            try E2EValidation().writeWAV(samples: samples, sampleRate: sampleRate, to: wavPath)
            print("  Written: \(wavPath) (\(String(format: "%.2f", audioDuration))s)")
            print()
        }

        // -- Step 7: Compute RTFx --
        print("--- Step 7: Performance summary ---")

        let sortedTimes = inferTimes.sorted()
        let medianTime = sortedTimes[sortedTimes.count / 2]
        let meanTime = inferTimes.reduce(0, +) / Double(inferTimes.count)
        let minTime = sortedTimes.first!
        let maxTime = sortedTimes.last!

        let medianRTFx = audioDuration / medianTime
        let meanRTFx = audioDuration / meanTime
        let bestRTFx = audioDuration / minTime

        print("  Model: \(primary.label)")
        print("  Compute units: cpuAndNeuralEngine")
        print("  Audio duration: \(String(format: "%.2f", audioDuration))s")
        print()
        print("  Inference timing (\(iterations) iterations):")
        print("    Median: \(String(format: "%.1f", medianTime * 1000))ms -> \(String(format: "%.1f", medianRTFx))x RTFx")
        print("    Mean:   \(String(format: "%.1f", meanTime * 1000))ms -> \(String(format: "%.1f", meanRTFx))x RTFx")
        print("    Min:    \(String(format: "%.1f", minTime * 1000))ms -> \(String(format: "%.1f", bestRTFx))x RTFx")
        print("    Max:    \(String(format: "%.1f", maxTime * 1000))ms")
        print()

        // -- Step 8: Compare with KokoroEngine --
        print("--- Step 8: Compare with KokoroEngine ---")

        _ = try engine.synthesize(text: "warmup", voice: voice)
        _ = try engine.synthesize(text: "warmup", voice: voice)

        var engineTimes: [Double] = []
        var engineResult: SynthesisResult?

        for i in 0..<iterations {
            let t = CFAbsoluteTimeGetCurrent()
            let result = try engine.synthesize(text: testText, voice: voice)
            let elapsed = CFAbsoluteTimeGetCurrent() - t
            engineTimes.append(elapsed)
            engineResult = result
            print("    iter \(i + 1): \(String(format: "%.1f", elapsed * 1000))ms")
        }

        if let result = engineResult {
            let eSorted = engineTimes.sorted()
            let eMedian = eSorted[eSorted.count / 2]
            let eMean = engineTimes.reduce(0, +) / Double(engineTimes.count)
            let eAudioDur = result.duration
            let eMedianRTFx = eAudioDur / eMedian
            let eMeanRTFx = eAudioDur / eMean

            print()
            print("  KokoroEngine (unified, cpuAndGPU):")
            print("    Audio duration: \(String(format: "%.2f", eAudioDur))s")
            print("    Median: \(String(format: "%.1f", eMedian * 1000))ms -> \(String(format: "%.1f", eMedianRTFx))x RTFx")
            print("    Mean:   \(String(format: "%.1f", eMean * 1000))ms -> \(String(format: "%.1f", eMeanRTFx))x RTFx")

            print()
            print(String(repeating: "=", count: 80))
            print("RESULT SUMMARY")
            print(String(repeating: "=", count: 80))
            print()
            print("  Direct ANE inference (cpuAndNeuralEngine):")
            print("    Median RTFx: \(String(format: "%.1f", medianRTFx))x")
            print("    Target (>10x): \(medianRTFx > 10 ? "PASS" : "FAIL")")
            print()
            print("  KokoroEngine comparison:")
            print("    Median RTFx: \(String(format: "%.1f", eMedianRTFx))x")
            let speedup = medianRTFx / eMedianRTFx
            print("    ANE speedup: \(String(format: "%.2f", speedup))x \(speedup > 1 ? "faster" : "slower")")
        }

        print()
        print("  Audio quality:")
        print("    RMS: \(String(format: "%.6f", rms))")
        print("    RMS > 0.001: \(rms > 0.001 ? "PASS (audible speech)" : "FAIL (silence/garbage)")")
        print()
        print("  NOTE: FluidInference unified models include iSTFTNet vocoder internally.")
        print("  A separate Vocos CoreML model cannot be created because Vocos uses")
        print("  complex-valued iSTFT operations (1j) which coremltools does not support.")
        print("  The HAR (Harmonic-phase) architecture used by mattmireles/kokoro-coreml")
        print("  works around this by separating harmonic and phase components.")
        print()

        fflush(stdout)
    }
}

private func vocosTypeName(_ type: MLMultiArrayDataType) -> String {
    switch type {
    case .float16: return "Float16"
    case .float32: return "Float32"
    case .float64: return "Float64"
    case .int32: return "Int32"
    case .double: return "Float64"
    @unknown default: return "Unknown(\(type.rawValue))"
    }
}

/// Extract float samples from an MLMultiArray, handling both float16 and float32.
private func extractFloatsVocos(from array: MLMultiArray, maxCount: Int? = nil) -> [Float] {
    let count = min(array.count, max(0, maxCount ?? array.count))
    guard count > 0 else { return [] }
    var samples = [Float](repeating: 0, count: count)
    if array.dataType == .float16 {
        let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
        for i in 0..<count { samples[i] = Float(ptr[i]) }
    } else {
        let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
        samples.withUnsafeMutableBufferPointer { dst in
            dst.baseAddress!.update(from: ptr, count: count)
        }
    }
    return samples
}

// Helper: String repetition operator
private func * (lhs: String, rhs: Int) -> String {
    String(repeating: lhs, count: rhs)
}

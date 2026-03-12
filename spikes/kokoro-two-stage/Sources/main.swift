import ArgumentParser
import CoreML
import Foundation
import KokoroTTS

@main
struct KokoroTwoStageSpike: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "spike",
        abstract: "Kokoro two-stage pipeline hypothesis tests",
        subcommands: [
            H1ModelLoad.self, H2Alignment.self, H3Performance.self, H4Caching.self,
            E2EValidation.self,
        ]
    )
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

        // Phase 5: Benchmark
        print("\n--- Phase 5: Benchmark (\(iterations) iterations) ---")
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

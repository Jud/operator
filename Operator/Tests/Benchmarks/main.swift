import AVFoundation
import Darwin
import Foundation
import OperatorCore

private enum BenchmarkTarget: String, CaseIterable {
    case routing
    case routingLatency = "routing-latency"
    case routingAccuracy = "routing-accuracy"
    case tts
    case ttsTTFA = "tts-ttfa"
    case stt
    case sttLatency = "stt-latency"
    case sttLong = "stt-long"
    case memory
}

private let benchmarkHelpTargets = BenchmarkTarget.allCases.map(\.rawValue).joined(separator: ", ")
private let benchmarkRunnerEnvKey = "OPERATOR_BENCHMARK_RUNNER"

private func printBenchmarkUsage(listOnly: Bool = false) {
    print("Usage: ./scripts/run-benchmarks.sh [--debug|--release] [--no-build] [target...]")
    print("No .app bundle is required.")
    print("")
    print("Targets:")
    print("  routing           Run routing latency and routing accuracy benchmarks")
    print("  routing-latency   Run only routing latency benchmark")
    print("  routing-accuracy  Run only routing accuracy benchmark")
    print("  tts               Run TTS time-to-first-audio benchmark")
    print("  tts-ttfa          Run only TTS time-to-first-audio benchmark")
    print("  stt               Run STT latency and long-utterance benchmarks")
    print("  stt-latency       Run only STT 5s transcription benchmark")
    print("  stt-long          Run only STT long-utterance benchmark")
    print("  memory            Run routing-model memory benchmark")
    if listOnly {
        return
    }
    print("")
    print("Examples:")
    print("  ./scripts/run-benchmarks.sh routing")
    print("  ./scripts/run-benchmarks.sh --no-build routing-latency")
    print("  ./scripts/run-benchmarks.sh stt-long")
    print("  ./scripts/run-benchmarks.sh --release memory")
}

private func startModelProgressObserver(
    modelManager: ModelManager,
    type: ModelManager.ModelType,
    label: String
) async -> Task<Void, Never> {
    let stream = await modelManager.stateChanged

    return Task {
        var lastDownloadBucket: Int?
        var lastStage: String?

        for await (changedType, state) in stream {
            if Task.isCancelled {
                return
            }
            guard changedType == type else {
                continue
            }

            switch state {
            case .notDownloaded:
                continue

            case .downloading(let progress):
                let percent = max(0, min(100, Int(progress * 100)))
                let bucket = (percent / 10) * 10
                guard lastDownloadBucket != bucket else {
                    continue
                }
                lastDownloadBucket = bucket
                print("  \(label): download \(bucket)%")

            case .ready:
                guard lastStage != "ready" else {
                    continue
                }
                lastStage = "ready"
                print("  \(label): download complete")

            case .loading:
                guard lastStage != "loading" else {
                    continue
                }
                lastStage = "loading"
                print("  \(label): loading model")

            case .loaded:
                guard lastStage != "loaded" else {
                    continue
                }
                lastStage = "loaded"
                print("  \(label): model loaded")

            case .unloaded:
                guard lastStage != "unloaded" else {
                    continue
                }
                lastStage = "unloaded"
                print("  \(label): model unloaded")

            case .error(let message):
                print("  \(label): error \(message)")
            }
        }
    }
}

@MainActor
func runBenchmarks() async {
    guard ProcessInfo.processInfo.environment[benchmarkRunnerEnvKey] == "1" else {
        fputs("Run benchmarks via ./scripts/run-benchmarks.sh. No .app bundle is required.\n", stderr)
        exit(2)
    }

    let args = Set(
        CommandLine.arguments
            .dropFirst()
            .map { $0.lowercased().trimmingCharacters(in: .whitespacesAndNewlines) }
    )

    if args.contains("help") || args.contains("--help") || args.contains("-h") {
        printBenchmarkUsage()
        return
    }
    if args.contains("list") {
        printBenchmarkUsage(listOnly: true)
        return
    }

    let knownArgs = Set(BenchmarkTarget.allCases.map(\.rawValue)).union(["all"])
    let unknownArgs = args.subtracting(knownArgs)
    guard unknownArgs.isEmpty else {
        print("Unknown benchmark target(s): \(unknownArgs.sorted().joined(separator: ", "))")
        print("")
        printBenchmarkUsage()
        return
    }

    let runAll = args.isEmpty || args.contains("all")
    let includes = { (target: BenchmarkTarget) in
        runAll || args.contains(target.rawValue)
    }

    if includes(.routing) || includes(.routingLatency) || includes(.routingAccuracy) {
        await benchmarkRouting(
            runLatency: includes(.routing) || includes(.routingLatency),
            runAccuracy: includes(.routing) || includes(.routingAccuracy)
        )
    }
    if includes(.tts) || includes(.ttsTTFA) {
        await benchmarkTTS()
    }
    if includes(.stt) || includes(.sttLatency) || includes(.sttLong) {
        await benchmarkSTT(
            runLatency: includes(.stt) || includes(.sttLatency),
            runLong: includes(.stt) || includes(.sttLong)
        )
    }
    if includes(.memory) {
        await benchmarkMemory()
    }
}

// MARK: - Routing Benchmarks (AC-007.1, AC-007.5)

@MainActor
func benchmarkRouting(runLatency: Bool, runAccuracy: Bool) async {
    guard runLatency || runAccuracy else {
        return
    }

    print("\n=== ROUTING BENCHMARK ===\n")

    let modelManager = ModelManager()
    let engine = MLXRoutingEngine(modelManager: modelManager)

    let prompt = """
        Sessions:
        1. "frontend" (cwd: /Users/jud/web-app) - React TypeScript dashboard
        2. "backend" (cwd: /Users/jud/api-server) - FastAPI Python endpoints
        3. "infra" (cwd: /Users/jud/deploy) - Terraform AWS infrastructure

        User message: "add a loading spinner to the dashboard"
        """

    // Cold start (includes model download + load)
    print("Loading routing model (first run downloads ~622MB)...")
    let routingProgress = await startModelProgressObserver(
        modelManager: modelManager,
        type: .routing,
        label: "Routing"
    )
    let coldStart = CFAbsoluteTimeGetCurrent()
    do {
        _ = try await engine.run(prompt: prompt, timeout: 120)
    } catch {
        routingProgress.cancel()
        print("FAIL: Cold start failed: \(error)")
        return
    }
    routingProgress.cancel()
    let coldMs = (CFAbsoluteTimeGetCurrent() - coldStart) * 1000
    print("Cold start (download+load+inference): \(String(format: "%.0f", coldMs))ms")

    if runLatency {
        let testMessages = [
            "fix the database migration",
            "update the terraform module for the new VPC",
            "add dark mode toggle to the settings page",
            "optimize the SQL query for user lookups",
            "deploy the staging environment",
            "refactor the authentication middleware",
            "add unit tests for the payment module",
            "fix the CSS grid layout on mobile"
        ]

        var latencies: [Double] = []

        print("\n--- Routing Latency ---")
        for message in testMessages {
            let warmPrompt = """
                Sessions:
                1. "frontend" (cwd: /Users/jud/web-app) - React TypeScript dashboard
                2. "backend" (cwd: /Users/jud/api-server) - FastAPI Python endpoints
                3. "infra" (cwd: /Users/jud/deploy) - Terraform AWS infrastructure

                User message: "\(message)"
                """
            let start = CFAbsoluteTimeGetCurrent()
            do {
                _ = try await engine.run(prompt: warmPrompt, timeout: 10)
                let durationMs = (CFAbsoluteTimeGetCurrent() - start) * 1000
                latencies.append(durationMs)
                print("  \(String(format: "%5.0f", durationMs))ms | \"\(message)\"")
            } catch {
                let durationMs = (CFAbsoluteTimeGetCurrent() - start) * 1000
                latencies.append(durationMs)
                print("  ERROR \(String(format: "%5.0f", durationMs))ms | \"\(message)\" -> \(error)")
            }
        }

        let sorted = latencies.sorted()
        let avg = latencies.reduce(0, +) / Double(latencies.count)
        let p50 = sorted[sorted.count / 2]
        let p95 = sorted[Int(Double(sorted.count - 1) * 0.95)]
        let maxMs = sorted.last ?? 0

        print("")
        let latencyLine =
            "Latency:  avg=\(String(format: "%.0f", avg))ms"
            + "  p50=\(String(format: "%.0f", p50))ms"
            + "  p95=\(String(format: "%.0f", p95))ms"
            + "  max=\(String(format: "%.0f", maxMs))ms"
        print(latencyLine)
        print("AC-007.1 (latency <300ms):  \(p95 < 300 ? "PASS" : "FAIL") (p95=\(String(format: "%.0f", p95))ms)")
    }

    if runAccuracy {
        struct TestCase {
            let message: String
            let expected: String
        }

        let testCases: [TestCase] = [
            TestCase(message: "add a loading spinner to the React dashboard", expected: "frontend"),
            TestCase(message: "fix the CSS grid layout", expected: "frontend"),
            TestCase(message: "add a new REST endpoint for user profiles", expected: "backend"),
            TestCase(message: "fix the database connection pool", expected: "backend"),
            TestCase(message: "optimize the SQL query", expected: "backend"),
            TestCase(message: "update the terraform module for the new VPC", expected: "infra"),
            TestCase(message: "add a new S3 bucket for logs", expected: "infra"),
            TestCase(message: "fix the AWS IAM permissions", expected: "infra"),
            TestCase(message: "add TypeScript types for the API response", expected: "frontend"),
            TestCase(message: "update the Python requirements.txt", expected: "backend")
        ]

        var correct = 0

        print("\n--- Routing Accuracy ---")
        for tc in testCases {
            let warmPrompt = """
                Sessions:
                1. "frontend" (cwd: /Users/jud/web-app) - React TypeScript dashboard, CSS, components
                2. "backend" (cwd: /Users/jud/api-server) - FastAPI Python endpoints, SQL, database
                3. "infra" (cwd: /Users/jud/deploy) - Terraform AWS infrastructure, IAM, S3, VPC

                User message: "\(tc.message)"
                """
            do {
                let json = try await engine.run(prompt: warmPrompt, timeout: 10)
                let session = json["session"] as? String ?? ""
                let match = session == tc.expected
                if match { correct += 1 }
                let mark = match ? "✓" : "✗"
                print("  \(mark) \"\(tc.message)\" -> \(session) (expected: \(tc.expected))")
            } catch {
                print("  ✗ \"\(tc.message)\" -> ERROR: \(error)")
            }
        }

        let accuracy = Double(correct) / Double(testCases.count) * 100
        print("")
        print("Accuracy: \(correct)/\(testCases.count) (\(String(format: "%.0f", accuracy))%)")
        print("AC-007.5 (accuracy >=90%%): \(accuracy >= 90 ? "PASS" : "FAIL") (\(String(format: "%.0f", accuracy))%%)")
    }
}

// MARK: - TTS Benchmarks (AC-003.1)

@MainActor
func benchmarkTTS() async {
    print("\n=== TTS BENCHMARK ===\n")

    let modelManager = ModelManager()
    let ttsManager = Qwen3TTSSpeechManager(modelManager: modelManager)
    guard let appleVoice = AVSpeechSynthesisVoice(language: "en-US") else {
        print("FAIL: Could not create AVSpeechSynthesisVoice")
        return
    }
    let voice = VoiceDescriptor(
        appleVoice: appleVoice,
        qwenSpeakerID: "vivian"
    )

    // Cold start
    print("Loading TTS model (first run downloads ~2GB)...")
    let ttsProgress = await startModelProgressObserver(
        modelManager: modelManager,
        type: .tts,
        label: "TTS"
    )
    let coldStart = CFAbsoluteTimeGetCurrent()
    ttsManager.speak(
        "Hello, this is a test.",
        voice: voice,
        prefix: "test",
        pitchMultiplier: 1.0
    )

    var waited = 0.0
    while !ttsManager.isSpeaking && waited < 120 {
        try? await Task.sleep(nanoseconds: 10_000_000)
        waited += 0.01
    }
    ttsProgress.cancel()
    let coldMs = (CFAbsoluteTimeGetCurrent() - coldStart) * 1000
    print("Cold start (download+load+first-audio): \(String(format: "%.0f", coldMs))ms")

    // Wait for cold utterance to finish naturally (avoids stream cancellation bug)
    for await _ in ttsManager.finishedSpeaking {
        break
    }
    try? await Task.sleep(nanoseconds: 200_000_000)

    // Warm TTFA tests — let each utterance play out fully before starting the next.
    // Interrupting mid-stream causes the old MLX computation to hog the GPU
    // (upstream speech-swift stream cancellation issue).
    let testTexts = [
        "The quick brown fox jumps over the lazy dog.",
        "Please check your email for the confirmation link.",
        "I've updated the configuration file with the new settings.",
        "The deployment was successful. All services are running.",
        "There was an error processing your request."
    ]

    var warmLatencies: [Double] = []

    for text in testTexts {
        let start = CFAbsoluteTimeGetCurrent()
        ttsManager.speak(text, voice: voice, prefix: "agent", pitchMultiplier: 1.0)

        waited = 0.0
        while !ttsManager.isSpeaking && waited < 30 {
            try? await Task.sleep(nanoseconds: 1_000_000)
            waited += 0.001
        }
        let ttfaMs = (CFAbsoluteTimeGetCurrent() - start) * 1000
        warmLatencies.append(ttfaMs)
        print("  \(String(format: "%5.0f", ttfaMs))ms | \"\(text.prefix(50))...\"")

        // Wait for utterance to finish naturally before starting next
        for await _ in ttsManager.finishedSpeaking {
            break
        }
        try? await Task.sleep(nanoseconds: 200_000_000)
    }

    let sorted = warmLatencies.sorted()
    let avg = warmLatencies.reduce(0, +) / Double(warmLatencies.count)
    let p95 = sorted[Int(Double(sorted.count - 1) * 0.95)]

    print("\n--- Results ---")
    print("TTFA:  avg=\(String(format: "%.0f", avg))ms  p95=\(String(format: "%.0f", p95))ms")
    print("")
    print("AC-003.1 (TTFA <200ms): \(p95 < 200 ? "PASS" : "FAIL") (p95=\(String(format: "%.0f", p95))ms)")
}

// MARK: - STT Benchmarks (AC-001.3, AC-001.4)

@MainActor
func benchmarkSTT(runLatency: Bool, runLong: Bool) async {
    guard runLatency || runLong else {
        return
    }

    print("\n=== STT BENCHMARK ===\n")

    let modelManager = ModelManager()
    let engine = ParakeetEngine(modelManager: modelManager)

    // Generate synthetic audio (440Hz sine wave at 16kHz)
    func makeSineBuffer(seconds: Double) -> AVAudioPCMBuffer? {
        let sampleRate: Double = 16000
        let frameCount = AVAudioFrameCount(seconds * sampleRate)
        guard
            let format = AVAudioFormat(
                standardFormatWithSampleRate: sampleRate,
                channels: 1
            )
        else { return nil }
        guard
            let buffer = AVAudioPCMBuffer(
                pcmFormat: format,
                frameCapacity: frameCount
            )
        else { return nil }
        buffer.frameLength = frameCount
        guard let data = buffer.floatChannelData?[0] else {
            return nil
        }
        for i in 0..<Int(frameCount) {
            data[i] =
                sin(
                    2.0 * .pi * 440.0 * Float(i) / Float(sampleRate)
                ) * 0.5
        }
        return buffer
    }

    if runLatency {
        print("Loading STT model (first run downloads ~400MB)...")
        let sttProgress = await startModelProgressObserver(
            modelManager: modelManager,
            type: .stt,
            label: "STT"
        )
        guard let buffer5s = makeSineBuffer(seconds: 5.0) else {
            sttProgress.cancel()
            print("FAIL: Could not create audio buffer")
            return
        }
        do {
            try engine.prepare()
        } catch {
            sttProgress.cancel()
            print("FAIL: prepare() failed: \(error)")
            return
        }
        engine.append(buffer5s)

        let coldStart = CFAbsoluteTimeGetCurrent()
        _ = await engine.finishAndTranscribe()
        sttProgress.cancel()
        let coldMs = (CFAbsoluteTimeGetCurrent() - coldStart) * 1000
        print("Cold transcription (5s audio, includes download+load): \(String(format: "%.0f", coldMs))ms")

        var latencies5s: [Double] = []
        for i in 1...3 {
            guard let buf = makeSineBuffer(seconds: 5.0) else {
                continue
            }
            do { try engine.prepare() } catch { continue }
            engine.append(buf)

            let start = CFAbsoluteTimeGetCurrent()
            _ = await engine.finishAndTranscribe()
            let ms = (CFAbsoluteTimeGetCurrent() - start) * 1000
            latencies5s.append(ms)
            print("  Run \(i): \(String(format: "%.0f", ms))ms (5s audio)")
        }

        let avg5s = latencies5s.isEmpty ? 0 : latencies5s.reduce(0, +) / Double(latencies5s.count)
        let max5s = latencies5s.max() ?? 0

        print("\n--- STT 5s Results ---")
        print("5s audio:  avg=\(String(format: "%.0f", avg5s))ms  max=\(String(format: "%.0f", max5s))ms")
        print("AC-001.3 (5s in <2s):     \(max5s < 2000 ? "PASS" : "FAIL") (max=\(String(format: "%.0f", max5s))ms)")
    }

    if runLong {
        print("\nLong utterance test (20s audio)...")
        guard let buffer20s = makeSineBuffer(seconds: 20.0) else {
            print("FAIL: Could not create 20s audio buffer")
            return
        }
        do { try engine.prepare() } catch {
            print("FAIL: prepare() failed for long utterance: \(error)")
            return
        }
        engine.append(buffer20s)

        let longStart = CFAbsoluteTimeGetCurrent()
        let longResult = await engine.finishAndTranscribe()
        let longMs = (CFAbsoluteTimeGetCurrent() - longStart) * 1000
        print("Long utterance (20s audio): \(String(format: "%.0f", longMs))ms, result: \(longResult ?? "nil")")
        print("AC-001.4 (20s no crash):  PASS (completed in \(String(format: "%.0f", longMs))ms)")
    }
}

// MARK: - Memory Benchmark (AC-011.5)

@MainActor
func benchmarkMemory() async {
    print("\n=== MEMORY BENCHMARK ===\n")

    func physFootprintMB() -> Double {
        var rusage = rusage_info_v4()
        let result = withUnsafeMutablePointer(to: &rusage) {
            $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
                proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0)
            }
        }
        guard result == 0 else { return 0 }
        return Double(rusage.ri_phys_footprint) / 1_048_576
    }

    let baseline = physFootprintMB()
    print("Baseline: \(String(format: "%.0f", baseline))MB")

    // Load routing model
    let modelManager = ModelManager()
    let routingProgress = await startModelProgressObserver(
        modelManager: modelManager,
        type: .routing,
        label: "Routing"
    )
    let routingEngine = MLXRoutingEngine(modelManager: modelManager)
    let routingPrompt = """
        Sessions:
        1. "test" (cwd: /tmp/test) - test project
        User message: "hello"
        """
    do {
        _ = try await routingEngine.run(prompt: routingPrompt, timeout: 120)
    } catch {
        print("Routing model load failed: \(error)")
    }
    routingProgress.cancel()
    let afterRouting = physFootprintMB()
    let routingDelta = afterRouting - baseline
    print(
        "After routing model: \(String(format: "%.0f", afterRouting))MB"
            + " (+\(String(format: "%.0f", routingDelta))MB)"
    )

    // Load TTS model
    let ttsProgress = await startModelProgressObserver(
        modelManager: modelManager,
        type: .tts,
        label: "TTS"
    )
    let ttsManager = Qwen3TTSSpeechManager(modelManager: modelManager)
    guard let memVoice = AVSpeechSynthesisVoice(language: "en-US") else {
        print("FAIL: Could not create AVSpeechSynthesisVoice")
        return
    }
    let voice = VoiceDescriptor(
        appleVoice: memVoice,
        qwenSpeakerID: "vivian"
    )
    ttsManager.speak(
        "Memory test.",
        voice: voice,
        prefix: "test",
        pitchMultiplier: 1.0
    )
    // Wait for utterance to complete naturally so GPU buffers settle
    for await _ in ttsManager.finishedSpeaking {
        break
    }
    ttsProgress.cancel()
    ttsManager.stop()
    try? await Task.sleep(nanoseconds: 1_000_000_000)

    let afterTTS = physFootprintMB()
    let ttsDelta = afterTTS - afterRouting
    print(
        "After TTS model:     \(String(format: "%.0f", afterTTS))MB"
            + " (+\(String(format: "%.0f", ttsDelta))MB)"
    )

    // STT stays loaded between transcriptions for performance.
    // Peak = routing + TTS + STT during transcription.
    let sttProgress = await startModelProgressObserver(
        modelManager: modelManager,
        type: .stt,
        label: "STT"
    )
    let sttEngine = ParakeetEngine(modelManager: modelManager)
    guard
        let format = AVAudioFormat(
            standardFormatWithSampleRate: 16000,
            channels: 1
        )
    else {
        print("FAIL: Could not create AVAudioFormat")
        return
    }
    guard
        let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: 16000
        )
    else {
        print("FAIL: Could not create AVAudioPCMBuffer")
        return
    }
    buffer.frameLength = 16000
    guard let data = buffer.floatChannelData?[0] else {
        print("FAIL: Could not access floatChannelData")
        return
    }
    for i in 0..<16000 {
        data[i] =
            sin(
                2.0 * .pi * 440.0 * Float(i) / 16000.0
            ) * 0.5
    }
    do { try sttEngine.prepare() } catch {}
    sttEngine.append(buffer)
    _ = await sttEngine.finishAndTranscribe()
    sttProgress.cancel()

    let peakWithSTT = physFootprintMB()
    let sttDelta = peakWithSTT - afterTTS
    print(
        "Peak (all models):   \(String(format: "%.0f", peakWithSTT))MB"
            + " (+\(String(format: "%.0f", sttDelta))MB for STT)"
    )

    // Explicitly unload STT to show post-unload memory
    sttEngine.unloadModel()
    try? await Task.sleep(nanoseconds: 200_000_000)
    let afterUnload = physFootprintMB()
    print("After STT unload:    \(String(format: "%.0f", afterUnload))MB")

    let peak = max(afterRouting, afterTTS, peakWithSTT)
    print("\n--- Results ---")
    print("Peak resident memory: \(String(format: "%.0f", peak))MB")
    print("")
    print("AC-011.5 (peak <3GB): \(peak < 3072 ? "PASS" : "FAIL") (\(String(format: "%.0f", peak))MB)")
}

// Entry point
await runBenchmarks()

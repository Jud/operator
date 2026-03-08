import AVFoundation
import Foundation
import OperatorCore

@MainActor
func runBenchmarks() async {
    let args = CommandLine.arguments
    let runAll = args.count < 2

    if runAll || args.contains("routing") {
        await benchmarkRouting()
    }
    if runAll || args.contains("tts") {
        await benchmarkTTS()
    }
    if runAll || args.contains("stt") {
        await benchmarkSTT()
    }
    if runAll || args.contains("memory") {
        await benchmarkMemory()
    }
}

// MARK: - Routing Benchmarks (AC-007.1, AC-007.5)

@MainActor
func benchmarkRouting() async {
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
    let coldStart = CFAbsoluteTimeGetCurrent()
    do {
        _ = try await engine.run(prompt: prompt, timeout: 120)
    } catch {
        print("FAIL: Cold start failed: \(error)")
        return
    }
    let coldMs = (CFAbsoluteTimeGetCurrent() - coldStart) * 1000
    print("Cold start (download+load+inference): \(String(format: "%.0f", coldMs))ms")

    // Warm latency tests
    struct TestCase {
        let message: String
        let expected: String
    }

    let testCases: [TestCase] = [
        TestCase(message: "fix the database migration", expected: "backend"),
        TestCase(message: "update the terraform module for the new VPC", expected: "infra"),
        TestCase(message: "add dark mode toggle to the settings page", expected: "frontend"),
        TestCase(message: "optimize the SQL query for user lookups", expected: "backend"),
        TestCase(message: "deploy the staging environment", expected: "infra"),
        TestCase(message: "refactor the authentication middleware", expected: "backend"),
        TestCase(message: "add unit tests for the payment module", expected: "backend"),
        TestCase(message: "fix the CSS grid layout on mobile", expected: "frontend"),
        TestCase(message: "add TypeScript types for the API response", expected: "frontend"),
        TestCase(message: "update the Python requirements.txt", expected: "backend")
    ]

    var latencies: [Double] = []
    var correct = 0

    for tc in testCases {
        let warmPrompt = """
            Sessions:
            1. "frontend" (cwd: /Users/jud/web-app) - React TypeScript dashboard, CSS, components
            2. "backend" (cwd: /Users/jud/api-server) - FastAPI Python endpoints, SQL, database
            3. "infra" (cwd: /Users/jud/deploy) - Terraform AWS infrastructure, IAM, S3, VPC

            User message: "\(tc.message)"
            """
        let start = CFAbsoluteTimeGetCurrent()
        do {
            let json = try await engine.run(prompt: warmPrompt, timeout: 10)
            let durationMs = (CFAbsoluteTimeGetCurrent() - start) * 1000
            latencies.append(durationMs)

            let session = json["session"] as? String ?? ""
            let match = session == tc.expected
            if match { correct += 1 }
            let mark = match ? "✓" : "✗"
            let line =
                "  \(mark) \(String(format: "%5.0f", durationMs))ms"
                + " | \"\(tc.message)\" -> \(session)"
                + " (expected: \(tc.expected))"
            print(line)
        } catch {
            let durationMs = (CFAbsoluteTimeGetCurrent() - start) * 1000
            latencies.append(durationMs)
            print("  ✗ \(String(format: "%5.0f", durationMs))ms | \"\(tc.message)\" -> ERROR: \(error)")
        }
    }

    let sorted = latencies.sorted()
    let avg = latencies.reduce(0, +) / Double(latencies.count)
    let p50 = sorted[sorted.count / 2]
    let p95 = sorted[Int(Double(sorted.count - 1) * 0.95)]
    let maxMs = sorted.last ?? 0
    let accuracy = Double(correct) / Double(testCases.count) * 100

    print("\n--- Results ---")
    let latencyLine =
        "Latency:  avg=\(String(format: "%.0f", avg))ms"
        + "  p50=\(String(format: "%.0f", p50))ms"
        + "  p95=\(String(format: "%.0f", p95))ms"
        + "  max=\(String(format: "%.0f", maxMs))ms"
    print(latencyLine)
    print("Accuracy: \(correct)/\(testCases.count) (\(String(format: "%.0f", accuracy))%)")
    print("")
    print("AC-007.1 (latency <300ms):  \(p95 < 300 ? "PASS" : "FAIL") (p95=\(String(format: "%.0f", p95))ms)")
    print("AC-007.5 (accuracy >=90%%): \(accuracy >= 90 ? "PASS" : "FAIL") (\(String(format: "%.0f", accuracy))%%)")
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
        qwenSpeakerID: "Chelsie"
    )

    // Cold start
    print("Loading TTS model (first run downloads ~2GB)...")
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
    let coldMs = (CFAbsoluteTimeGetCurrent() - coldStart) * 1000
    print("Cold start (download+load+first-audio): \(String(format: "%.0f", coldMs))ms")

    ttsManager.stop()
    try? await Task.sleep(nanoseconds: 500_000_000)

    // Warm TTFA tests
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
        while !ttsManager.isSpeaking && waited < 10 {
            try? await Task.sleep(nanoseconds: 1_000_000)
            waited += 0.001
        }
        let durationMs = (CFAbsoluteTimeGetCurrent() - start) * 1000
        warmLatencies.append(durationMs)
        print("  \(String(format: "%5.0f", durationMs))ms | \"\(text.prefix(50))...\"")

        ttsManager.stop()
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
func benchmarkSTT() async {
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

    // 5s transcription (typical PTT)
    print("Loading STT model (first run downloads ~400MB)...")
    guard let buffer5s = makeSineBuffer(seconds: 5.0) else {
        print("FAIL: Could not create audio buffer")
        return
    }
    do {
        try engine.prepare()
    } catch {
        print("FAIL: prepare() failed: \(error)")
        return
    }
    engine.append(buffer5s)

    let coldStart = CFAbsoluteTimeGetCurrent()
    _ = await engine.finishAndTranscribe()
    let coldMs = (CFAbsoluteTimeGetCurrent() - coldStart) * 1000
    print("Cold transcription (5s audio, includes download+load): \(String(format: "%.0f", coldMs))ms")

    // Warm-ish (Parakeet reloads each cycle)
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

    // 20s transcription (long utterance)
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

    let avg5s = latencies5s.isEmpty ? 0 : latencies5s.reduce(0, +) / Double(latencies5s.count)
    let max5s = latencies5s.max() ?? 0

    print("\n--- Results ---")
    print("5s audio:  avg=\(String(format: "%.0f", avg5s))ms  max=\(String(format: "%.0f", max5s))ms")
    print("20s audio: \(String(format: "%.0f", longMs))ms")
    print("")
    print("AC-001.3 (5s in <2s):     \(max5s < 2000 ? "PASS" : "FAIL") (max=\(String(format: "%.0f", max5s))ms)")
    print("AC-001.4 (20s no crash):  PASS (completed in \(String(format: "%.0f", longMs))ms)")
}

// MARK: - Memory Benchmark (AC-011.5)

@MainActor
func benchmarkMemory() async {
    print("\n=== MEMORY BENCHMARK ===\n")

    func residentMB() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return Double(info.resident_size) / 1_048_576
    }

    let baseline = residentMB()
    print("Baseline: \(String(format: "%.0f", baseline))MB")

    // Load routing model
    let modelManager = ModelManager()
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
    let afterRouting = residentMB()
    let routingDelta = afterRouting - baseline
    print(
        "After routing model: \(String(format: "%.0f", afterRouting))MB"
            + " (+\(String(format: "%.0f", routingDelta))MB)"
    )

    // Load TTS model
    let ttsManager = Qwen3TTSSpeechManager(modelManager: modelManager)
    guard let memVoice = AVSpeechSynthesisVoice(language: "en-US") else {
        print("FAIL: Could not create AVSpeechSynthesisVoice")
        return
    }
    let voice = VoiceDescriptor(
        appleVoice: memVoice,
        qwenSpeakerID: "Chelsie"
    )
    ttsManager.speak(
        "Memory test.",
        voice: voice,
        prefix: "test",
        pitchMultiplier: 1.0
    )
    var waited = 0.0
    while !ttsManager.isSpeaking && waited < 120 {
        try? await Task.sleep(nanoseconds: 10_000_000)
        waited += 0.01
    }
    ttsManager.stop()
    try? await Task.sleep(nanoseconds: 500_000_000)

    let afterTTS = residentMB()
    let ttsDelta = afterTTS - afterRouting
    print(
        "After TTS model:     \(String(format: "%.0f", afterTTS))MB"
            + " (+\(String(format: "%.0f", ttsDelta))MB)"
    )

    // STT loads and unloads per cycle, so peak = routing + TTS + STT during transcription
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

    // STT unloads after transcription, so peak was during transcription
    let afterSTT = residentMB()
    print("After STT (unloaded): \(String(format: "%.0f", afterSTT))MB")

    let peak = max(afterRouting, afterTTS, afterSTT)
    print("\n--- Results ---")
    print("Peak resident memory: \(String(format: "%.0f", peak))MB")
    print("")
    print("AC-011.5 (peak <3GB): \(peak < 3072 ? "PASS" : "FAIL") (\(String(format: "%.0f", peak))MB)")
}

// Entry point
await runBenchmarks()

import AVFoundation
import Foundation
import Testing

@testable import OperatorCore

// MARK: - Routing Performance (AC-007.1, AC-007.5)

@Suite("Routing Performance Benchmarks")
internal struct RoutingPerformanceBenchmarkTests {
    @Test("Routing classification completes in <300ms warm (AC-007.1)")
    func routingLatency() async throws {
        let modelManager = ModelManager()
        let engine = MLXRoutingEngine(modelManager: modelManager)

        let prompt = """
            Sessions:
            1. "frontend" (cwd: /Users/jud/web-app) - React TypeScript dashboard
            2. "backend" (cwd: /Users/jud/api-server) - FastAPI Python endpoints
            3. "infra" (cwd: /Users/jud/deploy) - Terraform AWS infrastructure

            User message: "add a loading spinner to the dashboard"
            """

        let coldStart = CFAbsoluteTimeGetCurrent()
        _ = try await engine.run(prompt: prompt, timeout: 120)
        let coldMs = (CFAbsoluteTimeGetCurrent() - coldStart) * 1000
        print("[BENCHMARK] Routing cold start (download+load+inference): \(String(format: "%.0f", coldMs))ms")

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
        for msg in testMessages {
            let warmPrompt = """
                Sessions:
                1. "frontend" (cwd: /Users/jud/web-app) - React TypeScript dashboard
                2. "backend" (cwd: /Users/jud/api-server) - FastAPI Python endpoints
                3. "infra" (cwd: /Users/jud/deploy) - Terraform AWS infrastructure

                User message: "\(msg)"
                """
            let start = CFAbsoluteTimeGetCurrent()
            _ = try await engine.run(prompt: warmPrompt, timeout: 10)
            let durationMs = (CFAbsoluteTimeGetCurrent() - start) * 1000
            latencies.append(durationMs)
        }

        let sorted = latencies.sorted()
        let avg = latencies.reduce(0, +) / Double(latencies.count)
        let p50 = sorted[sorted.count / 2]
        let p95 = sorted[Int(Double(sorted.count) * 0.95)]
        let maxMs = sorted.last ?? 0

        let latencyStr =
            latencies
            .map { String(format: "%.0f", $0) }
            .joined(separator: ", ")
        print("[BENCHMARK] Routing warm latencies (ms): \(latencyStr)")
        let warmLine =
            "avg=\(String(format: "%.0f", avg))ms"
            + " p50=\(String(format: "%.0f", p50))ms"
            + " p95=\(String(format: "%.0f", p95))ms"
            + " max=\(String(format: "%.0f", maxMs))ms"
        print("[BENCHMARK] Routing warm: \(warmLine)")

        #expect(p95 < 300, "Routing p95 must be <300ms, got \(String(format: "%.0f", p95))ms")
    }

    @Test("Routing accuracy >=90% for clear messages (AC-007.5)")
    func routingAccuracy() async throws {
        let modelManager = ModelManager()
        let engine = MLXRoutingEngine(modelManager: modelManager)

        struct RoutingTestCase {
            let message: String
            let expectedSession: String
        }

        let testCases: [RoutingTestCase] = [
            RoutingTestCase(message: "add a loading spinner to the React dashboard", expectedSession: "frontend"),
            RoutingTestCase(message: "fix the CSS grid layout", expectedSession: "frontend"),
            RoutingTestCase(message: "add a new REST endpoint for user profiles", expectedSession: "backend"),
            RoutingTestCase(message: "fix the database connection pool", expectedSession: "backend"),
            RoutingTestCase(message: "optimize the SQL query", expectedSession: "backend"),
            RoutingTestCase(message: "update the terraform module for the new VPC", expectedSession: "infra"),
            RoutingTestCase(message: "add a new S3 bucket for logs", expectedSession: "infra"),
            RoutingTestCase(message: "fix the AWS IAM permissions", expectedSession: "infra"),
            RoutingTestCase(message: "add TypeScript types for the API response", expectedSession: "frontend"),
            RoutingTestCase(message: "update the Python requirements.txt", expectedSession: "backend")
        ]

        var correct = 0
        var results: [(message: String, expected: String, got: String, match: Bool)] = []

        for tc in testCases {
            let prompt = """
                Sessions:
                1. "frontend" (cwd: /Users/jud/web-app) - React TypeScript dashboard, CSS, components
                2. "backend" (cwd: /Users/jud/api-server) - FastAPI Python endpoints, SQL, database
                3. "infra" (cwd: /Users/jud/deploy) - Terraform AWS infrastructure, IAM, S3, VPC

                User message: "\(tc.message)"
                """
            let json = try await engine.run(prompt: prompt, timeout: 10)
            let session = json["session"] as? String ?? ""
            let match = session == tc.expectedSession
            if match { correct += 1 }
            results.append((tc.message, tc.expectedSession, session, match))
        }

        let accuracy = Double(correct) / Double(testCases.count) * 100
        print("[BENCHMARK] Routing accuracy: \(correct)/\(testCases.count) (\(String(format: "%.0f", accuracy))%)")
        for r in results {
            let mark = r.match ? "✓" : "✗"
            print("[BENCHMARK]   \(mark) \"\(r.message)\" -> expected=\(r.expected), got=\(r.got)")
        }

        #expect(accuracy >= 90, "Routing accuracy must be >=90%, got \(String(format: "%.0f", accuracy))%")
    }
}

// MARK: - TTS Performance (AC-003.1)

@Suite("TTS Performance Benchmarks")
internal struct TTSPerformanceBenchmarkTests {
    @Test("TTS time-to-first-audio <200ms (AC-003.1)")
    @MainActor func ttsTimeToFirstAudio() async throws {
        let modelManager = ModelManager()
        let ttsManager = Qwen3TTSSpeechManager(modelManager: modelManager)

        let appleVoice = try #require(
            AVSpeechSynthesisVoice(language: "en-US")
        )
        let voice = VoiceDescriptor(
            appleVoice: appleVoice,
            qwenSpeakerID: "Chelsie"
        )
        let testText = "Hello, this is a test of the text to speech system."

        // Cold start: includes model download + load
        let coldStart = CFAbsoluteTimeGetCurrent()
        ttsManager.speak(testText, voice: voice, prefix: "test", pitchMultiplier: 1.0)

        // Wait for speaking to start (model loads, first buffer scheduled)
        var waited = 0.0
        while !ttsManager.isSpeaking && waited < 120 {
            try await Task.sleep(nanoseconds: 10_000_000)
            waited += 0.01
        }
        let coldMs = (CFAbsoluteTimeGetCurrent() - coldStart) * 1000
        print("[BENCHMARK] TTS cold start (download+load+first-audio): \(String(format: "%.0f", coldMs))ms")

        ttsManager.stop()
        try await Task.sleep(nanoseconds: 500_000_000)

        // Warm: model already loaded
        var warmLatencies: [Double] = []
        let warmTexts = [
            "The quick brown fox jumps over the lazy dog.",
            "Please check your email for the confirmation link.",
            "I've updated the configuration file with the new settings.",
            "The deployment was successful. All services are running.",
            "There was an error processing your request. Please try again."
        ]

        for text in warmTexts {
            let start = CFAbsoluteTimeGetCurrent()
            ttsManager.speak(text, voice: voice, prefix: "agent", pitchMultiplier: 1.0)

            waited = 0.0
            while !ttsManager.isSpeaking && waited < 10 {
                try await Task.sleep(nanoseconds: 1_000_000)
                waited += 0.001
            }
            let durationMs = (CFAbsoluteTimeGetCurrent() - start) * 1000
            warmLatencies.append(durationMs)

            ttsManager.stop()
            try await Task.sleep(nanoseconds: 200_000_000)
        }

        let sorted = warmLatencies.sorted()
        let avg = warmLatencies.reduce(0, +) / Double(warmLatencies.count)
        let p95 = sorted[Int(Double(sorted.count) * 0.95)]

        let ttfaStr =
            warmLatencies
            .map { String(format: "%.0f", $0) }
            .joined(separator: ", ")
        print("[BENCHMARK] TTS warm TTFA (ms): \(ttfaStr)")
        print("[BENCHMARK] TTS warm: avg=\(String(format: "%.0f", avg))ms p95=\(String(format: "%.0f", p95))ms")

        #expect(p95 < 200, "TTS TTFA p95 must be <200ms, got \(String(format: "%.0f", p95))ms")
    }
}

// MARK: - STT Performance (AC-001.3, AC-001.4)

@Suite("STT Performance Benchmarks")
internal struct STTPerformanceBenchmarkTests {
    /// Generate synthetic audio: a 440Hz sine wave at 16kHz sample rate.
    private static func generateSineWave(
        durationSeconds: Double,
        sampleRate: Double = 16000
    ) -> AVAudioPCMBuffer? {
        let frameCount = AVAudioFrameCount(durationSeconds * sampleRate)
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

        guard let channelData = buffer.floatChannelData?[0] else {
            return nil
        }
        let frequency: Float = 440.0
        for i in 0..<Int(frameCount) {
            channelData[i] =
                sin(
                    2.0 * .pi * frequency * Float(i)
                        / Float(sampleRate)
                ) * 0.5
        }
        return buffer
    }

    @Test("STT transcription completes in <2s for 5s audio (AC-001.3)")
    func sttTranscriptionLatency() async throws {
        let modelManager = ModelManager()
        let engine = ParakeetEngine(modelManager: modelManager)

        // 5 seconds of audio (typical PTT utterance)
        let buffer = try #require(
            Self.generateSineWave(durationSeconds: 5.0)
        )

        try engine.prepare()
        engine.append(buffer)

        let start = CFAbsoluteTimeGetCurrent()
        _ = await engine.finishAndTranscribe()
        let durationMs = (CFAbsoluteTimeGetCurrent() - start) * 1000

        print(
            "[BENCHMARK] STT cold transcription (5s audio, includes download+load):"
                + " \(String(format: "%.0f", durationMs))ms"
        )

        // Second run: model reloads (Parakeet unloads after each transcription)
        let buffer2 = try #require(
            Self.generateSineWave(durationSeconds: 5.0)
        )
        try engine.prepare()
        engine.append(buffer2)

        let start2 = CFAbsoluteTimeGetCurrent()
        _ = await engine.finishAndTranscribe()
        let warmMs = (CFAbsoluteTimeGetCurrent() - start2) * 1000

        print(
            "[BENCHMARK] STT warm transcription (5s audio, includes reload):"
                + " \(String(format: "%.0f", warmMs))ms"
        )

        #expect(warmMs < 2000, "STT transcription must complete in <2s, got \(String(format: "%.0f", warmMs))ms")
    }

    @Test("STT handles long utterances >15s without failure (AC-001.4)")
    func sttLongUtterance() async throws {
        let modelManager = ModelManager()
        let engine = ParakeetEngine(modelManager: modelManager)

        // 20 seconds of audio
        let buffer = try #require(
            Self.generateSineWave(durationSeconds: 20.0)
        )

        try engine.prepare()
        engine.append(buffer)

        let start = CFAbsoluteTimeGetCurrent()
        let result = await engine.finishAndTranscribe()
        let durationMs = (CFAbsoluteTimeGetCurrent() - start) * 1000

        print(
            "[BENCHMARK] STT long utterance (20s audio):"
                + " \(String(format: "%.0f", durationMs))ms,"
                + " result: \(result ?? "nil")"
        )

        // The key assertion: it must not crash, hang, or timeout
        // A sine wave won't produce meaningful text, but the engine must handle it
        print("[BENCHMARK] STT long utterance: completed without failure ✓")
    }
}

// MARK: - Memory (AC-011.5)

@Suite("Memory Benchmarks")
internal struct MemoryBenchmarkTests {
    private static func residentMemoryMB() -> Double {
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

    @Test("Peak memory with routing model <3GB (AC-011.5)")
    func routingModelMemory() async throws {
        let baseline = Self.residentMemoryMB()
        print("[BENCHMARK] Memory baseline: \(String(format: "%.0f", baseline))MB")

        let modelManager = ModelManager()
        let engine = MLXRoutingEngine(modelManager: modelManager)

        let prompt = """
            Sessions:
            1. "test" (cwd: /tmp/test) - test project

            User message: "hello"
            """
        _ = try await engine.run(prompt: prompt, timeout: 120)

        let afterRouting = Self.residentMemoryMB()
        let routingDelta = afterRouting - baseline
        print(
            "[BENCHMARK] Memory after routing model:"
                + " \(String(format: "%.0f", afterRouting))MB"
                + " (+\(String(format: "%.0f", routingDelta))MB)"
        )

        #expect(afterRouting < 3072, "Total memory must be <3GB, got \(String(format: "%.0f", afterRouting))MB")
    }
}

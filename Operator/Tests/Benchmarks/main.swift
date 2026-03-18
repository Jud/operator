import AVFoundation
import KokoroTTS
import OperatorCore
@preconcurrency import Speech
import VocabularyCorrector
import WhisperKit
import os

private enum BenchmarkTarget: String, CaseIterable {
    case routing
    case routingAccuracy = "routing-accuracy"
    case routingLatency = "routing-latency"
    case tts
    case ttsRoundtrip = "tts-roundtrip"
    case play
    case sttReplay = "stt-replay"
    case sttReplayBatch = "stt-replay-batch"
    case sttInspect = "stt-inspect"
}

private let benchmarkRunnerEnvKey = "OPERATOR_BENCHMARK_RUNNER"

private func printBenchmarkUsage(listOnly: Bool = false) {
    print("Usage: ./scripts/run-benchmarks.sh [--debug|--release] [--no-build] [target...]")
    print("")
    print("Targets:")
    print("  routing           Run routing accuracy and latency benchmarks (heuristic only)")
    print("  routing-accuracy  Run only routing accuracy benchmark")
    print("  routing-latency   Run only routing latency benchmark")
    print("  tts               Run TTS synthesis and speed control verification")
    print("  tts-roundtrip     Run TTS→STT roundtrip quality check")
    print("  play              Synthesize and play example phrases through speakers")
    print("  stt-replay        Replay WAV through WhisperKitEngine with real-time streaming")
    print("  stt-replay-batch  Replay WAV through WhisperKitEngine in one shot (no streaming)")
    if listOnly {
        return
    }
    print("")
    print("Environment:")
    print("  OPERATOR_STT_WAV    Path to WAV file for stt-replay (default: latest audio trace)")
    print("")
    print("Examples:")
    print("  ./scripts/run-benchmarks.sh routing")
    print("  ./scripts/run-benchmarks.sh --no-build routing-accuracy")
    print("  OPERATOR_STT_WAV=~/path/to/audio.wav ./scripts/run-benchmarks.sh stt-replay")
}

private func makeBenchmarkVoice() -> VoiceDescriptor? {
    guard let appleVoice = AVSpeechSynthesisVoice(language: "en-US") else {
        return nil
    }
    return .system(appleVoice)
}

private func makeRoutingBenchmarkSessions(voice: VoiceDescriptor) -> [SessionState] {
    [
        SessionState(
            name: "ui-shell",
            tty: "/dev/benchmark1",
            cwd: "/Users/jud/work/operator-ui",
            context:
                "Branch feat/settings-loading. Editing web/src/routes/settings.tsx, web/src/components/LoadingCard.tsx, and web/src/styles/dashboard.css for React loading states, layout polish, and responsive CSS.",
            recentMessages: [
                SessionMessage(
                    role: "user",
                    text:
                        "Fix the janky loading state on the settings dashboard and make the cards stop shifting on mobile."
                ),
                SessionMessage(
                    role: "assistant",
                    text:
                        "I updated LoadingCard.tsx, SettingsRoute.tsx, and dashboard.css; next I am wiring the pending query state and skeleton layout."
                )
            ],
            status: .idle,
            lastActivity: Date(),
            voice: voice,
            pitchMultiplier: 1.0
        ),
        SessionState(
            name: "profile-api",
            tty: "/dev/benchmark2",
            cwd: "/Users/jud/work/operator-api",
            context:
                "Branch feat/profile-endpoints. Editing api/routes/profile.py, services/user_profile.py, and db/queries/profile.sql for REST endpoints, auth middleware, and Postgres query performance.",
            recentMessages: [
                SessionMessage(
                    role: "user",
                    text:
                        "The user profile endpoint is timing out under load and we still need a new REST route for profile preferences."
                ),
                SessionMessage(
                    role: "assistant",
                    text:
                        "I traced the slowdown to profile.sql and the connection pool, and I am adding the preferences endpoint in profile.py."
                )
            ],
            status: .idle,
            lastActivity: Date(),
            voice: voice,
            pitchMultiplier: 1.0
        ),
        SessionState(
            name: "staging-infra",
            tty: "/dev/benchmark3",
            cwd: "/Users/jud/work/operator-infra",
            context:
                "Branch chore/staging-network. Editing infra/envs/staging/main.tf, modules/vpc, and modules/iam_policy for Terraform apply failures, private subnet layout, IAM policies, and S3 logging.",
            recentMessages: [
                SessionMessage(
                    role: "user",
                    text:
                        "The staging VPC rollout failed after the Terraform change and logging still needs the new S3 bucket policy."
                ),
                SessionMessage(
                    role: "assistant",
                    text:
                        "I am updating the private subnet module, the IAM policy document, and the S3 logging resources in staging Terraform."
                )
            ],
            status: .idle,
            lastActivity: Date(),
            voice: voice,
            pitchMultiplier: 1.0
        )
    ]
}

private func makeBenchmarkRouter(
    sessions: [SessionState]
) async -> MessageRouter {
    let registry = SessionRegistry(voiceManager: VoiceManager())
    for session in sessions {
        await registry.register(name: session.name, tty: session.tty, cwd: session.cwd, context: session.context)
        await registry.updateContext(
            tty: session.tty,
            summary: session.context,
            recentMessages: session.recentMessages
        )
    }
    return MessageRouter(registry: registry)
}

// MARK: - Accuracy Test Types

private struct AccuracyTestCase {
    let message: String
    let expected: String
}

private struct AccuracyScenario {
    let name: String
    let sessions: [SessionState]
    let cases: [AccuracyTestCase]
}

// MARK: - Routing Benchmark

private func benchmarkRouting(runLatency: Bool, runAccuracy: Bool) async {
    guard runLatency || runAccuracy else { return }

    print("\n=== ROUTING BENCHMARK (Heuristic Only) ===\n")

    guard let benchmarkVoice = makeBenchmarkVoice() else {
        print("FAIL: Could not create AVSpeechSynthesisVoice")
        return
    }
    let benchmarkSessions = makeRoutingBenchmarkSessions(voice: benchmarkVoice)

    if runLatency {
        await runLatencyBenchmark(sessions: benchmarkSessions)
    }

    if runAccuracy {
        await runAccuracyBenchmark(sessions: benchmarkSessions)
    }
}

private func runLatencyBenchmark(sessions: [SessionState]) async {
    let router = await makeBenchmarkRouter(sessions: sessions)
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

    print("--- Routing Latency (Heuristic) ---")
    for message in testMessages {
        let start = CFAbsoluteTimeGetCurrent()
        _ = await router.route(text: message, routingState: RoutingState())
        let durationMs = (CFAbsoluteTimeGetCurrent() - start) * 1000
        latencies.append(durationMs)
        print("  \(String(format: "%5.2f", durationMs))ms | \"\(message)\"")
    }

    let sorted = latencies.sorted()
    let avg = latencies.reduce(0, +) / Double(latencies.count)
    let p50 = sorted[sorted.count / 2]
    let p95 = sorted[Int(Double(sorted.count - 1) * 0.95)]
    let maxMs = sorted.last ?? 0

    print("")
    print(
        "Latency:  avg=\(String(format: "%.2f", avg))ms  p50=\(String(format: "%.2f", p50))ms  p95=\(String(format: "%.2f", p95))ms  max=\(String(format: "%.2f", maxMs))ms"
    )
}

private func runAccuracyBenchmark(sessions: [SessionState]) async {
    let scenarios: [AccuracyScenario] = [
        AccuracyScenario(
            name: "operator stack",
            sessions: sessions,
            cases: [
                AccuracyTestCase(message: "add a loading spinner to the React dashboard", expected: "ui-shell"),
                AccuracyTestCase(message: "fix the CSS grid layout", expected: "ui-shell"),
                AccuracyTestCase(message: "add a new REST endpoint for user profiles", expected: "profile-api"),
                AccuracyTestCase(message: "fix the database connection pool", expected: "profile-api"),
                AccuracyTestCase(message: "optimize the SQL query", expected: "profile-api"),
                AccuracyTestCase(message: "update the terraform module for the new VPC", expected: "staging-infra"),
                AccuracyTestCase(message: "add a new S3 bucket for logs", expected: "staging-infra"),
                AccuracyTestCase(message: "fix the AWS IAM permissions", expected: "staging-infra"),
                AccuracyTestCase(message: "add TypeScript types for the API response", expected: "ui-shell"),
                AccuracyTestCase(message: "update the Python requirements.txt", expected: "profile-api")
            ]
        )
    ]

    print("\n--- Routing Accuracy (Heuristic) ---\n")
    var totalCorrect = 0
    var totalCases = 0

    for scenario in scenarios {
        let router = await makeBenchmarkRouter(sessions: scenario.sessions)

        print("Scenario: \(scenario.name)")
        var correct = 0

        for testCase in scenario.cases {
            let result = await router.route(text: testCase.message, routingState: RoutingState())
            let routed: String
            switch result {
            case .route(let session, _):
                routed = session
            case .clarify:
                routed = "<clarify>"
            case .notConfident:
                routed = "<not-confident>"
            default:
                routed = "<other>"
            }

            let ok = routed == testCase.expected
            if ok { correct += 1 }
            let marker = ok ? "OK" : "MISS"
            print("  [\(marker)] \"\(testCase.message)\" -> \(routed) (expected: \(testCase.expected))")
        }

        let accuracy = Double(correct) / Double(scenario.cases.count) * 100
        print("  Score: \(correct)/\(scenario.cases.count) (\(String(format: "%.0f", accuracy))%)\n")
        totalCorrect += correct
        totalCases += scenario.cases.count
    }

    let overall = Double(totalCorrect) / Double(totalCases) * 100
    print("Overall: \(totalCorrect)/\(totalCases) (\(String(format: "%.0f", overall))%)")
}

// MARK: - TTS Benchmark

private func benchmarkTTS() {
    print("\n=== TTS BENCHMARK ===\n")

    let modelDir = ModelManager.defaultDirectory(for: "com.operator")
    guard ModelManager.modelsAvailable(at: modelDir) else {
        print("SKIP: Models not found at \(modelDir.path)")
        return
    }

    do {
        let engine = try KokoroEngine(modelDirectory: modelDir)
        guard let voice = engine.availableVoices.first else {
            print("SKIP: No voices found")
            return
        }
        print("Engine loaded, voice: \(voice)")

        // --- Speed verification ---
        print("\n--- Speed Control Verification ---\n")

        let text = "The quick brown fox jumps over the lazy dog."
        let speeds: [Float] = [0.5, 1.0, 1.5, 2.0]
        var results: [(speed: Float, samples: Int, duration: Double, synthMs: Double)] = []

        for speed in speeds {
            let result = try engine.synthesize(text: text, voice: voice, speed: speed)
            let synthMs = result.synthesisTime * 1000
            results.append((speed, result.samples.count, result.duration, synthMs))
            let line = String(
                format: "  speed=%.1f: %6d samples, %.2fs audio, %.0fms synth",
                speed,
                result.samples.count,
                result.duration,
                synthMs
            )
            print(line)
        }

        guard let normal = results.first(where: { $0.speed == 1.0 }) else { return }
        var allPass = true

        for r in results where r.speed != 1.0 {
            let ratio = Double(r.samples) / Double(normal.samples)
            let expectedRatio = 1.0 / Double(r.speed)
            let tolerance = 0.3
            let pass = abs(ratio - expectedRatio) < tolerance
            let marker = pass ? "PASS" : "FAIL"
            if !pass { allPass = false }
            let line = String(
                format: "  [%s] speed=%.1f ratio=%.2f (expected ~%.2f)",
                marker,
                r.speed,
                ratio,
                expectedRatio
            )
            print(line)
        }

        // --- Latency benchmark ---
        print("\n--- Synthesis Latency ---\n")

        let phrases = [
            "Hello.",
            "The quick brown fox jumps over the lazy dog.",
            "I've updated the configuration file and restarted the service. The deployment should be live within a few minutes."
        ]

        for phrase in phrases {
            var lastResult: SynthesisResult?
            for _ in 0..<3 {
                lastResult = try engine.synthesize(text: phrase, voice: voice)
            }
            guard let result = lastResult else { continue }
            let lastMs = result.synthesisTime * 1000
            let line = String(
                format: "  %.0fms (%.1fx RT) | \"%@\"",
                lastMs,
                result.realTimeFactor,
                String(phrase.prefix(60))
            )
            print(line)
        }

        print(allPass ? "\nSpeed control: ALL PASS" : "\nSpeed control: SOME FAILURES")
    } catch {
        print("ERROR: \(error)")
    }
}

// MARK: - TTS→STT Roundtrip

private func benchmarkTTSRoundtrip() async {
    print("\n=== TTS→STT ROUNDTRIP ===\n")

    let modelDir = ModelManager.defaultDirectory(for: "com.operator")
    guard ModelManager.modelsAvailable(at: modelDir) else {
        print("SKIP: Models not found at \(modelDir.path)")
        return
    }

    // Request speech recognition authorization.
    let authStatus = await AppleSpeechEngine.requestAuthorization()
    guard authStatus == .authorized else {
        print("SKIP: Speech recognition not authorized (status=\(authStatus.rawValue))")
        return
    }

    guard let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US")),
        recognizer.isAvailable
    else {
        print("SKIP: Speech recognizer not available")
        return
    }

    do {
        let engine = try KokoroEngine(modelDirectory: modelDir)
        let voice = "af_heart"
        guard engine.availableVoices.contains(voice) else {
            print("SKIP: Voice '\(voice)' not found")
            return
        }

        let testPhrases = [
            "Hi",
            "Hello",
            "Yes",
            "No",
            "Okay",
            "Sure",
            "Operator is ready",
            "Sure, I can help with that.",
            "The quick brown fox jumps over the lazy dog.",
            "All ten sections done. Twelve decisions made, zero unresolved."
        ]

        guard
            let format = AVAudioFormat(
                standardFormatWithSampleRate: Double(KokoroEngine.sampleRate),
                channels: 1
            )
        else {
            print("FAIL: Could not create audio format")
            return
        }

        var passed = 0
        var failed = 0

        for phrase in testPhrases {
            let result = try engine.synthesize(text: phrase, voice: voice)

            // Create AVAudioPCMBuffer from synthesis output.
            let frameCount = AVAudioFrameCount(result.samples.count)
            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
                print("  [FAIL] \"\(phrase)\" — could not create buffer")
                failed += 1
                continue
            }
            buffer.frameLength = frameCount
            result.samples.withUnsafeBufferPointer { src in
                guard let channelData = buffer.floatChannelData,
                    let baseAddr = src.baseAddress
                else { return }
                channelData[0].update(from: baseAddr, count: result.samples.count)
            }

            // Feed to SFSpeechRecognizer.
            let recognized = await recognizeFromBuffer(
                buffer,
                recognizer: recognizer
            )

            let synthMs = String(format: "%.0f", result.synthesisTime * 1000)
            let dur = String(format: "%.1f", result.duration)

            if let recognized {
                let normalizedInput = phrase.lowercased()
                    .replacingOccurrences(of: ".", with: "")
                    .trimmingCharacters(in: .whitespaces)
                let normalizedOutput = recognized.lowercased()
                    .replacingOccurrences(of: ".", with: "")
                    .trimmingCharacters(in: .whitespaces)

                let match =
                    normalizedOutput.contains(normalizedInput)
                    || normalizedInput.contains(normalizedOutput)
                    || levenshteinRatio(normalizedInput, normalizedOutput) > 0.7

                if match {
                    print("  [PASS] \"\(phrase)\" → \"\(recognized)\" (\(synthMs)ms, \(dur)s)")
                    passed += 1
                } else {
                    print("  [FAIL] \"\(phrase)\" → \"\(recognized)\" (\(synthMs)ms, \(dur)s)")
                    failed += 1
                }
            } else {
                print("  [FAIL] \"\(phrase)\" → <no recognition> (\(synthMs)ms, \(dur)s)")
                failed += 1
            }
        }

        print("\nResults: \(passed)/\(passed + failed) passed")
        if failed > 0 {
            print("WARNING: \(failed) phrases did not roundtrip correctly")
        }
    } catch {
        print("ERROR: \(error)")
    }
}

/// Recognize speech from an audio buffer using SFSpeechRecognizer.
private func recognizeFromBuffer(
    _ buffer: AVAudioPCMBuffer,
    recognizer: SFSpeechRecognizer
) async -> String? {
    let request = SFSpeechAudioBufferRecognitionRequest()
    request.shouldReportPartialResults = false
    if #available(macOS 15, *) {
        request.requiresOnDeviceRecognition = true
    }

    request.append(buffer)
    request.endAudio()

    return await withCheckedContinuation { continuation in
        let resumed = OSAllocatedUnfairLock(initialState: false)
        func claimOnce() -> Bool {
            resumed.withLock { val in
                guard !val else { return false }
                val = true
                return true
            }
        }
        let task = recognizer.recognitionTask(with: request) { result, error in
            if let result, result.isFinal {
                guard claimOnce() else { return }
                continuation.resume(returning: result.bestTranscription.formattedString)
            } else if error != nil {
                guard claimOnce() else { return }
                continuation.resume(returning: nil)
            }
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 10) {
            guard claimOnce() else { return }
            task.cancel()
            continuation.resume(returning: nil)
        }
    }
}

/// Levenshtein distance ratio (0.0 = completely different, 1.0 = identical).
private func levenshteinRatio(_ lhs: String, _ rhs: String) -> Double {
    VocabularyCorrector.editDistanceRatio(lhs, rhs)
}

// MARK: - Play

private func playExamples() async {
    print("\n=== KOKORO TTS PLAYBACK ===\n")

    let modelDir = ModelManager.defaultDirectory(for: "com.operator")
    guard ModelManager.modelsAvailable(at: modelDir) else {
        print("SKIP: Models not found at \(modelDir.path)")
        return
    }

    do {
        let engine = try KokoroEngine(modelDirectory: modelDir)
        engine.warmUp()
        let voices = ["af_heart", "af_bella", "am_adam", "bf_emma"]
            .filter { engine.availableVoices.contains($0) }
        let phrase =
            "I've updated the configuration and restarted the service."
            + " The deployment should be live within a few minutes."

        print("Available: \(engine.availableVoices.prefix(10).joined(separator: ", "))...\n")

        let audioEngine = AVAudioEngine()
        let playerNode = AVAudioPlayerNode()
        guard
            let format = AVAudioFormat(
                standardFormatWithSampleRate: Double(KokoroEngine.sampleRate),
                channels: 1
            )
        else {
            print("FAIL: Could not create AVAudioFormat")
            return
        }
        audioEngine.attach(playerNode)
        audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: format)
        try audioEngine.start()

        for voice in voices {
            print("  [\(voice)] \"\(phrase)\"")
            let result = try engine.synthesize(text: phrase, voice: voice)
            let synthMs = String(format: "%.0f", result.synthesisTime * 1000)
            let dur = String(format: "%.1f", result.duration)
            print("  synth: \(synthMs)ms, audio: \(dur)s")

            let frameCount = AVAudioFrameCount(result.samples.count)
            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
                continue
            }
            buffer.frameLength = frameCount
            result.samples.withUnsafeBufferPointer { src in
                guard let channelData = buffer.floatChannelData,
                    let baseAddr = src.baseAddress
                else { return }
                channelData[0].update(from: baseAddr, count: result.samples.count)
            }

            playerNode.stop()
            await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
                playerNode.scheduleBuffer(buffer) {
                    cont.resume()
                }
                playerNode.play()
            }
            print()
        }

        audioEngine.stop()
    } catch {
        print("ERROR: \(error)")
    }
}

// MARK: - STT Benchmark

/// Load a WhisperKit engine using the default model.
private func loadWhisperKitEngine() async throws -> WhisperKitEngine {
    let model = WhisperKitModelManager.defaultModel
    guard let modelPath = WhisperKitModelManager.modelPath(model) else {
        throw STTBenchError.modelNotFound(model)
    }
    return try await WhisperKitEngine.create(modelFolder: modelPath)
}

private enum STTBenchError: Error, CustomStringConvertible {
    case modelNotFound(String)
    case noWavFile
    case loadFailed(String)

    var description: String {
        switch self {
        case .modelNotFound(let m): return "Model not found: \(m). Run Operator once to download."
        case .noWavFile: return "No WAV file found in audio-traces"
        case .loadFailed(let p): return "Failed to load WAV: \(p)"
        }
    }
}

/// Resolve a WAV path from OPERATOR_STT_WAV env or the latest audio trace.
private func resolveWavPath() throws -> String {
    if let envPath = ProcessInfo.processInfo.environment["OPERATOR_STT_WAV"] {
        let expanded = NSString(string: envPath).expandingTildeInPath
        guard FileManager.default.fileExists(atPath: expanded) else {
            throw STTBenchError.loadFailed(expanded)
        }
        return expanded
    }

    let traceDir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)
        .first!
        .appendingPathComponent("Operator/audio-traces")

    guard
        let files = try? FileManager.default.contentsOfDirectory(
            at: traceDir,
            includingPropertiesForKeys: [.contentModificationDateKey],
            options: .skipsHiddenFiles
        )
    else {
        throw STTBenchError.noWavFile
    }

    let wavs = files.filter { $0.pathExtension == "wav" }
        .sorted { lhs, rhs in
            let d1 =
                (try? lhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
                ?? .distantPast
            let d2 =
                (try? rhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
                ?? .distantPast
            return d1 > d2
        }

    guard let latest = wavs.first else {
        throw STTBenchError.noWavFile
    }
    return latest.path
}

/// Load a WAV file as 16kHz mono Float32 samples using WhisperKit's AudioProcessor.
private func loadWavSamples(path: String) throws -> (samples: [Float], sampleRate: Int, originalSampleRate: Double) {
    let url = URL(fileURLWithPath: path)
    let audioFile = try AVAudioFile(forReading: url)
    let originalRate = audioFile.processingFormat.sampleRate
    let frameCount = AVAudioFrameCount(audioFile.length)

    guard let buffer = AVAudioPCMBuffer(pcmFormat: audioFile.processingFormat, frameCapacity: frameCount) else {
        throw STTBenchError.loadFailed(path)
    }
    try audioFile.read(into: buffer)

    // Convert to 16kHz mono for direct transcription
    let targetRate: Double = 16_000
    guard
        let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: targetRate,
            channels: 1,
            interleaved: false
        )
    else {
        throw STTBenchError.loadFailed("Cannot create 16kHz format")
    }

    guard let converter = AVAudioConverter(from: buffer.format, to: targetFormat) else {
        throw STTBenchError.loadFailed("Cannot create converter")
    }

    let ratio = targetRate / originalRate
    let outFrames = AVAudioFrameCount(Double(frameCount) * ratio) + 1
    guard let outBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outFrames) else {
        throw STTBenchError.loadFailed("Cannot create output buffer")
    }

    var error: NSError?
    var consumed = false
    converter.convert(to: outBuffer, error: &error) { _, outStatus in
        if consumed {
            outStatus.pointee = .noDataNow
            return nil
        }
        consumed = true
        outStatus.pointee = .haveData
        return buffer
    }

    if let error { throw error }

    guard let channelData = outBuffer.floatChannelData?[0] else {
        throw STTBenchError.loadFailed("No channel data after conversion")
    }
    let count = Int(outBuffer.frameLength)
    let samples = Array(UnsafeBufferPointer(start: channelData, count: count))
    return (samples, 16_000, originalRate)
}

/// Batch STT: load WAV, transcribe in one shot, measure latency.
private func benchmarkSTTLatency(label: String, wavPath: String) async {
    print("\n--- STT \(label) ---\n")
    print("  WAV: \(wavPath)")

    do {
        let (samples, _, _) = try loadWavSamples(path: wavPath)
        let audioDuration = Double(samples.count) / 16_000
        print("  Audio: \(String(format: "%.1f", audioDuration))s (\(samples.count) samples @ 16kHz)")

        let engine = try await loadWhisperKitEngine()
        print("  Engine loaded")

        // Warm up with a tiny slice
        try engine.prepare(contextualStrings: [])
        _ = await engine.finishAndTranscribe()

        // Benchmark: batch transcription
        try engine.prepare(contextualStrings: [])

        // Feed all samples at once via a buffer
        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16_000,
            channels: 1,
            interleaved: false
        )!
        let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))!
        buf.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { src in
            buf.floatChannelData![0].update(from: src.baseAddress!, count: samples.count)
        }
        engine.append(buf)

        let start = ContinuousClock.now
        let result = await engine.finishAndTranscribe()
        let elapsed = ContinuousClock.now - start
        let ms = Int(elapsed.components.seconds) * 1000 + Int(elapsed.components.attoseconds / 1_000_000_000_000_000)

        let rtf = Double(ms) / 1000.0 / audioDuration
        print("  Result: \(ms)ms (\(String(format: "%.2f", rtf))x RT)")
        if let text = result {
            print("  Text: \"\(text.prefix(120))\"")
        } else {
            print("  Text: <empty>")
        }
    } catch {
        print("  ERROR: \(error)")
    }
}

/// Streaming STT: replay WAV through engine in real-time chunks, exercising the
/// background retranscription loop and confirmation pipeline.
///
/// Matches the real mic tap exactly: 1024-frame buffers at the file's native
/// sample rate (typically 48kHz), delivered at real-time pace (~21ms apart).
private func benchmarkSTTStreaming(wavPath: String) async {
    print("\n--- STT Replay ---\n")
    print("  WAV: \(wavPath)")

    do {
        let url = URL(fileURLWithPath: wavPath)
        let audioFile = try AVAudioFile(forReading: url)
        let fileFormat = audioFile.processingFormat
        let totalFrames = AVAudioFrameCount(audioFile.length)
        let fileSampleRate = fileFormat.sampleRate
        let audioDuration = Double(totalFrames) / fileSampleRate

        // Match SpeechTranscriber's installTap(bufferSize: 1024)
        let tapBufferSize: AVAudioFrameCount = 1_024
        let chunkDuration = Double(tapBufferSize) / fileSampleRate
        let totalChunks = Int(ceil(Double(totalFrames) / Double(tapBufferSize)))

        print("  Audio:  \(String(format: "%.1f", audioDuration))s @ \(Int(fileSampleRate))Hz")
        print(
            "  Chunks: \(totalChunks) x \(tapBufferSize) frames (\(String(format: "%.1f", chunkDuration * 1000))ms each)"
        )

        let engine = try await loadWhisperKitEngine()
        print("  Engine loaded\n")

        // Warm up
        try engine.prepare(contextualStrings: [])
        _ = await engine.finishAndTranscribe()

        // Start session
        try engine.prepare(contextualStrings: [])

        let sessionStart = ContinuousClock.now
        var framesDelivered: AVAudioFrameCount = 0

        // Feed 1024-frame chunks at real-time pace, matching the mic tap
        while framesDelivered < totalFrames {
            let remaining = totalFrames - framesDelivered
            let thisChunk = min(tapBufferSize, remaining)

            guard let buf = AVAudioPCMBuffer(pcmFormat: fileFormat, frameCapacity: thisChunk) else {
                break
            }
            try audioFile.read(into: buf, frameCount: thisChunk)
            engine.append(buf)
            framesDelivered += thisChunk

            // Pace to real-time: sleep until this chunk's wall-clock delivery time
            let targetTime = Double(framesDelivered) / fileSampleRate
            let wallTime = ContinuousClock.now - sessionStart
            let wallSec = Double(wallTime.components.seconds) + Double(wallTime.components.attoseconds) / 1e18
            let sleepNeeded = targetTime - wallSec
            if sleepNeeded > 0.001 {
                try await Task.sleep(nanoseconds: UInt64(sleepNeeded * 1_000_000_000))
            }
        }

        let feedDone = ContinuousClock.now
        let feedMs = durationMs(feedDone - sessionStart)

        // Simulate key release
        let releaseStart = ContinuousClock.now
        let result = await engine.finishAndTranscribe()
        let releaseMs = durationMs(ContinuousClock.now - releaseStart)
        let totalMs = durationMs(ContinuousClock.now - sessionStart)

        print("  Feed:     \(feedMs)ms (real-time delivery of \(String(format: "%.1f", audioDuration))s)")
        print("  Release:  \(releaseMs)ms (post-release latency — what the user feels)")
        print("  Total:    \(totalMs)ms")

        if let text = result {
            print("  Text:     \"\(text.prefix(120))\"")
        } else {
            print("  Text:     <empty>")
        }

        // Assess
        let rating: String
        if releaseMs < 500 {
            rating = "FAST (< 500ms)"
        } else if releaseMs < 1000 {
            rating = "NORMAL (500ms-1s)"
        } else if releaseMs < 2000 {
            rating = "SLOW (1-2s)"
        } else {
            rating = "PROBLEM (> 2s)"
        }
        print("  Rating:   \(rating)")

        // Verify: run a batch decode and compare to streaming result.
        let (batchSamples, _, _) = try loadWavSamples(path: wavPath)
        try engine.prepare(contextualStrings: [])
        let batchFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16_000,
            channels: 1,
            interleaved: false
        )!
        let batchBuf = AVAudioPCMBuffer(
            pcmFormat: batchFormat,
            frameCapacity: AVAudioFrameCount(batchSamples.count)
        )!
        batchBuf.frameLength = AVAudioFrameCount(batchSamples.count)
        batchSamples.withUnsafeBufferPointer { src in
            batchBuf.floatChannelData![0].update(from: src.baseAddress!, count: batchSamples.count)
        }
        engine.append(batchBuf)
        let batchResult = await engine.finishAndTranscribe()

        let streamText = result ?? ""
        let batchText = batchResult ?? ""
        if streamText == batchText {
            print("  Verify:   PASS (streaming == batch)")
        } else {
            let ratio = levenshteinRatio(streamText.lowercased(), batchText.lowercased())
            let pct = String(format: "%.0f", ratio * 100)
            if ratio > 0.9 {
                print("  Verify:   OK (\(pct)% match)")
            } else {
                print("  Verify:   MISMATCH (\(pct)% match)")
                print("  Batch:    \"\(batchText.prefix(120))\"")
            }
        }
    } catch {
        print("  ERROR: \(error)")
    }
}

private func durationMs(_ d: Duration) -> Int {
    Int(d.components.seconds) * 1000 + Int(d.components.attoseconds / 1_000_000_000_000_000)
}

/// Transcribe a WAV directly via WhisperKit pipe and dump per-segment quality metrics.
private func inspectSTTSegments(wavPath: String) async {
    print("\n--- STT Segment Inspector ---\n")
    print("  WAV: \(wavPath)")

    do {
        let (samples, _, _) = try loadWavSamples(path: wavPath)
        let audioDuration = Double(samples.count) / 16_000
        print("  Audio: \(String(format: "%.1f", audioDuration))s (\(samples.count) samples @ 16kHz)\n")

        let model = WhisperKitModelManager.defaultModel
        guard let modelPath = WhisperKitModelManager.modelPath(model) else {
            print("  ERROR: Model not found: \(model)")
            return
        }
        let config = WhisperKitConfig(modelFolder: modelPath)
        let pipe = try await WhisperKit(config)

        var options = DecodingOptions()
        options.language = "en"
        options.skipSpecialTokens = true
        options.chunkingStrategy = ChunkingStrategy.none
        if audioDuration < 3 {
            options.windowClipTime = 0
        }

        let results = try await pipe.transcribe(audioArray: samples, decodeOptions: options)
        guard let result = results.first else {
            print("  No transcription result")
            return
        }

        print("  Segments: \(result.segments.count)")
        print("")
        print("  #  | Start  | End    | noSpeech | avgLogP  | compRatio | temp | Text")
        print("  ---|--------|--------|----------|----------|-----------|------|-----")

        for (i, seg) in result.segments.enumerated() {
            let noSpeech = String(format: "%.3f", seg.noSpeechProb)
            let avgLog = String(format: "%.3f", seg.avgLogprob)
            let comp = String(format: "%.2f", seg.compressionRatio)
            let temp = String(format: "%.1f", seg.temperature)
            let text = String(seg.text.prefix(60)).trimmingCharacters(in: .whitespaces)
            print(
                "  \(String(format: "%2d", i)) | \(String(format: "%5.1fs", seg.start)) | \(String(format: "%5.1fs", seg.end)) | \(noSpeech)    | \(avgLog)  | \(comp)     | \(temp)  | \"\(text)\""
            )

            // Per-token log probs
            if seg.tokens.count == seg.tokenLogProbs.count, let tokenizer = pipe.tokenizer {
                let specialBegin = tokenizer.specialTokens.specialTokenBegin
                var entries: [(id: Int, logProb: Float)] = []
                for (tokenId, logProbMap) in zip(seg.tokens, seg.tokenLogProbs) {
                    guard tokenId < specialBegin else { continue }
                    entries.append((tokenId, logProbMap.values.first ?? 0))
                }
                if !entries.isEmpty {
                    let perToken = entries.map { e in
                        let pct = exp(e.logProb) * 100
                        return String(format: "%d:%.0f%%", e.id, pct)
                    }
                    print("       Tokens: \(perToken.joined(separator: " "))")
                    // Flag tokens under 30% confidence
                    let lowConf = entries.filter { exp($0.logProb) < 0.3 }
                    if !lowConf.isEmpty {
                        let flagged = lowConf.map { e in
                            String(format: "id=%d (%.0f%%)", e.id, exp(e.logProb) * 100)
                        }
                        print("       ⚠ Low: \(flagged.joined(separator: ", "))")
                    }
                }
            }
        }

        // Summary stats
        let avgNoSpeech = result.segments.map(\.noSpeechProb).reduce(0, +) / Float(result.segments.count)
        let avgLogP = result.segments.map(\.avgLogprob).reduce(0, +) / Float(result.segments.count)
        let avgComp = result.segments.map(\.compressionRatio).reduce(0, +) / Float(result.segments.count)
        let maxNoSpeech = result.segments.map(\.noSpeechProb).max() ?? 0
        let minLogP = result.segments.map(\.avgLogprob).min() ?? 0
        let maxComp = result.segments.map(\.compressionRatio).max() ?? 0

        print("")
        print("  Summary:")
        print(
            "    noSpeechProb:    avg=\(String(format: "%.3f", avgNoSpeech))  max=\(String(format: "%.3f", maxNoSpeech))"
        )
        print("    avgLogprob:      avg=\(String(format: "%.3f", avgLogP))  min=\(String(format: "%.3f", minLogP))")
        print("    compressionRatio: avg=\(String(format: "%.2f", avgComp))  max=\(String(format: "%.2f", maxComp))")

        print(
            "\n  Full text: \"\(result.segments.map { $0.text.trimmingCharacters(in: .whitespaces) }.joined(separator: " "))\""
        )
    } catch {
        print("  ERROR: \(error)")
    }
}

// MARK: - Main

@main
enum BenchmarkRunner {
    static func main() async {
        guard ProcessInfo.processInfo.environment[benchmarkRunnerEnvKey] != nil else {
            print("Run benchmarks via: ./scripts/run-benchmarks.sh [target...]")
            return
        }

        let args = Array(CommandLine.arguments.dropFirst())

        if args.isEmpty || args.contains("list") || args.contains("help") {
            printBenchmarkUsage(listOnly: args.contains("list"))
            return
        }

        let targets: Set<BenchmarkTarget> = Set(args.compactMap { BenchmarkTarget(rawValue: $0) })

        if targets.isEmpty {
            print(
                "No recognized targets. Available: \(BenchmarkTarget.allCases.map(\.rawValue).joined(separator: ", "))"
            )
            return
        }

        let includesRouting = targets.contains(.routing)
        if includesRouting || targets.contains(.routingLatency) || targets.contains(.routingAccuracy) {
            await benchmarkRouting(
                runLatency: includesRouting || targets.contains(.routingLatency),
                runAccuracy: includesRouting || targets.contains(.routingAccuracy)
            )
        }

        if targets.contains(.tts) {
            benchmarkTTS()
        }

        if targets.contains(.ttsRoundtrip) {
            await benchmarkTTSRoundtrip()
        }

        if targets.contains(.play) {
            await playExamples()
        }

        if targets.contains(.sttInspect) {
            do {
                let wavPath = try resolveWavPath()
                await inspectSTTSegments(wavPath: wavPath)
            } catch {
                print("\nSTT ERROR: \(error)")
            }
        }

        if targets.contains(.sttReplay) || targets.contains(.sttReplayBatch) {
            do {
                let wavPath = try resolveWavPath()

                if targets.contains(.sttReplayBatch) {
                    await benchmarkSTTLatency(label: "Batch", wavPath: wavPath)
                }

                if targets.contains(.sttReplay) {
                    await benchmarkSTTStreaming(wavPath: wavPath)
                }
            } catch {
                print("\nSTT ERROR: \(error)")
            }
        }

        print("\nDone.")
    }
}

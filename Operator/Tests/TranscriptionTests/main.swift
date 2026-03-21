import AVFoundation
import OperatorCore
import WhisperKit

/// Transcription regression test suite.
///
/// Replays saved WAV fixtures through the WhisperKitEngine streaming pipeline
/// (background confirmation loop + final decode) and compares results against
/// known ground truth. Catches regressions in the confirmation algorithm,
/// tail decode, silence detection, and audio format conversion.
///
/// Fixtures live in ~/Library/Application Support/Operator/test-fixtures/
/// Each fixture is a pair: <name>.wav + <name>.expected.txt
///
/// Usage:
///   swift run TranscriptionTests              # Run all fixtures
///   swift run TranscriptionTests streaming     # Streaming pipeline only
///   swift run TranscriptionTests batch         # Batch decode only
///   swift run TranscriptionTests <name>        # Run a specific fixture

private let fixtureDir: URL = {
    let appSupport =
        FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first ?? FileManager.default.homeDirectoryForCurrentUser
    return appSupport.appendingPathComponent("Operator/test-fixtures", isDirectory: true)
}()

// MARK: - Fixture Loading

private struct Fixture {
    let name: String
    let wavPath: String
    let expectedText: String
    let audioDuration: Double
}

private func loadFixtures(filter: String? = nil) throws -> [Fixture] {
    let fm = FileManager.default
    guard
        let files = try? fm.contentsOfDirectory(
            at: fixtureDir,
            includingPropertiesForKeys: nil,
            options: .skipsHiddenFiles
        )
    else {
        print("No fixtures directory at \(fixtureDir.path)")
        return []
    }

    let wavFiles = files.filter { $0.pathExtension == "wav" }.sorted { $0.lastPathComponent < $1.lastPathComponent }

    if wavFiles.isEmpty {
        print("No .wav fixtures found in \(fixtureDir.path)")
        print("Capture one with: ./scripts/capture-fixture.sh <name> [wav-path]")
        return []
    }

    var fixtures: [Fixture] = []
    for wav in wavFiles {
        let name = wav.deletingPathExtension().lastPathComponent
        if let filter, name != filter && filter != "streaming" && filter != "batch" {
            continue
        }

        let expectedFile = wav.deletingPathExtension().appendingPathExtension("expected.txt")
        guard fm.fileExists(atPath: expectedFile.path) else {
            print("WARNING: No expected.txt for \(name), skipping")
            continue
        }

        let expectedText = try String(contentsOf: expectedFile, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)

        let audioFile = try AVAudioFile(forReading: wav)
        let duration = Double(audioFile.length) / audioFile.processingFormat.sampleRate

        fixtures.append(
            Fixture(
                name: name,
                wavPath: wav.path,
                expectedText: expectedText,
                audioDuration: duration
            )
        )
    }
    return fixtures
}

// MARK: - WAV Loading

private func loadWavSamples(path: String) throws -> (samples: [Float], sampleRate: Int) {
    let url = URL(fileURLWithPath: path)
    let audioFile = try AVAudioFile(forReading: url)
    let frameCount = AVAudioFrameCount(audioFile.length)

    guard
        let buffer = AVAudioPCMBuffer(
            pcmFormat: audioFile.processingFormat,
            frameCapacity: frameCount
        )
    else {
        throw NSError(
            domain: "TranscriptionTests",
            code: 1,
            userInfo: [
                NSLocalizedDescriptionKey: "Could not create buffer for \(path)"
            ]
        )
    }
    try audioFile.read(into: buffer)

    let targetRate: Double = 16_000
    guard
        let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: targetRate,
            channels: 1,
            interleaved: false
        )
    else {
        throw NSError(domain: "TranscriptionTests", code: 2)
    }

    guard let converter = AVAudioConverter(from: buffer.format, to: targetFormat) else {
        throw NSError(domain: "TranscriptionTests", code: 3)
    }

    let ratio = targetRate / audioFile.processingFormat.sampleRate
    let outFrames = AVAudioFrameCount(Double(frameCount) * ratio) + 1
    guard let outBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outFrames) else {
        throw NSError(domain: "TranscriptionTests", code: 4)
    }

    var error: NSError?
    var consumed = false
    converter.convert(to: outBuffer, error: &error) { _, outStatus in
        if consumed {
            outStatus.pointee = .endOfStream
            return nil
        }
        consumed = true
        outStatus.pointee = .haveData
        return buffer
    }

    if let error { throw error }

    guard let channelData = outBuffer.floatChannelData?[0] else {
        throw NSError(domain: "TranscriptionTests", code: 5)
    }
    let count = Int(outBuffer.frameLength)
    return (Array(UnsafeBufferPointer(start: channelData, count: count)), 16_000)
}

// MARK: - Engine Loading

private func loadEngine() async throws -> WhisperKitEngine {
    let variant =
        ProcessInfo.processInfo.environment["WHISPER_MODEL"]
        ?? WhisperKitModelManager.defaultModel
    guard let modelPath = WhisperKitModelManager.modelPath(variant) else {
        print("ERROR: WhisperKit model '\(variant)' not found. Download with:")
        print("  swift run Benchmarks stt-replay-batch")
        throw NSError(domain: "TranscriptionTests", code: 10)
    }
    return try await WhisperKitEngine.create(modelFolder: modelPath)
}

// MARK: - Levenshtein Similarity

private func levenshteinRatio(_ lhs: String, _ rhs: String) -> Double {
    let left = Array(lhs)
    let right = Array(rhs)
    let leftLen = left.count
    let rightLen = right.count
    if leftLen == 0 && rightLen == 0 {
        return 1.0
    }
    if leftLen == 0 || rightLen == 0 {
        return 0.0
    }

    var prev = Array(0...rightLen)
    var curr = [Int](repeating: 0, count: rightLen + 1)

    for li in 1...leftLen {
        curr[0] = li
        for ri in 1...rightLen {
            let cost = left[li - 1] == right[ri - 1] ? 0 : 1
            curr[ri] = min(prev[ri] + 1, curr[ri - 1] + 1, prev[ri - 1] + cost)
        }
        swap(&prev, &curr)
    }

    let distance = prev[rightLen]
    return 1.0 - Double(distance) / Double(max(leftLen, rightLen))
}

// MARK: - Test Runners

private struct TestResult {
    let fixture: String
    let mode: String
    let passed: Bool
    let similarity: Double
    let resultText: String
    let expectedText: String
    let durationMs: Int
    let detail: String
}

private func runBatchTest(engine: WhisperKitEngine, fixture: Fixture) async -> TestResult {
    let start = ContinuousClock.now

    try? engine.prepare(contextualStrings: [])
    guard let loaded = try? loadWavSamples(path: fixture.wavPath) else {
        return TestResult(
            fixture: fixture.name,
            mode: "batch",
            passed: false,
            similarity: 0,
            resultText: "",
            expectedText: fixture.expectedText,
            durationMs: 0,
            detail: "Failed to load WAV"
        )
    }
    let (samples, _) = loaded

    guard
        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16_000,
            channels: 1,
            interleaved: false
        ),
        let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))
    else {
        return TestResult(
            fixture: fixture.name,
            mode: "batch",
            passed: false,
            similarity: 0,
            resultText: "",
            expectedText: fixture.expectedText,
            durationMs: 0,
            detail: "Buffer creation failed"
        )
    }
    buf.frameLength = AVAudioFrameCount(samples.count)
    samples.withUnsafeBufferPointer { src in
        guard let channelData = buf.floatChannelData, let baseAddr = src.baseAddress
        else { return }
        channelData[0].update(from: baseAddr, count: samples.count)
    }
    engine.append(buf)

    let result = await engine.finishAndTranscribe() ?? ""
    let elapsed = ContinuousClock.now - start
    let ms =
        Int(elapsed.components.seconds) * 1_000
        + Int(elapsed.components.attoseconds / 1_000_000_000_000_000)

    let similarity = levenshteinRatio(result.lowercased(), fixture.expectedText.lowercased())
    let passed = similarity >= 0.85

    return TestResult(
        fixture: fixture.name,
        mode: "batch",
        passed: passed,
        similarity: similarity,
        resultText: result,
        expectedText: fixture.expectedText,
        durationMs: ms,
        detail: "\(String(format: "%.0f", similarity * 100))% match, \(ms)ms"
    )
}

private func runStreamingTest(engine: WhisperKitEngine, fixture: Fixture) async -> TestResult {
    let url = URL(fileURLWithPath: fixture.wavPath)
    guard let audioFile = try? AVAudioFile(forReading: url) else {
        return TestResult(
            fixture: fixture.name,
            mode: "stream",
            passed: false,
            similarity: 0,
            resultText: "",
            expectedText: fixture.expectedText,
            durationMs: 0,
            detail: "Could not read WAV"
        )
    }

    let fileFormat = audioFile.processingFormat
    let totalFrames = AVAudioFrameCount(audioFile.length)
    let fileSampleRate = fileFormat.sampleRate
    let tapBufferSize: AVAudioFrameCount = 1_024

    try? engine.prepare(contextualStrings: [])

    let sessionStart = ContinuousClock.now
    var framesDelivered: AVAudioFrameCount = 0

    // Feed chunks at real-time pace, matching the mic tap
    while framesDelivered < totalFrames {
        let remaining = totalFrames - framesDelivered
        let thisChunk = min(tapBufferSize, remaining)

        guard let buf = AVAudioPCMBuffer(pcmFormat: fileFormat, frameCapacity: thisChunk) else {
            break
        }
        do {
            try audioFile.read(into: buf, frameCount: thisChunk)
        } catch { break }
        engine.append(buf)
        framesDelivered += thisChunk

        let targetTime = Double(framesDelivered) / fileSampleRate
        let wallTime = ContinuousClock.now - sessionStart
        let wallSec =
            Double(wallTime.components.seconds)
            + Double(wallTime.components.attoseconds) / 1e18
        let sleepNeeded = targetTime - wallSec
        if sleepNeeded > 0.001 {
            try? await Task.sleep(nanoseconds: UInt64(sleepNeeded * 1_000_000_000))
        }
    }

    let releaseStart = ContinuousClock.now
    let result = await engine.finishAndTranscribe() ?? ""
    let releaseElapsed = ContinuousClock.now - releaseStart
    let releaseMs =
        Int(releaseElapsed.components.seconds) * 1_000
        + Int(releaseElapsed.components.attoseconds / 1_000_000_000_000_000)

    let totalElapsed = ContinuousClock.now - sessionStart
    let totalMs =
        Int(totalElapsed.components.seconds) * 1_000
        + Int(totalElapsed.components.attoseconds / 1_000_000_000_000_000)

    let similarity = levenshteinRatio(result.lowercased(), fixture.expectedText.lowercased())
    let passed = similarity >= 0.85

    return TestResult(
        fixture: fixture.name,
        mode: "stream",
        passed: passed,
        similarity: similarity,
        resultText: result,
        expectedText: fixture.expectedText,
        durationMs: releaseMs,
        detail: "\(String(format: "%.0f", similarity * 100))% match, \(releaseMs)ms post-release"
    )
}

// MARK: - Main

// MARK: - Entry Point

private func runTestSuite(
    label: String,
    engine: WhisperKitEngine,
    fixtures: [Fixture],
    testRunner: (WhisperKitEngine, Fixture) async -> TestResult
) async -> [TestResult] {
    print("--- \(label) ---\n")
    var results: [TestResult] = []
    for fixture in fixtures {
        let result = await testRunner(engine, fixture)
        results.append(result)
        let icon = result.passed ? "+" : "x"
        print("  \(icon) \(fixture.name): \(result.detail)")
        if !result.passed {
            print("    Expected: \"\(result.expectedText.prefix(80))...\"")
            print("    Got:      \"\(result.resultText.prefix(80))...\"")
        }
    }
    print("")
    return results
}

private func printSummary(_ results: [TestResult]) {
    let passed = results.filter(\.passed).count
    let total = results.count
    let allPassed = passed == total

    print("=== \(allPassed ? "ALL PASSED" : "FAILURES DETECTED") ===")
    print("  \(passed)/\(total) tests passed")

    if !allPassed {
        print("\nFailed tests:")
        for res in results where !res.passed {
            print("  x \(res.fixture) (\(res.mode)): \(res.detail)")
        }
    }
}

@MainActor
internal func runTranscriptionTests() async -> Int32 {
    let args = CommandLine.arguments.dropFirst()
    let filter = args.first

    let shouldRunBatch = filter == nil || filter == "batch"
    let shouldRunStreaming = filter == nil || filter == "streaming"
    let fixtureFilter: String? = (filter != "batch" && filter != "streaming") ? filter : nil

    print("=== Transcription Regression Tests ===\n")

    let fixtures: [Fixture]
    do {
        fixtures = try loadFixtures(filter: fixtureFilter)
    } catch {
        print("ERROR: \(error)")
        return 1
    }

    guard !fixtures.isEmpty else {
        print("No fixtures to test.")
        print("Capture one with: ./scripts/capture-fixture.sh <name>")
        return 1
    }

    print("Fixtures: \(fixtures.count)")
    for fix in fixtures {
        print("  \(fix.name) (\(String(format: "%.1f", fix.audioDuration))s)")
    }

    let engine: WhisperKitEngine
    do {
        print("\nLoading WhisperKit engine...")
        engine = try await loadEngine()
        print("Engine loaded.\n")
    } catch {
        print("ERROR: \(error)")
        return 1
    }

    try? engine.prepare(contextualStrings: [])
    _ = await engine.finishAndTranscribe()

    var results: [TestResult] = []

    if shouldRunBatch {
        results += await runTestSuite(
            label: "Batch Decode Tests",
            engine: engine,
            fixtures: fixtures,
            testRunner: runBatchTest
        )
    }

    if shouldRunStreaming {
        // Each streaming test gets a fresh engine to prevent state leaking
        // between fixtures (scheduler tasks, previousSchedulerTask, session state).
        // The WhisperKit pipeline (model) is shared — only the engine wrapper is new.
        print("--- Streaming Pipeline Tests ---\n")
        for fixture in fixtures {
            let freshEngine = engine.freshEngine()
            let result = await runStreamingTest(engine: freshEngine, fixture: fixture)
            results.append(result)
            let icon = result.passed ? "+" : "x"
            print("  \(icon) \(fixture.name): \(result.detail)")
            if !result.passed {
                print("    Expected: \"\(result.expectedText.prefix(80))...\"")
                print("    Got:      \"\(result.resultText.prefix(80))...\"")
            }
        }
        print("")
    }

    printSummary(results)
    let allPassed = results.filter(\.passed).count == results.count
    return allPassed ? 0 : 1
}

internal let exitCode: Int32 = await runTranscriptionTests()

exit(exitCode)

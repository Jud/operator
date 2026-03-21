import AVFoundation
import OperatorCore
import WhisperKit

// Manifest-driven replay tests. These feed audio at exact sample counts
// from production session logs and verify the streaming pipeline
// produces correct output.
//
// Unlike the real-time streaming tests (which are subject to timing
// variance), these use ManifestReplayScheduler to control exactly
// when background transcriptions fire.

// MARK: - Helpers

private func loadEngine() async throws -> WhisperKitEngine {
    let variant = WhisperKitModelManager.defaultModel
    guard let modelPath = WhisperKitModelManager.modelPath(variant)
    else {
        throw NSError(domain: "ManifestReplayTests", code: 1)
    }
    return try await WhisperKitEngine.create(modelFolder: modelPath)
}

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
        throw NSError(domain: "ManifestReplayTests", code: 2)
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
        throw NSError(domain: "ManifestReplayTests", code: 3)
    }

    guard let converter = AVAudioConverter(from: buffer.format, to: targetFormat)
    else {
        throw NSError(domain: "ManifestReplayTests", code: 4)
    }

    let ratio = targetRate / audioFile.processingFormat.sampleRate
    let outFrames = AVAudioFrameCount(Double(frameCount) * ratio) + 1
    guard let outBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outFrames)
    else {
        throw NSError(domain: "ManifestReplayTests", code: 5)
    }

    var error: NSError?
    nonisolated(unsafe) var consumed = false
    nonisolated(unsafe) let inputBuffer = buffer
    converter.convert(to: outBuffer, error: &error) { _, outStatus in
        if consumed {
            outStatus.pointee = .endOfStream
            return nil
        }
        consumed = true
        outStatus.pointee = .haveData
        return inputBuffer
    }

    if let error { throw error }

    guard let channelData = outBuffer.floatChannelData?[0]
    else {
        throw NSError(domain: "ManifestReplayTests", code: 6)
    }
    let count = Int(outBuffer.frameLength)
    return (Array(UnsafeBufferPointer(start: channelData, count: count)), 16_000)
}

private func levenshteinRatio(_ lhs: String, _ rhs: String) -> Double {
    let left = Array(lhs)
    let right = Array(rhs)
    let leftLen = left.count
    let rightLen = right.count
    if leftLen == 0, rightLen == 0 {
        return 1.0
    }
    if leftLen == 0 || rightLen == 0 {
        return 0.0
    }

    var prev = Array(0...rightLen)
    var curr = [Int](repeating: 0, count: rightLen + 1)

    for idx in 1...leftLen {
        curr[0] = idx
        for jdx in 1...rightLen {
            let cost = left[idx - 1] == right[jdx - 1] ? 0 : 1
            curr[jdx] = min(prev[jdx] + 1, curr[jdx - 1] + 1, prev[jdx - 1] + cost)
        }
        swap(&prev, &curr)
    }

    return 1.0 - Double(prev[rightLen]) / Double(max(leftLen, rightLen))
}

// MARK: - Test Runner

@MainActor
private func runManifestReplayTest(
    fixtureName: String,
    manifest: [ManifestReplayScheduler.Entry],
    baseEngine: WhisperKitEngine
) async -> (passed: Bool, similarity: Double, result: String, expected: String) {
    let appSupport =
        FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first ?? FileManager.default.homeDirectoryForCurrentUser
    let fixtureDir = appSupport.appendingPathComponent("Operator/test-fixtures")

    let wavPath = fixtureDir.appendingPathComponent("\(fixtureName).wav").path
    let expectedPath = fixtureDir.appendingPathComponent("\(fixtureName).expected.txt")

    guard
        let expectedText = try? String(contentsOf: expectedPath, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    else {
        print("  ERROR: Could not load expected text for \(fixtureName)")
        return (false, 0, "", "")
    }

    guard let audioFile = try? AVAudioFile(forReading: URL(fileURLWithPath: wavPath))
    else {
        print("  ERROR: Could not load WAV for \(fixtureName)")
        return (false, 0, "", expectedText)
    }

    let fileFormat = audioFile.processingFormat
    let totalFrames = AVAudioFrameCount(audioFile.length)
    let fileSampleRate = fileFormat.sampleRate
    let tapBufferSize: AVAudioFrameCount = 1_024

    // Create engine with manifest scheduler — shares WhisperKit pipeline
    let scheduler = ManifestReplayScheduler(entries: manifest)
    let engine = baseEngine.freshEngine(scheduler: scheduler)

    try? engine.prepare(contextualStrings: [])

    let sessionStart = ContinuousClock.now
    var framesDelivered: AVAudioFrameCount = 0

    // Feed audio at real-time pace
    while framesDelivered < totalFrames {
        let remaining = totalFrames - framesDelivered
        let thisChunk = min(tapBufferSize, remaining)

        guard let buf = AVAudioPCMBuffer(pcmFormat: fileFormat, frameCapacity: thisChunk)
        else { break }
        do {
            try audioFile.read(into: buf, frameCount: thisChunk)
        } catch { break }
        engine.append(buf)
        framesDelivered += thisChunk

        let targetTime = Double(framesDelivered) / fileSampleRate
        let wallTime = ContinuousClock.now - sessionStart
        let wallSec = Double(wallTime.components.seconds) + Double(wallTime.components.attoseconds) / 1e18
        let sleepNeeded = targetTime - wallSec
        if sleepNeeded > 0.001 {
            try? await Task.sleep(nanoseconds: UInt64(sleepNeeded * 1_000_000_000))
        }
    }

    let result = await engine.finishAndTranscribe() ?? ""
    let similarity = levenshteinRatio(result.lowercased(), expectedText.lowercased())

    return (similarity >= 0.85, similarity, result, expectedText)
}

// MARK: - Entry Point

/// Run manifest replay tests if any exist, otherwise skip.
@MainActor
internal func runManifestTests(baseEngine: WhisperKitEngine) async -> [String] {
    // For now, hardcoded manifests from known production failures.
    // Future: load from .manifest.json files alongside fixtures.

    var failures: [String] = []

    // short-utterance-loss: 1.8s audio, background loop fires once at ~1.5s
    // (24,000 samples at 16kHz), then key release.
    // Production session: background returned 3 words but first pass = miss.
    // Final tail decode returned only "failed." instead of "The CI run failed."
    let shortUtteranceManifest = [
        ManifestReplayScheduler.Entry(sampleCount: 24_000)  // ~1.5s
    ]

    print("--- Manifest Replay Tests ---\n")

    let shortResult = await runManifestReplayTest(
        fixtureName: "short-utterance-loss",
        manifest: shortUtteranceManifest,
        baseEngine: baseEngine
    )
    let pct = Int(shortResult.similarity * 100)
    let icon = shortResult.passed ? "+" : "x"
    print("  \(icon) short-utterance-loss (manifest): \(pct)% match")
    if !shortResult.passed {
        print("    Expected: \"\(shortResult.expected.prefix(80))...\"")
        print("    Got:      \"\(shortResult.result.prefix(80))...\"")
        failures.append("short-utterance-loss (manifest): \(pct)%")
    }
    print("")

    return failures
}

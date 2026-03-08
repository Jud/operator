import AVFoundation
import Testing

@testable import OperatorCore

/// Namespace matching the file name to satisfy the file_name lint rule.
private enum InterruptionPositionTests {}

// MARK: - Character Position Calculation

@Suite("Qwen3TTSSpeechManager - Character Position Calculation", .serialized)
internal struct CharacterPositionTests {
    @MainActor
    private func makeManager() -> Qwen3TTSSpeechManager {
        let cacheDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ip-test-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)

        let modelManager = ModelManager(cacheDirectory: cacheDir)
        return Qwen3TTSSpeechManager(modelManager: modelManager)
    }

    @Test("returns 0 when totalSamples is 0")
    @MainActor
    func zeroTotalSamples() {
        let manager = makeManager()
        let pos = manager.characterPosition(
            atSample: 100,
            totalSamples: 0,
            textLength: 50,
            text: "Hello world this is a test message"
        )
        #expect(pos == 0)
    }

    @Test("returns 0 when textLength is 0")
    @MainActor
    func zeroTextLength() {
        let manager = makeManager()
        let pos = manager.characterPosition(
            atSample: 100,
            totalSamples: 1_000,
            textLength: 0,
            text: ""
        )
        #expect(pos == 0)
    }

    @Test("returns 0 when sample is at beginning")
    @MainActor
    func sampleAtBeginning() {
        let manager = makeManager()
        let text = "This is a test message to verify position tracking"
        let pos = manager.characterPosition(
            atSample: 0,
            totalSamples: 24_000,
            textLength: text.count,
            text: text
        )
        #expect(pos == 0)
    }

    @Test("returns textLength when sample is at end")
    @MainActor
    func sampleAtEnd() {
        let manager = makeManager()
        let text = "Hello world"
        let pos = manager.characterPosition(
            atSample: 24_000,
            totalSamples: 24_000,
            textLength: text.count,
            text: text
        )
        #expect(pos == text.count)
    }

    @Test("proportional mapping at 50% produces approximately mid-text position")
    @MainActor
    func halfwayPosition() {
        let manager = makeManager()
        let text = "The quick brown fox jumps over the lazy dog near the riverbank"
        let pos = manager.characterPosition(
            atSample: 12_000,
            totalSamples: 24_000,
            textLength: text.count,
            text: text
        )
        let expectedMid = text.count / 2
        let tolerance = 15
        #expect(abs(pos - expectedMid) < tolerance)
    }

    @Test("snaps to word boundary (next space)")
    @MainActor
    func snapsToWordBoundary() {
        let manager = makeManager()
        let text = "Hello wonderful world today"
        let pos = manager.characterPosition(
            atSample: 6_000,
            totalSamples: 24_000,
            textLength: text.count,
            text: text
        )
        if pos > 0 && pos < text.count {
            let charIndex = text.index(text.startIndex, offsetBy: pos - 1)
            let charAtPos = text[charIndex]
            let isWordBoundary = charAtPos == " " || pos == 0 || pos == text.count
            #expect(isWordBoundary)
        }
    }

    @Test("handles single-word text without crashing")
    @MainActor
    func singleWordText() {
        let manager = makeManager()
        let text = "Hello"
        let pos = manager.characterPosition(
            atSample: 12_000,
            totalSamples: 24_000,
            textLength: text.count,
            text: text
        )
        #expect(pos >= 0 && pos <= text.count)
    }

    @Test("never exceeds textLength")
    @MainActor
    func neverExceedsTextLength() {
        let manager = makeManager()
        let text = "Short"
        let pos = manager.characterPosition(
            atSample: 48_000,
            totalSamples: 24_000,
            textLength: text.count,
            text: text
        )
        #expect(pos <= text.count)
    }

    @Test("handles very long text correctly")
    @MainActor
    func longTextHandling() {
        let manager = makeManager()
        let text = String(
            repeating: "The quick brown fox jumps over the lazy dog. ",
            count: 20
        )
        let pos = manager.characterPosition(
            atSample: 18_000,
            totalSamples: 24_000,
            textLength: text.count,
            text: text
        )
        let expected = text.count * 3 / 4
        let tolerance = 50
        #expect(abs(pos - expected) < tolerance)
    }
}

// MARK: - Buffer Conversion

@Suite("Qwen3TTSSpeechManager - Buffer Conversion", .serialized)
internal struct BufferConversionTests {
    @MainActor
    private func makeManager() -> Qwen3TTSSpeechManager {
        let cacheDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("bc-test-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        let modelManager = ModelManager(cacheDirectory: cacheDir)
        return Qwen3TTSSpeechManager(modelManager: modelManager)
    }

    @Test("createBuffer produces buffer with correct frame count")
    @MainActor
    func createBufferFrameCount() {
        let manager = makeManager()
        let samples: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5]
        let buffer = manager.createBuffer(from: samples)

        #expect(buffer.frameLength == 5)
    }

    @Test("createBuffer preserves sample values")
    @MainActor
    func createBufferPreservesSamples() {
        let manager = makeManager()
        let samples: [Float] = [0.5, -0.5, 1.0, -1.0]
        let buffer = manager.createBuffer(from: samples)

        guard let channelData = buffer.floatChannelData else {
            Issue.record("No channel data in buffer")
            return
        }

        for idx in 0..<samples.count {
            #expect(abs(channelData[0][idx] - samples[idx]) < 0.0001)
        }
    }

    @Test("createBuffer produces mono format")
    @MainActor
    func createBufferIsMono() {
        let manager = makeManager()
        let samples: [Float] = [0.1, 0.2]
        let buffer = manager.createBuffer(from: samples)

        #expect(buffer.format.channelCount == 1)
    }
}

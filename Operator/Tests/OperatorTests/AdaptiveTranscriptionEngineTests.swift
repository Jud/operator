import AVFoundation
import Foundation
import Testing

@testable import OperatorCore

/// Namespace matching the file name to satisfy the file_name lint rule.
private enum AdaptiveTranscriptionEngineTests {}

/// Thread-safe accumulator for fallback callback messages.
private actor FallbackCapture {
    private(set) var messages: [String] = []
    var notified: Bool { !messages.isEmpty }
    var count: Int { messages.count }
    var lastMessage: String? { messages.last }

    func append(_ message: String) { messages.append(message) }
}

private actor STTWarmupCapture {
    private(set) var count = 0
    var isEmpty: Bool { count < 1 }

    func record() {
        count += 1
    }
}

private struct StubLocalTranscriptionEngine: TranscriptionEngine, LocalModelResident {
    let isCachedHandler: @Sendable () async -> Bool
    let prewarmHandler: @Sendable () async -> Void
    let finishHandler: @Sendable () async -> String?

    func isCachedLocally() async -> Bool {
        await isCachedHandler()
    }

    func prewarm() async {
        await prewarmHandler()
    }

    func prepare() throws {}

    func append(_ buffer: AVAudioPCMBuffer) {}

    func finishAndTranscribe() async -> String? {
        await finishHandler()
    }

    func cancel() {}
}

// MARK: - Engine Switching

@Suite("AdaptiveTranscriptionEngine - Engine Switching")
internal struct AdaptiveSTTSwitchingTests {
    private func makeEngine(
        preferLocal: Bool = true
    ) async -> (
        AdaptiveTranscriptionEngine, ModelManager
    ) {
        let cacheDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ate-test-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)

        let modelManager = ModelManager(cacheDirectory: cacheDir)
        let localEngine = ParakeetEngine(modelManager: modelManager)
        let fallbackEngine = AppleSpeechEngine()

        let adaptive = AdaptiveTranscriptionEngine(
            localEngine: localEngine,
            fallbackEngine: fallbackEngine,
            modelManager: modelManager,
            preferLocal: preferLocal
        )
        return (adaptive, modelManager)
    }

    @Test("isUsingLocal reflects preference when no fallback has occurred")
    func isUsingLocalReflectsPreference() async {
        let (engine, _) = await makeEngine(preferLocal: true)
        #expect(engine.isUsingLocal == true)

        engine.setUseLocal(false)
        #expect(engine.isUsingLocal == false)
    }

    @Test("setUseLocal(true) re-enables local after manual switch to Apple")
    func setUseLocalReEnables() async {
        let (engine, _) = await makeEngine(preferLocal: true)
        engine.setUseLocal(false)
        #expect(engine.isUsingLocal == false)

        engine.setUseLocal(true)
        #expect(engine.isUsingLocal == true)
    }

    @Test("preferLocal false starts with Apple engine active")
    func preferLocalFalseStartsWithApple() async {
        let (engine, _) = await makeEngine(preferLocal: false)
        #expect(engine.isUsingLocal == false)
    }

    @Test("prepare succeeds when useLocal is true (Parakeet prepare never throws)")
    func prepareSucceedsWithLocal() async throws {
        let (engine, _) = await makeEngine(preferLocal: true)
        try engine.prepare()
    }

    @Test("prepare succeeds when useLocal is false")
    func prepareSucceedsWithFallback() async throws {
        let (engine, _) = await makeEngine(preferLocal: false)
        try engine.prepare()
    }

    @Test("cancel does not throw")
    func cancelDoesNotThrow() async {
        let (engine, _) = await makeEngine()
        engine.cancel()
    }

    @Test("prewarmSelectedLocalModel warms cached local STT engine")
    func prewarmSelectedLocalModelWarmsCachedLocalEngine() async {
        let capture = STTWarmupCapture()
        let cacheDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ate-test-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)

        let modelManager = ModelManager(cacheDirectory: cacheDir)
        let engine = AdaptiveTranscriptionEngine(
            localEngine: StubLocalTranscriptionEngine(
                isCachedHandler: { true },
                prewarmHandler: { await capture.record() },
                finishHandler: { nil }
            ),
            fallbackEngine: AppleSpeechEngine(),
            modelManager: modelManager,
            preferLocal: true
        )

        await engine.prewarmSelectedLocalModel()

        #expect(await capture.count == 1)
    }

    @Test("prewarmSelectedLocalModel skips uncached local STT engine")
    func prewarmSelectedLocalModelSkipsUncachedLocalEngine() async {
        let capture = STTWarmupCapture()
        let cacheDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ate-test-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)

        let modelManager = ModelManager(cacheDirectory: cacheDir)
        let engine = AdaptiveTranscriptionEngine(
            localEngine: StubLocalTranscriptionEngine(
                isCachedHandler: { false },
                prewarmHandler: { await capture.record() },
                finishHandler: { nil }
            ),
            fallbackEngine: AppleSpeechEngine(),
            modelManager: modelManager,
            preferLocal: true
        )

        await engine.prewarmSelectedLocalModel()

        #expect(await capture.isEmpty)
    }
}

// MARK: - Fallback Behavior

@Suite("AdaptiveTranscriptionEngine - Fallback Behavior")
internal struct AdaptiveSTTFallbackTests {
    private func makeEngine(
        preferLocal: Bool = true
    ) async -> (
        AdaptiveTranscriptionEngine, ModelManager
    ) {
        let cacheDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ate-test-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)

        let modelManager = ModelManager(cacheDirectory: cacheDir)
        let localEngine = ParakeetEngine(modelManager: modelManager)
        let fallbackEngine = AppleSpeechEngine()

        let adaptive = AdaptiveTranscriptionEngine(
            localEngine: localEngine,
            fallbackEngine: fallbackEngine,
            modelManager: modelManager,
            preferLocal: preferLocal
        )
        return (adaptive, modelManager)
    }

    @Test("fallback triggers when local finishAndTranscribe returns nil with no audio")
    func fallbackTriggersOnNilTranscription() async throws {
        let (engine, _) = await makeEngine(preferLocal: true)

        let capture = FallbackCapture()
        engine.onFallback = { message in
            Task { await capture.append(message) }
        }

        try engine.prepare()
        let result = await engine.finishAndTranscribe()
        try? await Task.sleep(for: .milliseconds(50))

        #expect(result == nil)
        #expect(engine.isUsingLocal == false)
        #expect(await capture.notified == true)
    }

    @Test("sticky fallback stays on Apple until setUseLocal(true)")
    func stickyFallback() async throws {
        let (engine, _) = await makeEngine(preferLocal: true)

        let capture = FallbackCapture()
        engine.onFallback = { message in
            Task { await capture.append(message) }
        }

        try engine.prepare()
        _ = await engine.finishAndTranscribe()
        #expect(engine.isUsingLocal == false)

        // Subsequent prepare should still use fallback
        try engine.prepare()
        #expect(engine.isUsingLocal == false)
    }

    @Test("setUseLocal(true) resets fallback state")
    func setUseLocalResetsFallback() async throws {
        let (engine, _) = await makeEngine(preferLocal: true)

        let capture = FallbackCapture()
        engine.onFallback = { message in
            Task { await capture.append(message) }
        }

        try engine.prepare()
        _ = await engine.finishAndTranscribe()
        #expect(engine.isUsingLocal == false)

        engine.setUseLocal(true)
        #expect(engine.isUsingLocal == true)
    }

    @Test("onFallback callback receives notification message")
    func onFallbackCallbackMessage() async throws {
        let (engine, _) = await makeEngine(preferLocal: true)

        let capture = FallbackCapture()
        engine.onFallback = { message in
            Task { await capture.append(message) }
        }

        try engine.prepare()
        _ = await engine.finishAndTranscribe()
        try? await Task.sleep(for: .milliseconds(50))

        let lastMessage = await capture.lastMessage
        #expect(lastMessage?.contains("Falling back") == true)
        #expect(lastMessage?.contains("Apple") == true)
    }

    @Test("onFallback not called when useLocal is false")
    func noFallbackCallbackWhenApple() async {
        let (engine, _) = await makeEngine(preferLocal: false)

        let capture = FallbackCapture()
        engine.onFallback = { message in
            Task { await capture.append(message) }
        }

        let result = await engine.finishAndTranscribe()
        try? await Task.sleep(for: .milliseconds(50))

        #expect(result == nil)
        #expect(await capture.notified == false)
    }

    @Test("finishAndTranscribe returns nil when no audio was appended regardless of engine")
    func noAudioReturnsNil() async throws {
        let (engine, _) = await makeEngine(preferLocal: false)
        try engine.prepare()
        let result = await engine.finishAndTranscribe()
        #expect(result == nil)
    }
}

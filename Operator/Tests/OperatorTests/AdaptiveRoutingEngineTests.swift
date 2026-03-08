import Foundation
import Testing

@testable import OperatorCore

/// Namespace matching the file name to satisfy the file_name lint rule.
private enum AdaptiveRoutingEngineTests {}

/// Thread-safe accumulator for fallback callback messages.
private actor RoutingFallbackCapture {
    private(set) var messages: [String] = []
    var notified: Bool { !messages.isEmpty }
    var count: Int { messages.count }
    var lastMessage: String? { messages.last }

    func append(_ message: String) { messages.append(message) }
}

// MARK: - Engine Switching

@Suite("AdaptiveRoutingEngine - Engine Switching")
internal struct AdaptiveRoutingSwitchingTests {
    private func makeEngine(
        preferLocal: Bool = true
    ) async -> (
        AdaptiveRoutingEngine, ModelManager
    ) {
        let cacheDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("are-test-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)

        let modelManager = ModelManager(cacheDirectory: cacheDir)
        let localEngine = MLXRoutingEngine(modelManager: modelManager)
        let fallbackEngine = ClaudePipeRoutingEngine()

        let adaptive = AdaptiveRoutingEngine(
            localEngine: localEngine,
            modelManager: modelManager,
            fallbackEngine: fallbackEngine,
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

    @Test("setUseLocal toggles engine preference")
    func setUseLocalToggles() async {
        let (engine, _) = await makeEngine(preferLocal: false)
        #expect(engine.isUsingLocal == false)

        engine.setUseLocal(true)
        #expect(engine.isUsingLocal == true)

        engine.setUseLocal(false)
        #expect(engine.isUsingLocal == false)
    }

    @Test("preferLocal false starts with Claude CLI active")
    func preferLocalFalseStartsWithClaude() async {
        let (engine, _) = await makeEngine(preferLocal: false)
        #expect(engine.isUsingLocal == false)
    }
}

// MARK: - Fallback Behavior

@Suite("AdaptiveRoutingEngine - Fallback Behavior")
internal struct AdaptiveRoutingFallbackTests {
    private func makeEngine(
        preferLocal: Bool = true
    ) async -> (
        AdaptiveRoutingEngine, ModelManager
    ) {
        let cacheDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("are-test-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)

        let modelManager = ModelManager(cacheDirectory: cacheDir)
        let localEngine = MLXRoutingEngine(modelManager: modelManager)
        let fallbackEngine = ClaudePipeRoutingEngine()

        let adaptive = AdaptiveRoutingEngine(
            localEngine: localEngine,
            modelManager: modelManager,
            fallbackEngine: fallbackEngine,
            preferLocal: preferLocal
        )
        return (adaptive, modelManager)
    }

    @Test("triggerFallback sets didFallBack and invokes onFallback callback")
    func triggerFallbackSetsState() async {
        let (engine, _) = await makeEngine(preferLocal: true)

        let capture = RoutingFallbackCapture()
        engine.onFallback = { message in
            Task { await capture.append(message) }
        }

        engine.triggerFallback()
        try? await Task.sleep(for: .milliseconds(50))

        #expect(engine.isUsingLocal == false)
        #expect(await capture.notified == true)
        #expect(await capture.lastMessage?.contains("Claude CLI") == true)
    }

    @Test("triggerFallback is idempotent")
    func triggerFallbackIdempotent() async {
        let (engine, _) = await makeEngine(preferLocal: true)

        let capture = RoutingFallbackCapture()
        engine.onFallback = { message in
            Task { await capture.append(message) }
        }

        engine.triggerFallback()
        engine.triggerFallback()
        engine.triggerFallback()
        try? await Task.sleep(for: .milliseconds(50))

        #expect(await capture.count == 1)
    }

    @Test("sticky fallback stays on Claude CLI until setUseLocal(true)")
    func stickyFallback() async {
        let (engine, _) = await makeEngine(preferLocal: true)

        let capture = RoutingFallbackCapture()
        engine.onFallback = { message in
            Task { await capture.append(message) }
        }

        engine.triggerFallback()
        #expect(engine.isUsingLocal == false)

        // Still false
        #expect(engine.isUsingLocal == false)
    }

    @Test("setUseLocal(true) resets fallback state after triggerFallback")
    func setUseLocalResetsFallback() async {
        let (engine, _) = await makeEngine(preferLocal: true)

        let capture = RoutingFallbackCapture()
        engine.onFallback = { message in
            Task { await capture.append(message) }
        }

        engine.triggerFallback()
        #expect(engine.isUsingLocal == false)

        engine.setUseLocal(true)
        #expect(engine.isUsingLocal == true)
    }

    @Test("onFallback callback receives descriptive message about Claude CLI")
    func fallbackMessageContent() async {
        let (engine, _) = await makeEngine(preferLocal: true)

        let capture = RoutingFallbackCapture()
        engine.onFallback = { message in
            Task { await capture.append(message) }
        }

        engine.triggerFallback()
        try? await Task.sleep(for: .milliseconds(50))

        let msg = await capture.lastMessage
        #expect(msg?.contains("Claude CLI routing") == true)
        #expect(msg?.contains("Local model unavailable") == true)
    }

    @Test(
        "run falls back to Claude CLI when local engine throws",
        .timeLimit(.minutes(2))
    )
    func runFallsBackOnLocalFailure() async throws {
        let (engine, _) = await makeEngine(preferLocal: true)

        let capture = RoutingFallbackCapture()
        engine.onFallback = { message in
            Task { await capture.append(message) }
        }

        // The local MLXRoutingEngine will fail (model not downloaded).
        // The fallback ClaudePipeRoutingEngine will also fail (no claude CLI).
        // The key assertion is that the fallback was triggered.
        do {
            _ = try await engine.run(prompt: "test prompt", timeout: 2.0)
        } catch {
            // Both engines failing in test environment is expected
        }
        try? await Task.sleep(for: .milliseconds(50))

        #expect(await capture.notified == true)
        #expect(engine.isUsingLocal == false)
    }

    @Test(
        "run uses fallback directly when useLocal is false, does not invoke onFallback",
        .timeLimit(.minutes(2))
    )
    func runUsesFallbackDirectly() async {
        let (engine, _) = await makeEngine(preferLocal: false)

        let capture = RoutingFallbackCapture()
        engine.onFallback = { message in
            Task { await capture.append(message) }
        }

        // When useLocal is false, goes directly to ClaudePipeRoutingEngine.
        // The onFallback callback should NOT be invoked.
        do {
            _ = try await engine.run(prompt: "test", timeout: 2.0)
        } catch {
            // Expected: claude CLI not available
        }
        try? await Task.sleep(for: .milliseconds(50))

        #expect(await capture.notified == false)
    }
}

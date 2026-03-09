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

private enum StubRoutingError: Error {
    case localFailed
    case fallbackFailed
}

private struct StubRoutingEngine: RoutingEngine {
    let isCachedHandler: @Sendable () async -> Bool
    let prewarmHandler: @Sendable () async -> Void
    let handler: @Sendable (String, TimeInterval) async throws -> [String: Any]

    func isCachedLocally() async -> Bool {
        await isCachedHandler()
    }

    func prewarm() async {
        await prewarmHandler()
    }

    func run(prompt: String, timeout: TimeInterval) async throws -> [String: Any] {
        try await handler(prompt, timeout)
    }
}

extension StubRoutingEngine: LocalModelResident {}

private actor RoutingWarmupCapture {
    private(set) var count = 0
    var isEmpty: Bool { count < 1 }

    func record() {
        count += 1
    }
}

// MARK: - Engine Switching

@Suite("AdaptiveRoutingEngine - Engine Switching")
internal struct AdaptiveRoutingSwitchingTests {
    private func makeEngine(
        preferLocal: Bool = true,
        localEngine: StubRoutingEngine = StubRoutingEngine(
            isCachedHandler: { true },
            prewarmHandler: {},
            handler: { _, _ in ["session": "local"] }
        ),
        fallbackEngine: StubRoutingEngine = StubRoutingEngine(
            isCachedHandler: { true },
            prewarmHandler: {},
            handler: { _, _ in ["session": "fallback"] }
        )
    ) -> AdaptiveRoutingEngine {
        let adaptive = AdaptiveRoutingEngine(
            localEngine: localEngine,
            fallbackEngine: fallbackEngine,
            preferLocal: preferLocal
        )
        return adaptive
    }

    @Test("isUsingLocal reflects preference when no fallback has occurred")
    func isUsingLocalReflectsPreference() async {
        let engine = makeEngine(preferLocal: true)
        #expect(engine.isUsingLocal == true)

        engine.setUseLocal(false)
        #expect(engine.isUsingLocal == false)
    }

    @Test("setUseLocal toggles engine preference")
    func setUseLocalToggles() async {
        let engine = makeEngine(preferLocal: false)
        #expect(engine.isUsingLocal == false)

        engine.setUseLocal(true)
        #expect(engine.isUsingLocal == true)

        engine.setUseLocal(false)
        #expect(engine.isUsingLocal == false)
    }

    @Test("preferLocal false starts with Claude CLI active")
    func preferLocalFalseStartsWithClaude() async {
        let engine = makeEngine(preferLocal: false)
        #expect(engine.isUsingLocal == false)
    }

    @Test("prewarmSelectedLocalModel warms cached local routing engine")
    func prewarmSelectedLocalModelWarmsCachedLocalEngine() async {
        let capture = RoutingWarmupCapture()
        let engine = makeEngine(
            preferLocal: true,
            localEngine: StubRoutingEngine(
                isCachedHandler: { true },
                prewarmHandler: { await capture.record() },
                handler: { _, _ in ["session": "local"] }
            )
        )

        await engine.prewarmSelectedLocalModel()

        #expect(await capture.count == 1)
    }

    @Test("prewarmSelectedLocalModel skips uncached local routing engine")
    func prewarmSelectedLocalModelSkipsUncachedLocalEngine() async {
        let capture = RoutingWarmupCapture()
        let engine = makeEngine(
            preferLocal: true,
            localEngine: StubRoutingEngine(
                isCachedHandler: { false },
                prewarmHandler: { await capture.record() },
                handler: { _, _ in ["session": "local"] }
            )
        )

        await engine.prewarmSelectedLocalModel()

        #expect(await capture.isEmpty)
    }
}

// MARK: - Fallback Behavior

@Suite("AdaptiveRoutingEngine - Fallback Behavior")
internal struct AdaptiveRoutingFallbackTests {
    private func makeEngine(
        preferLocal: Bool = true,
        localEngine: StubRoutingEngine = StubRoutingEngine(
            isCachedHandler: { true },
            prewarmHandler: {},
            handler: { _, _ in ["session": "local"] }
        ),
        fallbackEngine: StubRoutingEngine = StubRoutingEngine(
            isCachedHandler: { true },
            prewarmHandler: {},
            handler: { _, _ in ["session": "fallback"] }
        )
    ) -> AdaptiveRoutingEngine {
        let adaptive = AdaptiveRoutingEngine(
            localEngine: localEngine,
            fallbackEngine: fallbackEngine,
            preferLocal: preferLocal
        )
        return adaptive
    }

    @Test("triggerFallback sets didFallBack and invokes onFallback callback")
    func triggerFallbackSetsState() async {
        let engine = makeEngine(preferLocal: true)

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
        let engine = makeEngine(preferLocal: true)

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
        let engine = makeEngine(preferLocal: true)

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
        let engine = makeEngine(preferLocal: true)

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
        let engine = makeEngine(preferLocal: true)

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
        let engine = makeEngine(
            preferLocal: true,
            localEngine: StubRoutingEngine(
                isCachedHandler: { true },
                prewarmHandler: {},
                handler: { _, _ in throw StubRoutingError.localFailed }
            ),
            fallbackEngine: StubRoutingEngine(
                isCachedHandler: { true },
                prewarmHandler: {},
                handler: { _, _ in throw StubRoutingError.fallbackFailed }
            )
        )

        let capture = RoutingFallbackCapture()
        engine.onFallback = { message in
            Task { await capture.append(message) }
        }

        // The local engine fails immediately.
        // The fallback engine also fails.
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
        let engine = makeEngine(
            preferLocal: false,
            localEngine: StubRoutingEngine(
                isCachedHandler: { true },
                prewarmHandler: {},
                handler: { _, _ in throw StubRoutingError.localFailed }
            ),
            fallbackEngine: StubRoutingEngine(
                isCachedHandler: { true },
                prewarmHandler: {},
                handler: { _, _ in throw StubRoutingError.fallbackFailed }
            )
        )

        let capture = RoutingFallbackCapture()
        engine.onFallback = { message in
            Task { await capture.append(message) }
        }

        // When useLocal is false, goes directly to the fallback engine.
        // The onFallback callback should NOT be invoked.
        do {
            _ = try await engine.run(prompt: "test", timeout: 2.0)
        } catch {
            // Expected: fallback engine failure
        }
        try? await Task.sleep(for: .milliseconds(50))

        #expect(await capture.notified == false)
    }
}

import Foundation
import Testing
import os

@testable import OperatorCore

// MARK: - Stub Bridge

private final class StubBridge: TerminalBridge, @unchecked Sendable {
    var discoverResult: [DiscoveredSession] = []
    var discoverError: Error?

    func discoverSessions() async throws -> [DiscoveredSession] {
        if let error = discoverError { throw error }
        return discoverResult
    }

    func writeToSession(identifier: TerminalIdentifier, text: String) async throws -> Bool {
        true
    }
}

/// Thread-safe collector for removed session names in tests.
private final class RemovedNameCollector: Sendable {
    private let lock = OSAllocatedUnfairLock<[String]>(initialState: [])

    var names: [String] {
        lock.withLock { $0 }
    }

    func append(_ name: String) {
        lock.withLock { $0.append(name) }
    }
}

// MARK: - Helpers

private func makeItermSession(
    tty: String = "/dev/ttys001",
    name: String = "backend"
) -> DiscoveredSession {
    DiscoveredSession(
        id: tty,
        tty: tty,
        name: name,
        isProcessing: false,
        workingDirectory: "/tmp",
        jobName: nil,
        terminalType: .iterm
    )
}

private func makeGhosttySession(
    id: String = "ghost-1",
    name: String = "frontend"
) -> DiscoveredSession {
    DiscoveredSession(
        id: id,
        tty: "",
        name: name,
        isProcessing: false,
        workingDirectory: "/tmp",
        jobName: nil,
        terminalType: .ghostty
    )
}

private func makeRegistry() -> SessionRegistry {
    SessionRegistry(voiceManager: VoiceManager())
}

// MARK: - Tests

@Suite("SessionDiscoveryService - Per-Type Reconciliation")
internal struct SessionDiscoveryServiceTests {
    @Test("iTerm sessions survive when Ghostty discovery fails")
    func itermSessionsSurviveGhosttyFailure() async {
        let registry = makeRegistry()
        await registry.register(name: "backend", tty: "/dev/ttys001", cwd: "/tmp", context: nil)
        await registry.register(
            name: "frontend",
            identifier: .ghosttyTerminal("ghost-1"),
            cwd: "/tmp",
            context: nil
        )

        let itermBridge = StubBridge()
        itermBridge.discoverResult = [makeItermSession(tty: "/dev/ttys001")]

        let ghosttyBridge = StubBridge()
        ghosttyBridge.discoverError = TerminalBridgeError.terminalNotRunning(terminal: .ghostty)

        let composite = MultiTerminalBridge(itermBridge: itermBridge, ghosttyBridge: ghosttyBridge)
        let collector = RemovedNameCollector()

        let service = SessionDiscoveryService(
            terminalBridge: composite,
            registry: registry
        ) { name in
            collector.append(name)
        }

        await service.reconcileOnce()

        #expect(await registry.sessionCount == 2)
        #expect(await registry.session(byTTY: "/dev/ttys001") != nil)
        #expect(await registry.session(byIdentifier: .ghosttyTerminal("ghost-1")) != nil)
        #expect(collector.names.isEmpty)
    }

    @Test("Ghostty sessions survive when iTerm discovery fails")
    func ghosttySessionsSurviveItermFailure() async {
        let registry = makeRegistry()
        await registry.register(name: "backend", tty: "/dev/ttys001", cwd: "/tmp", context: nil)
        await registry.register(
            name: "frontend",
            identifier: .ghosttyTerminal("ghost-1"),
            cwd: "/tmp",
            context: nil
        )

        let itermBridge = StubBridge()
        itermBridge.discoverError = TerminalBridgeError.terminalNotRunning(terminal: .iterm)

        let ghosttyBridge = StubBridge()
        ghosttyBridge.discoverResult = [makeGhosttySession(id: "ghost-1")]

        let composite = MultiTerminalBridge(itermBridge: itermBridge, ghosttyBridge: ghosttyBridge)
        let collector = RemovedNameCollector()

        let service = SessionDiscoveryService(
            terminalBridge: composite,
            registry: registry
        ) { name in
            collector.append(name)
        }

        await service.reconcileOnce()

        #expect(await registry.sessionCount == 2)
        #expect(await registry.session(byTTY: "/dev/ttys001") != nil)
        #expect(await registry.session(byIdentifier: .ghosttyTerminal("ghost-1")) != nil)
        #expect(collector.names.isEmpty)
    }

    @Test("stale iTerm session removed when iTerm polled with zero sessions")
    func staleItermRemovedWhenPolledEmpty() async {
        let registry = makeRegistry()
        await registry.register(name: "backend", tty: "/dev/ttys001", cwd: "/tmp", context: nil)
        await registry.register(
            name: "frontend",
            identifier: .ghosttyTerminal("ghost-1"),
            cwd: "/tmp",
            context: nil
        )

        let itermBridge = StubBridge()
        itermBridge.discoverResult = []

        let ghosttyBridge = StubBridge()
        ghosttyBridge.discoverResult = [makeGhosttySession(id: "ghost-1")]

        let composite = MultiTerminalBridge(itermBridge: itermBridge, ghosttyBridge: ghosttyBridge)
        let collector = RemovedNameCollector()

        let service = SessionDiscoveryService(
            terminalBridge: composite,
            registry: registry
        ) { name in
            collector.append(name)
        }

        await service.reconcileOnce()

        #expect(await registry.sessionCount == 1)
        #expect(await registry.session(byTTY: "/dev/ttys001") == nil)
        #expect(await registry.session(byIdentifier: .ghosttyTerminal("ghost-1")) != nil)
        #expect(collector.names == ["backend"])
    }

    @Test("stale Ghostty session removed when Ghostty polled with zero sessions")
    func staleGhosttyRemovedWhenPolledEmpty() async {
        let registry = makeRegistry()
        await registry.register(name: "backend", tty: "/dev/ttys001", cwd: "/tmp", context: nil)
        await registry.register(
            name: "frontend",
            identifier: .ghosttyTerminal("ghost-1"),
            cwd: "/tmp",
            context: nil
        )

        let itermBridge = StubBridge()
        itermBridge.discoverResult = [makeItermSession(tty: "/dev/ttys001")]

        let ghosttyBridge = StubBridge()
        ghosttyBridge.discoverResult = []

        let composite = MultiTerminalBridge(itermBridge: itermBridge, ghosttyBridge: ghosttyBridge)
        let collector = RemovedNameCollector()

        let service = SessionDiscoveryService(
            terminalBridge: composite,
            registry: registry
        ) { name in
            collector.append(name)
        }

        await service.reconcileOnce()

        #expect(await registry.sessionCount == 1)
        #expect(await registry.session(byTTY: "/dev/ttys001") != nil)
        #expect(await registry.session(byIdentifier: .ghosttyTerminal("ghost-1")) == nil)
        #expect(collector.names == ["frontend"])
    }

    @Test("both bridges succeed, stale sessions of each type removed")
    func bothPolledStaleRemoved() async {
        let registry = makeRegistry()
        await registry.register(name: "alive-iterm", tty: "/dev/ttys001", cwd: "/tmp", context: nil)
        await registry.register(name: "stale-iterm", tty: "/dev/ttys002", cwd: "/tmp", context: nil)
        await registry.register(
            name: "alive-ghostty",
            identifier: .ghosttyTerminal("ghost-1"),
            cwd: "/tmp",
            context: nil
        )
        await registry.register(
            name: "stale-ghostty",
            identifier: .ghosttyTerminal("ghost-2"),
            cwd: "/tmp",
            context: nil
        )

        let itermBridge = StubBridge()
        itermBridge.discoverResult = [makeItermSession(tty: "/dev/ttys001")]

        let ghosttyBridge = StubBridge()
        ghosttyBridge.discoverResult = [makeGhosttySession(id: "ghost-1")]

        let composite = MultiTerminalBridge(itermBridge: itermBridge, ghosttyBridge: ghosttyBridge)
        let collector = RemovedNameCollector()

        let service = SessionDiscoveryService(
            terminalBridge: composite,
            registry: registry
        ) { name in
            collector.append(name)
        }

        await service.reconcileOnce()

        #expect(await registry.sessionCount == 2)
        #expect(await registry.session(named: "alive-iterm") != nil)
        #expect(await registry.session(named: "alive-ghostty") != nil)
        #expect(collector.names.sorted() == ["stale-ghostty", "stale-iterm"])
    }

    @Test("both bridges fail, all sessions survive")
    func bothFailAllSurvive() async {
        let registry = makeRegistry()
        await registry.register(name: "backend", tty: "/dev/ttys001", cwd: "/tmp", context: nil)
        await registry.register(
            name: "frontend",
            identifier: .ghosttyTerminal("ghost-1"),
            cwd: "/tmp",
            context: nil
        )

        let itermBridge = StubBridge()
        itermBridge.discoverError = TerminalBridgeError.terminalNotRunning(terminal: .iterm)

        let ghosttyBridge = StubBridge()
        ghosttyBridge.discoverError = TerminalBridgeError.terminalNotRunning(terminal: .ghostty)

        let composite = MultiTerminalBridge(itermBridge: itermBridge, ghosttyBridge: ghosttyBridge)
        let collector = RemovedNameCollector()

        let service = SessionDiscoveryService(
            terminalBridge: composite,
            registry: registry
        ) { name in
            collector.append(name)
        }

        await service.reconcileOnce()

        #expect(await registry.sessionCount == 2)
        #expect(collector.names.isEmpty)
    }
}

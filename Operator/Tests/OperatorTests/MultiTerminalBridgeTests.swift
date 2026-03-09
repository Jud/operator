import Foundation
import Testing

@testable import OperatorCore

// MARK: - Mock Sub-Bridges

private final class StubBridge: TerminalBridge, @unchecked Sendable {
    var discoverResult: [DiscoveredSession] = []
    var discoverError: Error?
    var writeResult: Bool = true
    var writeError: Error?
    var lastWrittenIdentifier: TerminalIdentifier?
    var lastWrittenText: String?

    func discoverSessions() async throws -> [DiscoveredSession] {
        if let error = discoverError { throw error }
        return discoverResult
    }

    func writeToSession(identifier: TerminalIdentifier, text: String) async throws -> Bool {
        lastWrittenIdentifier = identifier
        lastWrittenText = text
        if let error = writeError { throw error }
        return writeResult
    }
}

private func makeItermSession(id: String = "sess1", tty: String = "/dev/ttys001") -> DiscoveredSession {
    DiscoveredSession(
        id: id,
        tty: tty,
        name: "backend",
        isProcessing: false,
        workingDirectory: "/tmp",
        jobName: nil,
        terminalType: .iterm
    )
}

private func makeGhosttySession(id: String = "ghostty-1") -> DiscoveredSession {
    DiscoveredSession(
        id: id,
        tty: "",
        name: "frontend",
        isProcessing: false,
        workingDirectory: "/tmp",
        jobName: nil,
        terminalType: .ghostty
    )
}

// MARK: - Tests

@Suite("MultiTerminalBridge")
struct MultiTerminalBridgeTests {

    // MARK: - Discovery Aggregation

    @Test("discovery aggregates sessions from both bridges")
    func discoveryAggregatesBoth() async throws {
        let iterm = StubBridge()
        iterm.discoverResult = [makeItermSession()]
        let ghostty = StubBridge()
        ghostty.discoverResult = [makeGhosttySession()]

        let composite = MultiTerminalBridge(itermBridge: iterm, ghosttyBridge: ghostty)
        let sessions = try await composite.discoverSessions()

        #expect(sessions.count == 2)
        #expect(sessions.contains { $0.terminalType == .iterm })
        #expect(sessions.contains { $0.terminalType == .ghostty })
    }

    @Test("discovery returns iTerm sessions when Ghostty fails")
    func discoveryItermOnlyWhenGhosttyFails() async throws {
        let iterm = StubBridge()
        iterm.discoverResult = [makeItermSession()]
        let ghostty = StubBridge()
        ghostty.discoverError = GhosttyBridgeError.ghosttyNotRunning

        let composite = MultiTerminalBridge(itermBridge: iterm, ghosttyBridge: ghostty)
        let sessions = try await composite.discoverSessions()

        #expect(sessions.count == 1)
        #expect(sessions[0].terminalType == .iterm)
    }

    @Test("discovery returns Ghostty sessions when iTerm fails")
    func discoveryGhosttyOnlyWhenItermFails() async throws {
        let iterm = StubBridge()
        iterm.discoverError = ITermBridgeError.itermNotRunning
        let ghostty = StubBridge()
        ghostty.discoverResult = [makeGhosttySession()]

        let composite = MultiTerminalBridge(itermBridge: iterm, ghosttyBridge: ghostty)
        let sessions = try await composite.discoverSessions()

        #expect(sessions.count == 1)
        #expect(sessions[0].terminalType == .ghostty)
    }

    @Test("discovery returns empty when both bridges fail")
    func discoveryEmptyWhenBothFail() async throws {
        let iterm = StubBridge()
        iterm.discoverError = ITermBridgeError.itermNotRunning
        let ghostty = StubBridge()
        ghostty.discoverError = GhosttyBridgeError.ghosttyNotRunning

        let composite = MultiTerminalBridge(itermBridge: iterm, ghosttyBridge: ghostty)
        let sessions = try await composite.discoverSessions()

        #expect(sessions.isEmpty)
    }

    // MARK: - Polled Terminal Types

    @Test("polledTerminalTypes tracks which bridges succeeded")
    func polledTypesTracksBothSuccess() async throws {
        let iterm = StubBridge()
        iterm.discoverResult = [makeItermSession()]
        let ghostty = StubBridge()
        ghostty.discoverResult = [makeGhosttySession()]

        let composite = MultiTerminalBridge(itermBridge: iterm, ghosttyBridge: ghostty)
        _ = try await composite.discoverSessions()

        #expect(composite.polledTerminalTypes == [.iterm, .ghostty])
    }

    @Test("polledTerminalTypes excludes failed bridge")
    func polledTypesExcludesFailed() async throws {
        let iterm = StubBridge()
        iterm.discoverResult = [makeItermSession()]
        let ghostty = StubBridge()
        ghostty.discoverError = GhosttyBridgeError.ghosttyNotRunning

        let composite = MultiTerminalBridge(itermBridge: iterm, ghosttyBridge: ghostty)
        _ = try await composite.discoverSessions()

        #expect(composite.polledTerminalTypes == [.iterm])
    }

    @Test("polledTerminalTypes includes bridge that returned zero sessions")
    func polledTypesIncludesEmptySuccess() async throws {
        let iterm = StubBridge()
        iterm.discoverResult = []
        let ghostty = StubBridge()
        ghostty.discoverResult = []

        let composite = MultiTerminalBridge(itermBridge: iterm, ghosttyBridge: ghostty)
        _ = try await composite.discoverSessions()

        #expect(composite.polledTerminalTypes == [.iterm, .ghostty])
    }

    @Test("polledTerminalTypes is empty before first discovery")
    func polledTypesEmptyBeforeDiscovery() {
        let composite = MultiTerminalBridge(
            itermBridge: StubBridge(),
            ghosttyBridge: StubBridge()
        )
        #expect(composite.polledTerminalTypes.isEmpty)
    }

    // MARK: - Delivery Routing

    @Test("writeToSession routes .tty to iTerm bridge")
    func writeRoutesToIterm() async throws {
        let iterm = StubBridge()
        let ghostty = StubBridge()
        let composite = MultiTerminalBridge(itermBridge: iterm, ghosttyBridge: ghostty)

        let result = try await composite.writeToSession(
            identifier: .tty("/dev/ttys001"),
            text: "hello"
        )

        #expect(result == true)
        #expect(iterm.lastWrittenIdentifier == .tty("/dev/ttys001"))
        #expect(iterm.lastWrittenText == "hello")
        #expect(ghostty.lastWrittenIdentifier == nil)
    }

    @Test("writeToSession routes .ghosttyTerminal to Ghostty bridge")
    func writeRoutesToGhostty() async throws {
        let iterm = StubBridge()
        let ghostty = StubBridge()
        let composite = MultiTerminalBridge(itermBridge: iterm, ghosttyBridge: ghostty)

        let result = try await composite.writeToSession(
            identifier: .ghosttyTerminal("term-42"),
            text: "world"
        )

        #expect(result == true)
        #expect(ghostty.lastWrittenIdentifier == .ghosttyTerminal("term-42"))
        #expect(ghostty.lastWrittenText == "world")
        #expect(iterm.lastWrittenIdentifier == nil)
    }

    @Test("writeToSession propagates errors from the targeted bridge")
    func writePropagatesBridgeError() async {
        let iterm = StubBridge()
        iterm.writeError = ITermBridgeError.itermNotRunning
        let composite = MultiTerminalBridge(itermBridge: iterm, ghosttyBridge: StubBridge())

        do {
            _ = try await composite.writeToSession(
                identifier: .tty("/dev/ttys001"),
                text: "hello"
            )
            Issue.record("Expected error to be thrown")
        } catch is ITermBridgeError {
            // Expected
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }
}

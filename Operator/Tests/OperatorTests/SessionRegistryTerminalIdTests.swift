import Foundation
import Testing

@testable import OperatorCore

/// Namespace matching the file name to satisfy the file_name lint rule.
private enum SessionRegistryTerminalIdTests {}

// MARK: - SessionRegistry TerminalIdentifier Tests

@Suite("SessionRegistry - TerminalIdentifier")
internal struct SessionRegistryTerminalIdTestSuite {
    // MARK: - TerminalIdentifier Keying

    @Test("register with TerminalIdentifier stores session under that key")
    func registerWithIdentifierStoresCorrectly() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        let identifier = TerminalIdentifier.ghosttyTerminal("ghost-123")
        await registry.register(name: "frontend", identifier: identifier, cwd: "/tmp", context: "styling")

        let session = await registry.session(byIdentifier: identifier)
        #expect(session?.name == "frontend")
        #expect(session?.identifier == identifier)
    }

    @Test("deregister with TerminalIdentifier removes session")
    func deregisterWithIdentifierRemoves() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        let identifier = TerminalIdentifier.ghosttyTerminal("ghost-456")
        await registry.register(name: "backend", identifier: identifier, cwd: "/tmp", context: nil)

        let removed = await registry.deregister(identifier: identifier)
        #expect(removed == "backend")
        #expect(await registry.sessionCount == 0)
    }

    @Test("session(byIdentifier:) returns nil for unknown identifier")
    func sessionByIdentifierUnknownReturnsNil() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        let session = await registry.session(byIdentifier: .ghosttyTerminal("nonexistent"))
        #expect(session == nil)
    }

    @Test("SessionState.tty computed property returns path for iTerm sessions")
    func sessionStateTtyComputedForIterm() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        await registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)

        let session = await registry.session(named: "sudo")
        #expect(session?.tty == "/dev/ttys001")
        #expect(session?.identifier == .tty("/dev/ttys001"))
    }

    @Test("SessionState.tty computed property returns ghostty prefix for Ghostty sessions")
    func sessionStateTtyComputedForGhostty() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        await registry.register(
            name: "frontend",
            identifier: .ghosttyTerminal("ghost-789"),
            cwd: "/tmp",
            context: nil
        )

        let session = await registry.session(named: "frontend")
        #expect(session?.tty == "ghostty:ghost-789")
    }

    // MARK: - Terminal ID Resolution

    @Test("handleTerminalIdResolution re-keys session from TTY to Ghostty terminal ID")
    func handleTerminalIdResolutionRekeys() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        await registry.handleSessionStart(
            sessionId: "sess-100",
            tty: "/dev/ttys010",
            cwd: "/tmp",
            terminalType: .ghostty
        )

        let resolved = await registry.handleTerminalIdResolution(
            tty: "/dev/ttys010",
            ghosttyId: "ghost-ABC"
        )
        #expect(resolved == true)

        let byOldTty = await registry.session(byIdentifier: .tty("/dev/ttys010"))
        #expect(byOldTty == nil)

        let byGhosttyId = await registry.session(byIdentifier: .ghosttyTerminal("ghost-ABC"))
        #expect(byGhosttyId?.name == "sess-100")
        #expect(byGhosttyId?.sessionId == "sess-100")
    }

    @Test("handleTerminalIdResolution returns false when no session found for TTY")
    func handleTerminalIdResolutionNoSession() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        let resolved = await registry.handleTerminalIdResolution(
            tty: "/dev/ttys999",
            ghosttyId: "ghost-XYZ"
        )
        #expect(resolved == false)
    }

    @Test("after resolution, hooks with same TTY find re-keyed session")
    func afterResolutionHooksResolveToGhosttySession() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        await registry.handleSessionStart(
            sessionId: "sess-200",
            tty: "/dev/ttys020",
            cwd: "/tmp",
            terminalType: .ghostty
        )

        await registry.handleTerminalIdResolution(
            tty: "/dev/ttys020",
            ghosttyId: "ghost-DEF"
        )

        await registry.handleStop(
            sessionId: "sess-200",
            tty: "/dev/ttys020",
            lastAssistantMessage: "I resolved the issue"
        )

        let session = await registry.session(byIdentifier: .ghosttyTerminal("ghost-DEF"))
        #expect(session?.context == "I resolved the issue")
    }

    @Test("after resolution, session(byTTY:) finds re-keyed session")
    func afterResolutionSessionByTtyFindsRekeyed() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        await registry.handleSessionStart(
            sessionId: "sess-300",
            tty: "/dev/ttys030",
            cwd: "/tmp",
            terminalType: .ghostty
        )

        await registry.handleTerminalIdResolution(
            tty: "/dev/ttys030",
            ghosttyId: "ghost-GHI"
        )

        let session = await registry.session(byTTY: "/dev/ttys030")
        #expect(session?.name == "sess-300")
    }

    @Test("after resolution, handleSessionEnd with TTY removes re-keyed session")
    func afterResolutionHandleSessionEndRemovesRekeyed() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        await registry.handleSessionStart(
            sessionId: "sess-400",
            tty: "/dev/ttys040",
            cwd: "/tmp",
            terminalType: .ghostty
        )

        await registry.handleTerminalIdResolution(
            tty: "/dev/ttys040",
            ghosttyId: "ghost-JKL"
        )

        let removed = await registry.handleSessionEnd(sessionId: "sess-400", tty: "/dev/ttys040")
        #expect(removed == "sess-400")
        #expect(await registry.sessionCount == 0)
    }

    @Test("after resolution, updateContext with TTY updates re-keyed session")
    func afterResolutionUpdateContextUpdatesRekeyed() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        await registry.handleSessionStart(
            sessionId: "sess-500",
            tty: "/dev/ttys050",
            cwd: "/tmp",
            terminalType: .ghostty
        )

        await registry.handleTerminalIdResolution(
            tty: "/dev/ttys050",
            ghosttyId: "ghost-MNO"
        )

        await registry.updateContext(
            tty: "/dev/ttys050",
            summary: "updated context",
            recentMessages: [SessionMessage(role: "user", text: "hello")]
        )

        let session = await registry.session(byIdentifier: .ghosttyTerminal("ghost-MNO"))
        #expect(session?.context == "updated context")
        #expect(session?.recentMessages.count == 1)
    }
}

// MARK: - SessionRegistry Hook + Snapshot Tests

@Suite("SessionRegistry - Ghostty Hooks and Snapshots")
internal struct SessionRegistryGhosttyHookTests {
    @Test("handleSessionStart returns needsTerminalId true for Ghostty without mapping")
    func handleSessionStartGhosttyNeedsTerminalId() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        let result = await registry.handleSessionStart(
            sessionId: "sess-600",
            tty: "/dev/ttys060",
            cwd: "/tmp",
            terminalType: .ghostty
        )

        #expect(result.ok == true)
        #expect(result.needsTerminalId == true)
    }

    @Test("handleSessionStart returns needsTerminalId false for iTerm")
    func handleSessionStartItermNoTerminalId() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        let result = await registry.handleSessionStart(
            sessionId: "sess-700",
            tty: "/dev/ttys070",
            cwd: "/tmp",
            terminalType: .iterm
        )

        #expect(result.ok == true)
        #expect(result.needsTerminalId == false)
    }

    @Test("handleSessionStart returns needsTerminalId false for Ghostty with existing mapping")
    func handleSessionStartGhosttyWithMappingNoTerminalId() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        await registry.handleSessionStart(
            sessionId: "sess-800",
            tty: "/dev/ttys080",
            cwd: "/tmp",
            terminalType: .ghostty
        )

        await registry.handleTerminalIdResolution(
            tty: "/dev/ttys080",
            ghosttyId: "ghost-PQR"
        )

        let result = await registry.handleSessionStart(
            sessionId: "sess-800",
            tty: "/dev/ttys080",
            cwd: "/tmp",
            terminalType: .ghostty
        )

        #expect(result.ok == true)
        #expect(result.needsTerminalId == false)
    }

    @Test("handleSessionStart defaults to iTerm when terminalType omitted")
    func handleSessionStartDefaultTerminalType() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        let result = await registry.handleSessionStart(
            sessionId: "sess-900",
            tty: "/dev/ttys090",
            cwd: "/tmp"
        )

        #expect(result.ok == true)
        #expect(result.needsTerminalId == false)
    }

    // MARK: - Snapshot with TerminalIdentifier

    @Test("getState snapshot includes identifier and backward-compatible tty")
    func getStateSnapshotIncludesIdentifier() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        await registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: "building")

        let snapshot = await registry.getState()
        let session = snapshot.sessions.first
        #expect(session?.identifier == .tty("/dev/ttys001"))
        #expect(session?.tty == "/dev/ttys001")
    }

    @Test("getState snapshot JSON includes tty field for backward compatibility")
    func getStateSnapshotJsonIncludesTty() async throws {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        await registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: "building")

        let snapshot = await registry.getState()
        let json = snapshot.toJSON()
        #expect(json.contains("\"tty\""))

        guard let data = json.data(using: .utf8) else {
            Issue.record("Failed to convert JSON string to data")
            return
        }
        let parsed = try JSONSerialization.jsonObject(with: data)
        guard let decoded = parsed as? [String: Any],
            let sessions = decoded["sessions"] as? [[String: Any]]
        else {
            Issue.record("Failed to parse JSON structure")
            return
        }
        #expect(sessions.first?["tty"] as? String == "/dev/ttys001")
    }
}

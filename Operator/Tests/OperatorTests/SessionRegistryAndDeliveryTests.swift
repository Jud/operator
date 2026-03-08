import AVFoundation
import Foundation
import Testing

@testable import OperatorCore

/// Namespace matching the file name to satisfy the file_name lint rule.
private enum SessionRegistryAndDeliveryTests {}

// MARK: - SessionRegistry Tests

@Suite("SessionRegistry")
internal struct SessionRegistryTests {
    // MARK: - Registration

    @Test("register a session and verify it appears in allSessions")
    func registerAppearsInAllSessions() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        await registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)

        let all = await registry.allSessions()
        #expect(all.count == 1)
        #expect(all.first?.name == "sudo")
    }

    @Test("register returns true for valid names")
    func registerReturnsTrueForValidName() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        let result = await registry.register(name: "frontend", tty: "/dev/ttys001", cwd: "/tmp", context: nil)
        #expect(result == true)
    }

    @Test("register returns false for 'operator' (case insensitive)")
    func registerReturnsFalseForOperator() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        let r1 = await registry.register(name: "operator", tty: "/dev/ttys001", cwd: "/tmp", context: nil)
        #expect(r1 == false)

        let r2 = await registry.register(name: "Operator", tty: "/dev/ttys002", cwd: "/tmp", context: nil)
        #expect(r2 == false)

        let r3 = await registry.register(name: "OPERATOR", tty: "/dev/ttys003", cwd: "/tmp", context: nil)
        #expect(r3 == false)

        let all = await registry.allSessions()
        #expect(all.isEmpty)
    }

    // MARK: - Deregistration

    @Test("deregister removes session and returns name")
    func deregisterRemovesAndReturnsName() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        await registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)
        let name = await registry.deregister(tty: "/dev/ttys001")

        #expect(name == "sudo")
        let all = await registry.allSessions()
        #expect(all.isEmpty)
    }

    @Test("deregister unknown TTY returns nil")
    func deregisterUnknownReturnsNil() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        let name = await registry.deregister(tty: "/dev/ttys999")
        #expect(name == nil)
    }

    // MARK: - Re-registration

    @Test("re-registration on same TTY updates session")
    func reRegistrationUpdateSession() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        await registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: "old context")
        await registry.register(name: "frontend", tty: "/dev/ttys001", cwd: "/home", context: "new context")

        let all = await registry.allSessions()
        #expect(all.count == 1)
        #expect(all.first?.name == "frontend")
        #expect(all.first?.cwd == "/home")
    }

    // MARK: - Session Lookup

    @Test("session(named:) returns correct session")
    func sessionNamedReturnsCorrect() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        await registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)
        await registry.register(name: "frontend", tty: "/dev/ttys002", cwd: "/home", context: nil)

        let session = await registry.session(named: "sudo")
        #expect(session?.tty == "/dev/ttys001")
    }

    @Test("session(byTTY:) returns correct session")
    func sessionByTTYReturnsCorrect() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        await registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)

        let session = await registry.session(byTTY: "/dev/ttys001")
        #expect(session?.name == "sudo")
    }

    @Test("session not found by name returns nil")
    func sessionNotFoundByNameReturnsNil() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        let session = await registry.session(named: "nonexistent")
        #expect(session == nil)
    }

    // MARK: - Voice Lookup

    @Test("voiceFor returns default voice for unknown session")
    func voiceForUnknownReturnsDefault() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        let voice = await registry.voiceFor(session: "unknown")
        #expect(voice.identifier == voiceManager.defaultAgentVoice.identifier)
    }

    @Test("pitchFor returns 1.0 for unknown session")
    func pitchForUnknownReturnsDefault() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        let pitch = await registry.pitchFor(session: "unknown")
        #expect(pitch == 1.0)
    }

    // MARK: - Context Updates

    @Test("updateContext updates context and recentMessages")
    func updateContextUpdatesFields() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        await registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)

        let messages = [SessionMessage(role: "user", text: "hello")]
        await registry.updateContext(tty: "/dev/ttys001", summary: "working on tests", recentMessages: messages)

        let session = await registry.session(byTTY: "/dev/ttys001")
        #expect(session?.context == "working on tests")
        #expect(session?.recentMessages.count == 1)
        #expect(session?.recentMessages.first?.text == "hello")
    }

    @Test("updateContext for unregistered TTY is a no-op")
    func updateContextUnregisteredIsNoOp() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        await registry.updateContext(
            tty: "/dev/ttys999",
            summary: "should not crash",
            recentMessages: []
        )

        let all = await registry.allSessions()
        #expect(all.isEmpty)
    }

    // MARK: - Session Count

    @Test("sessionCount tracks correctly after register/deregister")
    func sessionCountTracksCorrectly() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        #expect(await registry.sessionCount == 0)

        await registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)
        #expect(await registry.sessionCount == 1)

        await registry.register(name: "frontend", tty: "/dev/ttys002", cwd: "/tmp", context: nil)
        #expect(await registry.sessionCount == 2)

        await registry.deregister(tty: "/dev/ttys001")
        #expect(await registry.sessionCount == 1)
    }

    // MARK: - Hook Handlers

    @Test("handleSessionStart creates new session if TTY not registered")
    func handleSessionStartCreatesNew() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        await registry.handleSessionStart(sessionId: "sess-123", tty: "/dev/ttys001", cwd: "/tmp")

        let session = await registry.session(byTTY: "/dev/ttys001")
        #expect(session != nil)
        #expect(session?.sessionId == "sess-123")
        #expect(session?.name == "sess-123")
    }

    @Test("handleSessionStart updates sessionId if TTY already registered")
    func handleSessionStartUpdatesExisting() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        await registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)
        await registry.handleSessionStart(sessionId: "sess-456", tty: "/dev/ttys001", cwd: "/tmp")

        let session = await registry.session(byTTY: "/dev/ttys001")
        #expect(session?.name == "sudo")
        #expect(session?.sessionId == "sess-456")
    }

    @Test("handleStop updates context with lastAssistantMessage")
    func handleStopUpdatesContext() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        await registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: "old")
        await registry.handleStop(
            sessionId: "sess-123",
            tty: "/dev/ttys001",
            lastAssistantMessage: "I fixed the bug"
        )

        let session = await registry.session(byTTY: "/dev/ttys001")
        #expect(session?.context == "I fixed the bug")
        #expect(session?.sessionId == "sess-123")
    }

    @Test("handleStop for unknown TTY is a no-op")
    func handleStopUnknownIsNoOp() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        await registry.handleStop(
            sessionId: "sess-123",
            tty: "/dev/ttys999",
            lastAssistantMessage: "should not crash"
        )

        let all = await registry.allSessions()
        #expect(all.isEmpty)
    }

    @Test("handleSessionEnd removes session and returns name")
    func handleSessionEndRemovesAndReturnsName() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        await registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)
        let name = await registry.handleSessionEnd(sessionId: "sess-123", tty: "/dev/ttys001")

        #expect(name == "sudo")
        let all = await registry.allSessions()
        #expect(all.isEmpty)
    }

    @Test("handleSessionEnd for unknown TTY returns nil")
    func handleSessionEndUnknownReturnsNil() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        let name = await registry.handleSessionEnd(sessionId: "sess-123", tty: "/dev/ttys999")
        #expect(name == nil)
    }

    // MARK: - State Snapshot

    @Test("getState returns correct RegistrySnapshot")
    func getStateReturnsCorrectSnapshot() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        await registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: "building")
        await registry.register(name: "frontend", tty: "/dev/ttys002", cwd: "/home", context: "styling")

        let snapshot = await registry.getState()
        #expect(snapshot.sessionCount == 2)
        #expect(snapshot.sessions.count == 2)

        let names = Set(snapshot.sessions.map(\.name))
        #expect(names.contains("sudo"))
        #expect(names.contains("frontend"))
    }
}

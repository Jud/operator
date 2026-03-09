import Foundation
import Testing

@testable import OperatorCore

@Suite("MessageRouter - Clarification")
internal struct MessageRouterClarificationTests {
    @Test("clarification: keyword match resolves to session")
    func clarificationKeywordMatch() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)
        await registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)
        await registry.register(name: "frontend", tty: "/dev/ttys002", cwd: "/tmp", context: nil)
        let router = MessageRouter(registry: registry)

        let result = await router.handleClarification(
            response: "sudo",
            candidates: ["sudo", "frontend"],
            originalText: "looks good, merge it"
        )
        guard case .resolved(let session) = result else {
            Issue.record("Expected .resolved, got \(result)")
            return
        }
        #expect(session == "sudo")
    }

    @Test("clarification: cancel intent")
    func clarificationCancel() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)
        await registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)
        let router = MessageRouter(registry: registry)

        let result = await router.handleClarification(
            response: "never mind",
            candidates: ["sudo", "frontend"],
            originalText: "test"
        )
        guard case .cancelled = result else {
            Issue.record("Expected .cancelled, got \(result)")
            return
        }
    }

    @Test("clarification: cancel with 'cancel'")
    func clarificationCancelWord() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)
        await registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)
        let router = MessageRouter(registry: registry)

        let result = await router.handleClarification(
            response: "cancel",
            candidates: ["sudo"],
            originalText: "test"
        )
        guard case .cancelled = result else {
            Issue.record("Expected .cancelled, got \(result)")
            return
        }
    }

    @Test("clarification: both/all sends to all")
    func clarificationBoth() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)
        await registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)
        let router = MessageRouter(registry: registry)

        let result = await router.handleClarification(
            response: "both",
            candidates: ["sudo", "frontend"],
            originalText: "test"
        )
        guard case .sendToAll = result else {
            Issue.record("Expected .sendToAll, got \(result)")
            return
        }
    }

    @Test("clarification: 'all of them' sends to all")
    func clarificationAllOfThem() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)
        await registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)
        let router = MessageRouter(registry: registry)

        let result = await router.handleClarification(
            response: "all of them",
            candidates: ["sudo", "frontend"],
            originalText: "test"
        )
        guard case .sendToAll = result else {
            Issue.record("Expected .sendToAll, got \(result)")
            return
        }
    }

    @Test("clarification: repeat question")
    func clarificationRepeat() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)
        await registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)
        let router = MessageRouter(registry: registry)

        let result = await router.handleClarification(
            response: "repeat",
            candidates: ["sudo"],
            originalText: "test"
        )
        guard case .repeatQuestion = result else {
            Issue.record("Expected .repeatQuestion, got \(result)")
            return
        }
    }

    @Test("clarification: ordinal 'the first one'")
    func clarificationOrdinalFirst() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)
        await registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)
        await registry.register(name: "frontend", tty: "/dev/ttys002", cwd: "/tmp", context: nil)
        let router = MessageRouter(registry: registry)

        let result = await router.handleClarification(
            response: "the first one",
            candidates: ["sudo", "frontend"],
            originalText: "test"
        )
        guard case .resolved(let session) = result else {
            Issue.record("Expected .resolved, got \(result)")
            return
        }
        #expect(session == "sudo")
    }

    @Test("clarification: ordinal 'second'")
    func clarificationOrdinalSecond() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)
        await registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)
        await registry.register(name: "frontend", tty: "/dev/ttys002", cwd: "/tmp", context: nil)
        let router = MessageRouter(registry: registry)

        let result = await router.handleClarification(
            response: "second",
            candidates: ["sudo", "frontend"],
            originalText: "test"
        )
        guard case .resolved(let session) = result else {
            Issue.record("Expected .resolved, got \(result)")
            return
        }
        #expect(session == "frontend")
    }

    @Test("clarification: ordinal 'the second one'")
    func clarificationOrdinalSecondOne() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)
        await registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)
        await registry.register(name: "frontend", tty: "/dev/ttys002", cwd: "/tmp", context: nil)
        let router = MessageRouter(registry: registry)

        let result = await router.handleClarification(
            response: "the second one",
            candidates: ["sudo", "frontend"],
            originalText: "test"
        )
        guard case .resolved(let session) = result else {
            Issue.record("Expected .resolved, got \(result)")
            return
        }
        #expect(session == "frontend")
    }
}

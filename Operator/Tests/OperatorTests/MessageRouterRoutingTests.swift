import AVFoundation
import Foundation
import Testing

@testable import OperatorCore

// MARK: - Routing & Clarification Tests

@Suite("MessageRouter - Routing & Clarification")
internal struct MessageRouterRoutingTests {
    @Test("single session routes directly without routing overhead")
    func singleSessionBypass() async {
        let bridge = ITermBridge()
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(itermBridge: bridge, voiceManager: voiceManager)
        await registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)
        let router = MessageRouter(registry: registry)
        let state = RoutingState()

        let result = await router.route(text: "make the button bigger", routingState: state)
        guard case .route(let session, let message) = result else {
            Issue.record("Expected .route, got \(result)")
            return
        }
        #expect(session == "sudo")
        #expect(message == "make the button bigger")
    }

    @Test("returns noSessions when no sessions registered")
    func noSessionsRegistered() async {
        let bridge = ITermBridge()
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(itermBridge: bridge, voiceManager: voiceManager)
        let router = MessageRouter(registry: registry)
        let state = RoutingState()

        let result = await router.route(text: "fix the build", routingState: state)
        guard case .noSessions = result else {
            Issue.record("Expected .noSessions, got \(result)")
            return
        }
    }

    @Test("session affinity routes to last target within 15s window")
    func sessionAffinityWithinWindow() async {
        let bridge = ITermBridge()
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(itermBridge: bridge, voiceManager: voiceManager)
        await registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp/sudo", context: nil)
        await registry.register(name: "frontend", tty: "/dev/ttys002", cwd: "/tmp/frontend", context: nil)
        let router = MessageRouter(registry: registry)

        var state = RoutingState()
        state.recordRoute(text: "fix the build", session: "sudo", timestamp: Date())

        let result = await router.route(text: "looks good", routingState: state)
        guard case .route(let session, _) = result else {
            Issue.record("Expected .route via affinity, got \(result)")
            return
        }
        #expect(session == "sudo")
    }

    @Test("clarification: keyword match resolves to session")
    func clarificationKeywordMatch() async {
        let bridge = ITermBridge()
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(itermBridge: bridge, voiceManager: voiceManager)
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
        let bridge = ITermBridge()
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(itermBridge: bridge, voiceManager: voiceManager)
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
        let bridge = ITermBridge()
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(itermBridge: bridge, voiceManager: voiceManager)
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
        let bridge = ITermBridge()
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(itermBridge: bridge, voiceManager: voiceManager)
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
        let bridge = ITermBridge()
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(itermBridge: bridge, voiceManager: voiceManager)
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
        let bridge = ITermBridge()
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(itermBridge: bridge, voiceManager: voiceManager)
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
        let bridge = ITermBridge()
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(itermBridge: bridge, voiceManager: voiceManager)
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
        let bridge = ITermBridge()
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(itermBridge: bridge, voiceManager: voiceManager)
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
        let bridge = ITermBridge()
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(itermBridge: bridge, voiceManager: voiceManager)
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

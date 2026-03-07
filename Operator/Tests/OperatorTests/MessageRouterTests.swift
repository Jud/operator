import AVFoundation
import Foundation
import Testing

@testable import OperatorCore

@Suite("MessageRouter")
internal struct MessageRouterTests {
    // MARK: - Operator Command Detection

    @Test("operator status detected as operator command")
    func operatorStatusDetected() async {
        let bridge = ITermBridge()
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(itermBridge: bridge, voiceManager: voiceManager)
        let router = MessageRouter(registry: registry)
        let state = RoutingState()

        let result = await router.route(text: "operator status", routingState: state)
        guard case .operatorCommand(.status) = result else {
            Issue.record("Expected .operatorCommand(.status), got \(result)")
            return
        }
    }

    @Test("list agents detected as operator command")
    func listAgentsDetected() async {
        let bridge = ITermBridge()
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(itermBridge: bridge, voiceManager: voiceManager)
        let router = MessageRouter(registry: registry)
        let state = RoutingState()

        let result = await router.route(text: "list agents", routingState: state)
        guard case .operatorCommand(.listAgents) = result else {
            Issue.record("Expected .operatorCommand(.listAgents), got \(result)")
            return
        }
    }

    @Test("who's running detected as operator command")
    func whosRunningDetected() async {
        let bridge = ITermBridge()
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(itermBridge: bridge, voiceManager: voiceManager)
        let router = MessageRouter(registry: registry)
        let state = RoutingState()

        let result = await router.route(text: "who's running", routingState: state)
        guard case .operatorCommand(.listAgents) = result else {
            Issue.record("Expected .operatorCommand(.listAgents), got \(result)")
            return
        }
    }

    @Test("what did i miss detected as operator command")
    func whatDidIMissDetected() async {
        let bridge = ITermBridge()
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(itermBridge: bridge, voiceManager: voiceManager)
        let router = MessageRouter(registry: registry)
        let state = RoutingState()

        let result = await router.route(text: "what did I miss", routingState: state)
        guard case .operatorCommand(.whatDidIMiss) = result else {
            Issue.record("Expected .operatorCommand(.whatDidIMiss), got \(result)")
            return
        }
    }

    @Test("operator help detected as operator command")
    func operatorHelpDetected() async {
        let bridge = ITermBridge()
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(itermBridge: bridge, voiceManager: voiceManager)
        let router = MessageRouter(registry: registry)
        let state = RoutingState()

        let result = await router.route(text: "operator help", routingState: state)
        guard case .operatorCommand(.help) = result else {
            Issue.record("Expected .operatorCommand(.help), got \(result)")
            return
        }
    }

    @Test("operator command checked before single-session bypass")
    func operatorCommandBeforeSingleSession() async {
        let bridge = ITermBridge()
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(itermBridge: bridge, voiceManager: voiceManager)
        await registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)
        let router = MessageRouter(registry: registry)
        let state = RoutingState()

        let result = await router.route(text: "operator status", routingState: state)
        guard case .operatorCommand(.status) = result else {
            Issue.record("Expected .operatorCommand(.status) even with single session, got \(result)")
            return
        }
    }

    @Test("what's everyone working on detected as operator command")
    func whatsEveryoneWorkingOnDetected() async {
        let bridge = ITermBridge()
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(itermBridge: bridge, voiceManager: voiceManager)
        let router = MessageRouter(registry: registry)
        let state = RoutingState()

        let result = await router.route(text: "what's everyone working on", routingState: state)
        guard case .operatorCommand(.status) = result else {
            Issue.record("Expected .operatorCommand(.status), got \(result)")
            return
        }
    }
}

// MARK: - Keyword Extraction Tests

@Suite("MessageRouter - Keyword Extraction")
internal struct MessageRouterKeywordTests {
    @Test("tell <agent> extracts agent name and message")
    func tellAgentExtracts() async {
        let bridge = ITermBridge()
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(itermBridge: bridge, voiceManager: voiceManager)
        await registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)
        let router = MessageRouter(registry: registry)
        let state = RoutingState()

        let result = await router.route(text: "tell sudo to fix the build", routingState: state)
        guard case .route(let session, let message) = result else {
            Issue.record("Expected .route, got \(result)")
            return
        }
        #expect(session == "sudo")
        #expect(message == "to fix the build")
    }

    @Test("hey <agent> extracts agent name and message")
    func heyAgentExtracts() async {
        let bridge = ITermBridge()
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(itermBridge: bridge, voiceManager: voiceManager)
        await registry.register(name: "frontend", tty: "/dev/ttys002", cwd: "/tmp", context: nil)
        let router = MessageRouter(registry: registry)
        let state = RoutingState()

        let result = await router.route(text: "hey frontend, make it blue", routingState: state)
        guard case .route(let session, let message) = result else {
            Issue.record("Expected .route, got \(result)")
            return
        }
        #expect(session == "frontend")
        #expect(message == "make it blue")
    }

    @Test("@agent extracts agent name and message")
    func atAgentExtracts() async {
        let bridge = ITermBridge()
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(itermBridge: bridge, voiceManager: voiceManager)
        await registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)
        let router = MessageRouter(registry: registry)
        let state = RoutingState()

        let result = await router.route(text: "@sudo check the logs", routingState: state)
        guard case .route(let session, let message) = result else {
            Issue.record("Expected .route, got \(result)")
            return
        }
        #expect(session == "sudo")
        #expect(message == "check the logs")
    }

    @Test("keyword extraction is case-insensitive")
    func keywordCaseInsensitive() async {
        let bridge = ITermBridge()
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(itermBridge: bridge, voiceManager: voiceManager)
        await registry.register(name: "Sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)
        let router = MessageRouter(registry: registry)
        let state = RoutingState()

        let result = await router.route(text: "tell SUDO to fix it", routingState: state)
        guard case .route(let session, _) = result else {
            Issue.record("Expected .route, got \(result)")
            return
        }
        #expect(session == "Sudo")
    }

    @Test("keyword extraction ignores unregistered agents")
    func keywordIgnoresUnregistered() async {
        let bridge = ITermBridge()
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(itermBridge: bridge, voiceManager: voiceManager)
        await registry.register(name: "frontend", tty: "/dev/ttys001", cwd: "/tmp", context: nil)
        let router = MessageRouter(registry: registry)
        let state = RoutingState()

        let result = await router.route(text: "tell bob to fix the build", routingState: state)
        guard case .route(let session, let message) = result else {
            Issue.record("Expected .route via single-session bypass, got \(result)")
            return
        }
        #expect(session == "frontend")
        #expect(message == "tell bob to fix the build")
    }
}

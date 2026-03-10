import AVFoundation
import Foundation
import Testing

@testable import OperatorCore

/// Shared factory for BimodalDecisionEngine tests.
private func makeEngine(
    isTerminal: Bool = false,
    isTextField: Bool = true,
    hasPermission: Bool = true,
    bundleID: String? = "com.example.app"
) async -> (engine: BimodalDecisionEngine, registry: SessionRegistry) {
    let mock = MockAccessibilityQuery(
        bundleID: bundleID,
        isTerminal: isTerminal,
        isTextField: isTextField,
        hasPermission: hasPermission
    )
    let voiceManager = VoiceManager()
    let registry = SessionRegistry(voiceManager: voiceManager)
    let router = MessageRouter(registry: registry)
    let engine = BimodalDecisionEngine(
        accessibilityQuery: mock,
        router: router,
        registry: registry
    )
    return (engine, registry)
}

// MARK: - Step 1: Affinity / Pending State

@Suite("BimodalDecisionEngine - Pending State Priority")
internal struct BimodalDecisionEngineTests {
    @Test("active affinity routes to agent regardless of frontmost app")
    func affinityActiveRoutesToAgent() async {
        let (engine, registry) = await makeEngine(isTerminal: false, isTextField: true)
        await registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)

        var state = RoutingState()
        state.recordRoute(text: "previous message", session: "sudo", timestamp: Date())

        let decision = await engine.decide(text: "follow up message", routingState: state)
        guard case .routeToAgent(let text) = decision else {
            Issue.record("Expected .routeToAgent, got \(decision)")
            return
        }
        #expect(text == "follow up message")
    }

    @Test("pending interruption routes to agent")
    func pendingInterruptionRoutesToAgent() async {
        let (engine, _) = await makeEngine(isTerminal: false, isTextField: true)

        var state = RoutingState()
        state.storeInterruption(heardText: "heard", unheardText: "unheard", session: "sudo")

        let decision = await engine.decide(text: "skip that", routingState: state)
        guard case .routeToAgent = decision else {
            Issue.record("Expected .routeToAgent, got \(decision)")
            return
        }
    }

    @Test("pending clarification routes to agent")
    func pendingClarificationRoutesToAgent() async {
        let (engine, _) = await makeEngine(isTerminal: false, isTextField: true)

        var state = RoutingState()
        state.storeClarification(candidates: ["sudo", "frontend"], originalText: "fix the build")

        let decision = await engine.decide(text: "sudo", routingState: state)
        guard case .routeToAgent = decision else {
            Issue.record("Expected .routeToAgent, got \(decision)")
            return
        }
    }

    @Test("affinity overrides non-terminal frontmost app (follow-up routing)")
    func affinityOverridesNonTerminal() async {
        let (engine, registry) = await makeEngine(isTerminal: false, isTextField: true)
        await registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)

        var state = RoutingState()
        state.recordRoute(text: "deploy staging", session: "sudo", timestamp: Date())

        let decision = await engine.decide(text: "actually, use flexbox instead", routingState: state)
        guard case .routeToAgent = decision else {
            Issue.record("Expected .routeToAgent, got \(decision)")
            return
        }
    }
}

// MARK: - Step 2: Keyword Extraction

@Suite("BimodalDecisionEngine - Keyword Routing")
internal struct BimodalKeywordTests {
    @Test("keyword match routes to agent regardless of frontmost app")
    func keywordMatchRoutesToAgent() async {
        let (engine, registry) = await makeEngine(isTerminal: false, isTextField: true)
        await registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)

        let decision = await engine.decide(
            text: "tell sudo to deploy the staging branch",
            routingState: RoutingState()
        )
        guard case .routeToAgent(let text) = decision else {
            Issue.record("Expected .routeToAgent, got \(decision)")
            return
        }
        #expect(text == "tell sudo to deploy the staging branch")
    }

    @Test("hey keyword match routes to agent")
    func heyKeywordRoutesToAgent() async {
        let (engine, registry) = await makeEngine(isTerminal: false, isTextField: true)
        await registry.register(name: "frontend", tty: "/dev/ttys001", cwd: "/tmp", context: nil)

        let decision = await engine.decide(
            text: "hey frontend, make it blue",
            routingState: RoutingState()
        )
        guard case .routeToAgent = decision else {
            Issue.record("Expected .routeToAgent, got \(decision)")
            return
        }
    }

    @Test("at-sign keyword match routes to agent")
    func atSignKeywordRoutesToAgent() async {
        let (engine, registry) = await makeEngine(isTerminal: false, isTextField: true)
        await registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)

        let decision = await engine.decide(
            text: "@sudo check the logs",
            routingState: RoutingState()
        )
        guard case .routeToAgent = decision else {
            Issue.record("Expected .routeToAgent, got \(decision)")
            return
        }
    }

    @Test("keyword match for unregistered agent does not route to agent")
    func keywordUnregisteredAgentDoesNotRoute() async {
        let (engine, registry) = await makeEngine(isTerminal: false, isTextField: true)
        await registry.register(name: "frontend", tty: "/dev/ttys001", cwd: "/tmp", context: nil)

        let decision = await engine.decide(
            text: "tell bob to fix the build",
            routingState: RoutingState()
        )
        guard case .dictate = decision else {
            Issue.record("Expected .dictate, got \(decision)")
            return
        }
    }
}

// MARK: - Step 3 & 4: Terminal / Non-Terminal Frontmost

@Suite("BimodalDecisionEngine - Frontmost App")
internal struct BimodalFrontmostAppTests {
    @Test("terminal frontmost with single session routes to agent")
    func terminalSingleSessionRoutesToAgent() async {
        let (engine, registry) = await makeEngine(
            isTerminal: true,
            isTextField: false,
            bundleID: "com.googlecode.iterm2"
        )
        await registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)

        let decision = await engine.decide(text: "run the test suite", routingState: RoutingState())
        guard case .routeToAgent(let text) = decision else {
            Issue.record("Expected .routeToAgent, got \(decision)")
            return
        }
        #expect(text == "run the test suite")
    }

    @Test("terminal frontmost with multiple sessions and no confident match dictates")
    func terminalMultiSessionNotConfidentDictates() async {
        let (engine, registry) = await makeEngine(
            isTerminal: true,
            isTextField: true,
            bundleID: "com.googlecode.iterm2"
        )
        await registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)
        await registry.register(name: "frontend", tty: "/dev/ttys002", cwd: "/tmp", context: nil)

        let decision = await engine.decide(text: "some ambiguous message", routingState: RoutingState())
        guard case .dictate(let text) = decision else {
            Issue.record("Expected .dictate, got \(decision)")
            return
        }
        #expect(text == "some ambiguous message")
    }

    @Test("terminal frontmost with not confident and no text field returns noTextField")
    func terminalNotConfidentNoTextField() async {
        let (engine, registry) = await makeEngine(
            isTerminal: true,
            isTextField: false,
            bundleID: "com.googlecode.iterm2"
        )
        await registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)
        await registry.register(name: "frontend", tty: "/dev/ttys002", cwd: "/tmp", context: nil)

        let decision = await engine.decide(text: "some ambiguous message", routingState: RoutingState())
        guard case .noTextField = decision else {
            Issue.record("Expected .noTextField, got \(decision)")
            return
        }
    }

    @Test("terminal frontmost with operator command routes to agent")
    func terminalOperatorCommand() async {
        let (engine, registry) = await makeEngine(
            isTerminal: true,
            isTextField: false,
            bundleID: "com.googlecode.iterm2"
        )
        await registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)

        let decision = await engine.decide(text: "operator status", routingState: RoutingState())
        guard case .routeToAgent = decision else {
            Issue.record("Expected .routeToAgent, got \(decision)")
            return
        }
    }

    @Test("non-terminal frontmost with text field dictates")
    func nonTerminalTextFieldDictates() async {
        let (engine, _) = await makeEngine(isTerminal: false, isTextField: true)

        let decision = await engine.decide(
            text: "add a TODO comment about fixing the parser",
            routingState: RoutingState()
        )
        guard case .dictate(let text) = decision else {
            Issue.record("Expected .dictate, got \(decision)")
            return
        }
        #expect(text == "add a TODO comment about fixing the parser")
    }

    @Test("non-terminal frontmost with no text field returns noTextField")
    func nonTerminalNoTextField() async {
        let (engine, _) = await makeEngine(isTerminal: false, isTextField: false)

        let decision = await engine.decide(text: "some message", routingState: RoutingState())
        guard case .noTextField = decision else {
            Issue.record("Expected .noTextField, got \(decision)")
            return
        }
    }
}

// MARK: - Edge Cases

@Suite("BimodalDecisionEngine - Edge Cases")
internal struct BimodalEdgeCaseTests {
    @Test("no sessions registered and non-terminal dictates")
    func noSessionsRegisteredDictates() async {
        let (engine, _) = await makeEngine(isTerminal: false, isTextField: true)
        let decision = await engine.decide(text: "some message", routingState: RoutingState())
        guard case .dictate = decision else {
            Issue.record("Expected .dictate, got \(decision)")
            return
        }
    }

    @Test("no sessions registered and terminal with text field dictates")
    func noSessionsTerminalDictates() async {
        let (engine, _) = await makeEngine(
            isTerminal: true,
            isTextField: true,
            bundleID: "com.googlecode.iterm2"
        )
        let decision = await engine.decide(text: "some message", routingState: RoutingState())
        guard case .dictate = decision else {
            Issue.record("Expected .dictate, got \(decision)")
            return
        }
    }

    @Test("permission not granted returns permissionRequired")
    func noPermissionReturnsPermissionRequired() async {
        let (engine, _) = await makeEngine(hasPermission: false)
        let decision = await engine.decide(text: "some message", routingState: RoutingState())
        guard case .permissionRequired = decision else {
            Issue.record("Expected .permissionRequired, got \(decision)")
            return
        }
    }

    @Test("HYP-001: multi-session heuristic results discarded for terminal frontmost")
    func hypMultiSessionHeuristicDiscarded() async {
        let (engine, _) = await makeMultiSessionTerminalEngine()

        let decision = await engine.decide(
            text: "tighten the button spacing in the React shell",
            routingState: RoutingState()
        )
        guard case .dictate = decision else {
            Issue.record("Expected .dictate (heuristic discarded per HYP-001), got \(decision)")
            return
        }
    }
}

/// Build a BimodalDecisionEngine with 3 sessions and terminal frontmost for HYP-001 testing.
private func makeMultiSessionTerminalEngine() async -> (BimodalDecisionEngine, SessionRegistry) {
    let (engine, registry) = await makeEngine(
        isTerminal: true,
        isTextField: true,
        bundleID: "com.googlecode.iterm2"
    )
    await registerHeuristicFixtures(into: registry)
    return (engine, registry)
}

import AVFoundation
import Foundation
import Testing

@testable import OperatorCore

private final class CountingRoutingEngine: RoutingEngine, @unchecked Sendable {
    private(set) var callCount = 0
    private let result: [String: Any]

    init(result: [String: Any] = ["session": "", "confident": false]) {
        self.result = result
    }

    func run(prompt _: String, timeout _: TimeInterval) async throws -> [String: Any] {
        callCount += 1
        return result
    }
}

// MARK: - Routing & Clarification Tests

@Suite("MessageRouter - Routing & Clarification")
internal struct MessageRouterRoutingTests {
    private func makeHeuristicRouter(engine: CountingRoutingEngine) async -> MessageRouter {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)

        await registry.register(
            name: "ui-shell",
            tty: "/dev/ttys101",
            cwd: "/Users/jud/work/operator-ui",
            context: "Branch feat/settings-loading"
        )
        await registry.updateContext(
            tty: "/dev/ttys101",
            summary: "Editing web/src/routes/settings.tsx, web/src/components/LoadingCard.tsx, and web/src/styles/dashboard.css for React loading states and layout polish.",
            recentMessages: [
                SessionMessage(role: "assistant", text: "I am tightening the dashboard skeletons, CSS grid layout, and button spacing in the React shell.")
            ]
        )

        await registry.register(
            name: "profile-api",
            tty: "/dev/ttys102",
            cwd: "/Users/jud/work/operator-api",
            context: "Branch feat/profile-endpoints"
        )
        await registry.updateContext(
            tty: "/dev/ttys102",
            summary: "Editing api/routes/profile.py, services/user_profile.py, and db/queries/profile.sql for REST endpoints, Postgres query performance, and connection-pool tuning.",
            recentMessages: [
                SessionMessage(role: "assistant", text: "I am adding the preferences endpoint, fixing the database pool, and optimizing the SQL query.")
            ]
        )

        await registry.register(
            name: "staging-infra",
            tty: "/dev/ttys103",
            cwd: "/Users/jud/work/operator-infra",
            context: "Branch chore/staging-network"
        )
        await registry.updateContext(
            tty: "/dev/ttys103",
            summary: "Editing infra/envs/staging/main.tf, modules/vpc, and modules/iam_policy for Terraform rollout failures, private subnets, IAM policies, and S3 logging.",
            recentMessages: [
                SessionMessage(role: "assistant", text: "I am updating the Terraform networking modules, VPC layout, IAM policy, and S3 logging bucket policy.")
            ]
        )

        return MessageRouter(registry: registry, engine: engine)
    }

    @Test("single session routes directly without routing overhead")
    func singleSessionBypass() async {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)
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
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)
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
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)
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

    @Test("context heuristic routes frontend work without invoking model")
    func heuristicRoutesFrontend() async {
        let engine = CountingRoutingEngine()
        let router = await makeHeuristicRouter(engine: engine)

        let result = await router.route(text: "tighten the button spacing in the React shell", routingState: RoutingState())

        guard case .route(let session, _) = result else {
            Issue.record("Expected .route, got \(result)")
            return
        }
        #expect(session == "ui-shell")
        #expect(engine.callCount == 0)
    }

    @Test("context heuristic routes backend work without invoking model")
    func heuristicRoutesBackend() async {
        let engine = CountingRoutingEngine()
        let router = await makeHeuristicRouter(engine: engine)

        let result = await router.route(text: "fix the database connection pool", routingState: RoutingState())

        guard case .route(let session, _) = result else {
            Issue.record("Expected .route, got \(result)")
            return
        }
        #expect(session == "profile-api")
        #expect(engine.callCount == 0)
    }

    @Test("context heuristic routes infra work without invoking model")
    func heuristicRoutesInfra() async {
        let engine = CountingRoutingEngine()
        let router = await makeHeuristicRouter(engine: engine)

        let result = await router.route(text: "update the terraform module for the new VPC", routingState: RoutingState())

        guard case .route(let session, _) = result else {
            Issue.record("Expected .route, got \(result)")
            return
        }
        #expect(session == "staging-infra")
        #expect(engine.callCount == 0)
    }

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

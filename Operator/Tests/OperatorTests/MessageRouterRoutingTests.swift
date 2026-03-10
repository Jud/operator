import AVFoundation
import Foundation
import Testing

@testable import OperatorCore

internal final class CountingRoutingEngine: RoutingEngine, @unchecked Sendable {
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

internal struct SessionFixture {
    let name: String
    let tty: String
    let cwd: String
    let ctx: String
    let summary: String
    let recent: String
}

internal let heuristicFixtures: [SessionFixture] = [
    SessionFixture(
        name: "ui-shell",
        tty: "/dev/ttys101",
        cwd: "/Users/jud/work/operator-ui",
        ctx: "Branch feat/settings-loading",
        summary: "Editing settings.tsx, LoadingCard.tsx, dashboard.css for React.",
        recent: "Tightening dashboard skeletons, CSS grid, button spacing in React shell."
    ),
    SessionFixture(
        name: "profile-api",
        tty: "/dev/ttys102",
        cwd: "/Users/jud/work/operator-api",
        ctx: "Branch feat/profile-endpoints",
        summary: "Editing profile.py, user_profile.py, profile.sql for REST and Postgres.",
        recent: "Adding preferences endpoint, fixing database pool, optimizing SQL query."
    ),
    SessionFixture(
        name: "staging-infra",
        tty: "/dev/ttys103",
        cwd: "/Users/jud/work/operator-infra",
        ctx: "Branch chore/staging-network",
        summary: "Editing main.tf, modules/vpc, modules/iam_policy for Terraform rollout.",
        recent: "Updating Terraform networking, VPC layout, IAM policy, S3 logging."
    )
]

/// Registers all `heuristicFixtures` sessions into the given registry.
///
/// Shared across test files that need the 3-session heuristic scenario.
internal func registerHeuristicFixtures(into registry: SessionRegistry) async {
    for fixture in heuristicFixtures {
        await registry.register(
            name: fixture.name,
            tty: fixture.tty,
            cwd: fixture.cwd,
            context: fixture.ctx
        )
        await registry.updateContext(
            tty: fixture.tty,
            summary: fixture.summary,
            recentMessages: [
                SessionMessage(role: "assistant", text: fixture.recent)
            ]
        )
    }
}

// MARK: - Routing Tests

@Suite("MessageRouter - Routing")
internal struct MessageRouterRoutingTests {
    /// Creates a router with the given session names (and optional per-session cwds).
    private func makeRouter(
        sessions: [SessionFixture] = []
    ) async -> (MessageRouter, SessionRegistry) {
        let registry = SessionRegistry(voiceManager: VoiceManager())
        for session in sessions {
            await registry.register(name: session.name, tty: session.tty, cwd: session.cwd, context: nil)
        }
        return (MessageRouter(registry: registry), registry)
    }

    private func makeHeuristicRouter(
        engine: CountingRoutingEngine
    ) async -> MessageRouter {
        let registry = SessionRegistry(voiceManager: VoiceManager())
        await registerHeuristicFixtures(into: registry)
        return MessageRouter(registry: registry, engine: engine)
    }

    @Test("single session routes directly without routing overhead")
    func singleSessionBypass() async {
        let (router, _) = await makeRouter(sessions: [
            SessionFixture(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", ctx: "", summary: "", recent: "")
        ])

        let result = await router.route(
            text: "make the button bigger",
            routingState: RoutingState()
        )
        guard case .route(let session, let message) = result else {
            Issue.record("Expected .route, got \(result)")
            return
        }
        #expect(session == "sudo")
        #expect(message == "make the button bigger")
    }

    @Test("returns noSessions when no sessions registered")
    func noSessionsRegistered() async {
        let (router, _) = await makeRouter()

        let result = await router.route(text: "fix the build", routingState: RoutingState())
        guard case .noSessions = result else {
            Issue.record("Expected .noSessions, got \(result)")
            return
        }
    }

    @Test("session affinity routes to last target within 15s window")
    func sessionAffinityWithinWindow() async {
        let (router, _) = await makeRouter(sessions: [
            SessionFixture(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp/sudo", ctx: "", summary: "", recent: ""),
            SessionFixture(
                name: "frontend",
                tty: "/dev/ttys002",
                cwd: "/tmp/frontend",
                ctx: "",
                summary: "",
                recent: ""
            )
        ])

        var state = RoutingState()
        state.recordRoute(
            text: "fix the build",
            session: "sudo",
            timestamp: Date()
        )

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

        let result = await router.route(
            text: "tighten the button spacing in the React shell",
            routingState: RoutingState()
        )

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

        let result = await router.route(
            text: "fix the database connection pool",
            routingState: RoutingState()
        )

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

        let result = await router.route(
            text: "update the terraform module for the new VPC",
            routingState: RoutingState()
        )

        guard case .route(let session, _) = result else {
            Issue.record("Expected .route, got \(result)")
            return
        }
        #expect(session == "staging-infra")
        #expect(engine.callCount == 0)
    }
}

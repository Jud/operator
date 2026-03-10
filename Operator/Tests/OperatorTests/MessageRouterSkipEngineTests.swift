import AVFoundation
import Foundation
import Testing

@testable import OperatorCore

/// Bundles objects for skipEngine tests.
private struct SkipEngineTestContext {
    let router: MessageRouter
    let registry: SessionRegistry
    let engine: CountingRoutingEngine
}

@Suite("MessageRouter - routeSkipEngine")
internal struct MessageRouterSkipEngineTests {
    // MARK: - Helpers

    private func makeRouter(engine: CountingRoutingEngine = CountingRoutingEngine()) async -> SkipEngineTestContext {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)
        let router = MessageRouter(registry: registry, engine: engine)
        return SkipEngineTestContext(router: router, registry: registry, engine: engine)
    }

    // MARK: - Keyword Match

    @Test("skipEngine: keyword match returns .route")
    func keywordMatchReturnsRoute() async {
        let ctx = await makeRouter()
        await ctx.registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)

        let result = await ctx.router.routeSkipEngine(
            text: "tell sudo to fix the build",
            routingState: RoutingState()
        )
        guard case .route(let session, let message) = result else {
            Issue.record("Expected .route, got \(result)")
            return
        }
        #expect(session == "sudo")
        #expect(message == "to fix the build")
        #expect(ctx.engine.callCount == 0)
    }

    // MARK: - Single Session Bypass

    @Test("skipEngine: single session bypass works")
    func singleSessionBypass() async {
        let ctx = await makeRouter()
        await ctx.registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)

        let result = await ctx.router.routeSkipEngine(
            text: "make the button bigger",
            routingState: RoutingState()
        )
        guard case .route(let session, let message) = result else {
            Issue.record("Expected .route, got \(result)")
            return
        }
        #expect(session == "sudo")
        #expect(message == "make the button bigger")
        #expect(ctx.engine.callCount == 0)
    }

    // MARK: - Affinity

    @Test("skipEngine: affinity target returns .route")
    func affinityReturnsRoute() async {
        let ctx = await makeRouter()
        await ctx.registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp/sudo", context: nil)
        await ctx.registry.register(
            name: "frontend",
            tty: "/dev/ttys002",
            cwd: "/tmp/frontend",
            context: nil
        )

        var state = RoutingState()
        state.recordRoute(text: "previous", session: "sudo", timestamp: Date())

        let result = await ctx.router.routeSkipEngine(text: "looks good", routingState: state)
        guard case .route(let session, _) = result else {
            Issue.record("Expected .route, got \(result)")
            return
        }
        #expect(session == "sudo")
        #expect(ctx.engine.callCount == 0)
    }

    // MARK: - Operator Commands

    @Test("skipEngine: operator command returns .operatorCommand")
    func operatorCommandReturns() async {
        let ctx = await makeRouter()
        await ctx.registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)

        let result = await ctx.router.routeSkipEngine(
            text: "operator status",
            routingState: RoutingState()
        )
        guard case .operatorCommand(.status) = result else {
            Issue.record("Expected .operatorCommand(.status), got \(result)")
            return
        }
    }

    // MARK: - Not Confident

    @Test("skipEngine: no match returns .notConfident with original text")
    func noMatchReturnsNotConfident() async {
        let ctx = await makeRouter()
        await ctx.registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)
        await ctx.registry.register(
            name: "frontend",
            tty: "/dev/ttys002",
            cwd: "/tmp",
            context: nil
        )

        let result = await ctx.router.routeSkipEngine(
            text: "some ambiguous message that matches nothing",
            routingState: RoutingState()
        )
        guard case .notConfident(let text) = result else {
            Issue.record("Expected .notConfident, got \(result)")
            return
        }
        #expect(text == "some ambiguous message that matches nothing")
        #expect(ctx.engine.callCount == 0)
    }

    @Test("skipEngine: no sessions returns .noSessions")
    func noSessionsReturnsNoSessions() async {
        let ctx = await makeRouter()

        let result = await ctx.router.routeSkipEngine(
            text: "fix the build",
            routingState: RoutingState()
        )
        guard case .noSessions = result else {
            Issue.record("Expected .noSessions, got \(result)")
            return
        }
        #expect(ctx.engine.callCount == 0)
    }

    // MARK: - Never Spawns Engine

    @Test("skipEngine: never invokes routing engine regardless of outcome")
    func neverInvokesRoutingEngine() async {
        let engine = CountingRoutingEngine()
        let ctx = await makeRouter(engine: engine)
        // Register multiple sessions to avoid single-session bypass
        await ctx.registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)
        await ctx.registry.register(name: "frontend", tty: "/dev/ttys002", cwd: "/tmp", context: nil)
        await ctx.registry.register(name: "backend", tty: "/dev/ttys003", cwd: "/tmp", context: nil)

        // Multiple calls with different messages
        _ = await ctx.router.routeSkipEngine(text: "hello", routingState: RoutingState())
        _ = await ctx.router.routeSkipEngine(text: "fix the database", routingState: RoutingState())
        _ = await ctx.router.routeSkipEngine(text: "deploy to staging", routingState: RoutingState())
        _ = await ctx.router.routeSkipEngine(
            text: "what's the status",
            routingState: RoutingState()
        )

        #expect(ctx.engine.callCount == 0, "routeSkipEngine must never invoke the routing engine")
    }

    // MARK: - Existing route() Unaffected

    @Test("existing route() method still invokes routing engine for ambiguous messages")
    func existingRouteStillUsesEngine() async {
        let engine = CountingRoutingEngine()
        let ctx = await makeRouter(engine: engine)
        await ctx.registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)
        await ctx.registry.register(name: "frontend", tty: "/dev/ttys002", cwd: "/tmp", context: nil)

        // This should fall through to claude -p (routing engine)
        _ = await ctx.router.route(
            text: "some ambiguous message",
            routingState: RoutingState()
        )

        #expect(ctx.engine.callCount > 0, "regular route() should invoke routing engine for ambiguous messages")
    }

    // MARK: - Heuristic Match

    private func makeHeuristicContext() async -> SkipEngineTestContext {
        let engine = CountingRoutingEngine()
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)
        await registerHeuristicFixtures(into: registry)
        let router = MessageRouter(registry: registry, engine: engine)
        return SkipEngineTestContext(router: router, registry: registry, engine: engine)
    }

    @Test("skipEngine: strong heuristic match returns .route")
    func strongHeuristicMatchReturnsRoute() async {
        let ctx = await makeHeuristicContext()

        let result = await ctx.router.routeSkipEngine(
            text: "tighten the button spacing in the React shell",
            routingState: RoutingState()
        )
        guard case .route(let session, _) = result else {
            Issue.record("Expected .route from strong heuristic, got \(result)")
            return
        }
        #expect(session == "ui-shell")
        #expect(ctx.engine.callCount == 0)
    }
}

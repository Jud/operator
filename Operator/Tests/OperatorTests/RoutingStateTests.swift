import Foundation
import Testing

@testable import OperatorCore

@Suite("RoutingState")
internal struct RoutingStateTests {
    // MARK: - Routing History Ring Buffer

    @Test("recordRoute appends to history")
    func recordRouteAppendsToHistory() {
        var state = RoutingState()
        state.recordRoute(text: "fix the build", session: "sudo")

        #expect(state.routedMessages.count == 1)
        #expect(state.routedMessages[0].text == "fix the build")
        #expect(state.routedMessages[0].session == "sudo")
    }

    @Test("routing history capped at 10 entries")
    func routingHistoryCappedAtTen() {
        var state = RoutingState()

        for i in 0..<15 {
            state.recordRoute(text: "message \(i)", session: "agent\(i)")
        }

        #expect(state.routedMessages.count == 10)
        // Oldest entries (0-4) should have been evicted; entries 5-14 remain.
        #expect(state.routedMessages[0].text == "message 5")
        #expect(state.routedMessages[9].text == "message 14")
    }

    @Test("recording route updates affinity fields")
    func recordRouteUpdatesAffinity() {
        var state = RoutingState()

        let ts = Date(timeIntervalSince1970: 1_000)

        state.recordRoute(text: "test", session: "frontend", timestamp: ts)

        #expect(state.lastRoutedSession == "frontend")
        #expect(state.lastRoutedTimestamp == ts)
    }

    // MARK: - Session Affinity

    @Test("affinity active within 15-second window")
    func affinityActiveWithinWindow() {
        var state = RoutingState()

        let routeTime = Date()

        state.recordRoute(text: "test", session: "sudo", timestamp: routeTime)

        let checkTime = routeTime.addingTimeInterval(10)
        #expect(state.isAffinityActive(now: checkTime) == true)
        #expect(state.affinityTarget(now: checkTime) == "sudo")
    }

    @Test("affinity expires after 15-second window")
    func affinityExpiresAfterWindow() {
        var state = RoutingState()

        let routeTime = Date()

        state.recordRoute(text: "test", session: "sudo", timestamp: routeTime)

        let checkTime = routeTime.addingTimeInterval(16)
        #expect(state.isAffinityActive(now: checkTime) == false)
        #expect(state.affinityTarget(now: checkTime) == nil)
    }

    @Test("affinity at exactly 15 seconds is expired")
    func affinityAtBoundaryIsExpired() {
        var state = RoutingState()

        let routeTime = Date()

        state.recordRoute(text: "test", session: "sudo", timestamp: routeTime)

        let checkTime = routeTime.addingTimeInterval(15)
        #expect(state.isAffinityActive(now: checkTime) == false)
    }

    @Test("affinity inactive with no routing history")
    func affinityInactiveWithNoHistory() {
        let state = RoutingState()
        #expect(state.isAffinityActive() == false)
        #expect(state.affinityTarget() == nil)
    }

    @Test("affinity tracks most recent session")
    func affinityTracksMostRecentSession() {
        var state = RoutingState()

        let now = Date()

        state.recordRoute(text: "msg1", session: "sudo", timestamp: now)
        state.recordRoute(text: "msg2", session: "frontend", timestamp: now.addingTimeInterval(1))

        let checkTime = now.addingTimeInterval(5)
        #expect(state.affinityTarget(now: checkTime) == "frontend")
    }

    // MARK: - Pending Interruption

    @Test("store and clear interruption")
    func storeAndClearInterruption() {
        var state = RoutingState()
        #expect(state.pendingInterruption == nil)

        state.storeInterruption(heardText: "I finished the", unheardText: "refactor of the parser", session: "sudo")

        #expect(state.pendingInterruption != nil)
        #expect(state.pendingInterruption?.heardText == "I finished the")
        #expect(state.pendingInterruption?.unheardText == "refactor of the parser")
        #expect(state.pendingInterruption?.session == "sudo")

        state.clearInterruption()
        #expect(state.pendingInterruption == nil)
    }

    // MARK: - Pending Clarification

    @Test("store and clear clarification")
    func storeAndClearClarification() {
        var state = RoutingState()
        #expect(state.pendingClarification == nil)

        state.storeClarification(candidates: ["sudo", "frontend"], originalText: "looks good, merge it")

        #expect(state.pendingClarification != nil)
        #expect(state.pendingClarification?.candidates == ["sudo", "frontend"])
        #expect(state.pendingClarification?.originalText == "looks good, merge it")

        state.clearClarification()
        #expect(state.pendingClarification == nil)
    }

    // MARK: - Clear All Pending

    @Test("clearAllPending clears both interruption and clarification")
    func clearAllPendingClearsBoth() {
        var state = RoutingState()
        state.storeInterruption(heardText: "heard", unheardText: "unheard", session: "sudo")
        state.storeClarification(candidates: ["a", "b"], originalText: "text")

        #expect(state.pendingInterruption != nil)
        #expect(state.pendingClarification != nil)

        state.clearAllPending()

        #expect(state.pendingInterruption == nil)
        #expect(state.pendingClarification == nil)
    }
}

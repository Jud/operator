import Foundation
import OperatorCore

// MARK: - New Scenarios (Cancel + Hook Lifecycle)

extension E2EHarness {
    // MARK: 11. TriggerCancel

    /// Start recording, press Escape via mock cancel, verify state returns to IDLE
    /// without delivering any message.
    func testTriggerCancel() async -> ScenarioResult {
        let name = "TriggerCancel"
        guard let sm = stateMachine else {
            return ScenarioResult(name: name, passed: false, detail: "stateMachine is nil")
        }

        let countBefore = mockSpeechManager.spokenMessages.count
        mockTranscriber.nextTranscription = "tell alpha this should be cancelled"

        mockTrigger.simulateTriggerStart()
        try? await Task.sleep(nanoseconds: 100_000_000)

        guard sm.currentState == .listening else {
            return ScenarioResult(
                name: name,
                passed: false,
                detail: "Expected LISTENING after triggerStart, got \(sm.currentState)"
            )
        }

        mockTrigger.simulateTriggerCancel()
        try? await Task.sleep(nanoseconds: 500_000_000)

        guard sm.currentState == .idle else {
            return ScenarioResult(
                name: name,
                passed: false,
                detail: "Expected IDLE after cancel, got \(sm.currentState)"
            )
        }

        let newMessages = mockSpeechManager.spokenMessages.dropFirst(countBefore)
        let hasDelivery = newMessages.contains { $0.text.contains("this should be cancelled") }

        guard !hasDelivery else {
            return ScenarioResult(name: name, passed: false, detail: "Cancelled message was still delivered")
        }
        return ScenarioResult(name: name, passed: true, detail: "Cancel returned to IDLE without delivery")
    }

    // MARK: 12. HookSessionLifecycle

    /// Test the hook HTTP endpoints for session lifecycle management.
    func testHookSessionLifecycle() async -> ScenarioResult {
        let name = "HookSessionLifecycle"
        guard let reg = registry else {
            return ScenarioResult(name: name, passed: false, detail: "registry is nil")
        }

        let sid = "e2e-hook-test-\(UUID().uuidString.prefix(8))"
        let tty = "/dev/ttys999"

        // Ensure cleanup if test fails after session-start but before session-end
        defer {
            Task { await reg.deregister(tty: tty) }
        }

        do {
            try await httpPost(
                path: "/hook/session-start",
                body: ["session_id": sid, "tty": tty, "cwd": "/tmp/hook-test"]
            )
            let afterStart = await reg.allSessions()
            guard afterStart.contains(where: { $0.tty == tty }) else {
                return ScenarioResult(name: name, passed: false, detail: "Session not found after start")
            }

            // Smoke-test /hook/stop: verifies it doesn't error. Side-effect (context update)
            // is internal to SessionRegistry and not directly observable here.
            try await httpPost(
                path: "/hook/stop",
                body: ["session_id": sid, "tty": tty, "last_assistant_message": "I fixed the bug"]
            )

            try await httpPost(
                path: "/hook/session-end",
                body: ["session_id": sid, "tty": tty]
            )
            let afterEnd = await reg.allSessions()
            guard !afterEnd.contains(where: { $0.tty == tty }) else {
                return ScenarioResult(name: name, passed: false, detail: "Session still exists after end")
            }

            return ScenarioResult(name: name, passed: true, detail: "Hook lifecycle working")
        } catch {
            return ScenarioResult(name: name, passed: false, detail: "Hook request failed: \(error)")
        }
    }
}

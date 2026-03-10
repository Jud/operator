import AVFoundation
import Foundation
import Testing

@testable import OperatorCore

// MARK: - Mock Implementations

internal final class MockTerminalBridge: TerminalBridge, @unchecked Sendable {
    var writeResult: Bool = true
    var writeError: Error?
    var lastWrittenIdentifier: TerminalIdentifier?
    var lastWrittenText: String?

    func writeToSession(identifier: TerminalIdentifier, text: String) async throws -> Bool {
        lastWrittenIdentifier = identifier
        lastWrittenText = text
        if let error = writeError { throw error }
        return writeResult
    }

    func discoverSessions() async throws -> [DiscoveredSession] { [] }
}

@MainActor
internal final class MockAudioFeedback: AudioFeedbackProviding {
    var playedCues: [AudioCue] = []

    func play(_ cue: AudioCue) { playedCues.append(cue) }
}

// MARK: - DeliveryCoordinator Tests

/// Helper that bundles the objects created for each DeliveryCoordinator test.
internal struct DeliveryTestContext {
    let coordinator: DeliveryCoordinator
    let registry: SessionRegistry
    let bridge: MockTerminalBridge
    let feedback: MockAudioFeedback
}

@Suite("DeliveryCoordinator")
@MainActor
internal struct DeliveryCoordinatorTests {
    // MARK: - Helpers

    private func makeCoordinator(
        bridge: MockTerminalBridge = MockTerminalBridge(),
        feedback: MockAudioFeedback = MockAudioFeedback()
    ) -> DeliveryTestContext {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)
        let coordinator = DeliveryCoordinator(
            terminalBridge: bridge,
            registry: registry,
            feedback: feedback
        )
        return DeliveryTestContext(
            coordinator: coordinator,
            registry: registry,
            bridge: bridge,
            feedback: feedback
        )
    }

    // MARK: - deliver

    @Test("deliver to registered session succeeds with .success")
    func deliverToRegisteredSucceeds() async {
        let ctx = makeCoordinator()
        await ctx.registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)

        let result = await ctx.coordinator.deliver("fix the build", to: "sudo")
        guard case .success(let message, let session) = result else {
            Issue.record("Expected .success, got \(result)")
            return
        }
        #expect(message == "fix the build")
        #expect(session == "sudo")
    }

    @Test("deliver to unregistered session returns .sessionNotFound")
    func deliverToUnregisteredReturnsNotFound() async {
        let ctx = makeCoordinator()

        let result = await ctx.coordinator.deliver("hello", to: "nonexistent")
        guard case .sessionNotFound(let session) = result else {
            Issue.record("Expected .sessionNotFound, got \(result)")
            return
        }
        #expect(session == "nonexistent")
    }

    @Test("deliver when bridge returns false returns .writeFailed")
    func deliverBridgeReturnsFalse() async {
        let bridge = MockTerminalBridge()
        bridge.writeResult = false
        let ctx = makeCoordinator(bridge: bridge)
        await ctx.registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)

        let result = await ctx.coordinator.deliver("hello", to: "sudo")
        guard case .writeFailed(let session) = result else {
            Issue.record("Expected .writeFailed, got \(result)")
            return
        }
        #expect(session == "sudo")
    }

    @Test("deliver when bridge throws terminalNotRunning for iTerm returns .terminalNotRunning")
    func deliverBridgeThrowsItermNotRunning() async {
        let bridge = MockTerminalBridge()
        bridge.writeError = TerminalBridgeError.terminalNotRunning(terminal: .iterm)
        let ctx = makeCoordinator(bridge: bridge)
        await ctx.registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)

        let result = await ctx.coordinator.deliver("hello", to: "sudo")
        guard case .terminalNotRunning(let session) = result else {
            Issue.record("Expected .terminalNotRunning, got \(result)")
            return
        }
        #expect(session == "sudo")
    }

    @Test("deliver when bridge throws terminalNotRunning for Ghostty returns .terminalNotRunning")
    func deliverBridgeThrowsGhosttyNotRunning() async {
        let bridge = MockTerminalBridge()
        bridge.writeError = TerminalBridgeError.terminalNotRunning(terminal: .ghostty)
        let ctx = makeCoordinator(bridge: bridge)
        await ctx.registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)

        let result = await ctx.coordinator.deliver("hello", to: "sudo")
        guard case .terminalNotRunning(let session) = result else {
            Issue.record("Expected .terminalNotRunning, got \(result)")
            return
        }
        #expect(session == "sudo")
    }

    @Test("deliver when bridge throws other error returns .deliveryError")
    func deliverBridgeThrowsOtherError() async {
        let bridge = MockTerminalBridge()
        bridge.writeError = NSError(domain: "test", code: 42)
        let ctx = makeCoordinator(bridge: bridge)
        await ctx.registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)

        let result = await ctx.coordinator.deliver("hello", to: "sudo")
        guard case .deliveryError(let session) = result else {
            Issue.record("Expected .deliveryError, got \(result)")
            return
        }
        #expect(session == "sudo")
    }

    @Test("successful deliver plays .delivered feedback tone")
    func deliverPlaysDeliveredTone() async {
        let feedback = MockAudioFeedback()
        let ctx = makeCoordinator(feedback: feedback)
        await ctx.registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)

        _ = await ctx.coordinator.deliver("hello", to: "sudo")
        #expect(feedback.playedCues == [.delivered])
    }

    // MARK: - deliverDirect

    @Test("deliverDirect to registered session returns true")
    func deliverDirectToRegisteredReturnsTrue() async {
        let ctx = makeCoordinator()
        await ctx.registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)

        let result = await ctx.coordinator.deliverDirect("hello", to: "sudo")
        #expect(result == true)
    }

    @Test("deliverDirect to unregistered session returns false")
    func deliverDirectToUnregisteredReturnsFalse() async {
        let ctx = makeCoordinator()

        let result = await ctx.coordinator.deliverDirect("hello", to: "nonexistent")
        #expect(result == false)
    }

    @Test("deliverDirect doesn't play feedback tone")
    func deliverDirectNoFeedback() async {
        let feedback = MockAudioFeedback()
        let ctx = makeCoordinator(feedback: feedback)
        await ctx.registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)

        _ = await ctx.coordinator.deliverDirect("hello", to: "sudo")
        #expect(feedback.playedCues.isEmpty)
    }
}

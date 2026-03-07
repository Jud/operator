import Foundation
import OperatorCore

/// Mock trigger source for E2E testing that allows programmatic trigger activation.
///
/// Call simulateTriggerStart() and simulateTriggerStop() to simulate push-to-talk
/// key press and release events. The callbacks are dispatched to @MainActor
/// just like the real FNKeyTrigger.
internal final class MockTriggerSource: TriggerSource {
    var onStart: (@Sendable @MainActor () -> Void)?
    var onStop: (@Sendable @MainActor () -> Void)?

    /// Simulate the user pressing the push-to-talk key.
    @MainActor
    func simulateTriggerStart() {
        onStart?()
    }

    /// Simulate the user releasing the push-to-talk key.
    @MainActor
    func simulateTriggerStop() {
        onStop?()
    }
}

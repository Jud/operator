import Foundation

/// Abstraction for push-to-talk activation sources.
///
/// The protocol decouples the state machine from the specific input mechanism,
/// enabling multiple trigger implementations: FN/Globe key (primary), configurable
/// global hotkey (secondary), and future Bluetooth media button.
///
/// Callbacks are dispatched on the main thread so the StateMachine (@MainActor)
/// can process them directly without cross-thread hops.
public protocol TriggerSource: AnyObject {
    /// Called when the push-to-talk key is pressed (voice capture should begin).
    var onStart: (@Sendable @MainActor () -> Void)? { get set }

    /// Called when the push-to-talk key is released (voice capture should end).
    var onStop: (@Sendable @MainActor () -> Void)? { get set }
}

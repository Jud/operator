import Foundation

/// Result of attempting to deliver a message to a Claude Code session via terminal bridge.
///
/// The StateMachine uses this to determine the appropriate state transition
/// and user feedback after a delivery attempt.
public enum DeliveryResult: Sendable {
    /// Message was successfully written to the session.
    case success(message: String, session: String)

    /// The target session was not found in the registry.
    case sessionNotFound(session: String)

    /// The terminal bridge write call returned false (session may have closed).
    case writeFailed(session: String)

    /// The target terminal application is not running.
    case terminalNotRunning(session: String)

    /// An unexpected error occurred during delivery.
    case deliveryError(session: String)
}

/// Coordinates message delivery to Claude Code sessions via the terminal bridge.
///
/// Encapsulates the registry lookup, bridge write, and error classification
/// that were previously inline in StateMachine. The StateMachine calls the
/// coordinator and handles state transitions based on the result.
///
/// Reference: technical-spec.md Component 5; research doc "Proven Recipe"
@MainActor
public struct DeliveryCoordinator {
    private static let logger = Log.logger(for: "DeliveryCoordinator")

    private let terminalBridge: any TerminalBridge
    private let registry: SessionRegistry
    private let feedback: any AudioFeedbackProviding

    /// Creates a new delivery coordinator.
    ///
    /// - Parameters:
    ///   - terminalBridge: Bridge for terminal text delivery.
    ///   - registry: Actor managing registered Claude Code sessions.
    ///   - feedback: Non-verbal audio tone player.
    public init(
        terminalBridge: any TerminalBridge,
        registry: SessionRegistry,
        feedback: any AudioFeedbackProviding
    ) {
        self.terminalBridge = terminalBridge
        self.registry = registry
        self.feedback = feedback
    }

    /// Deliver a message to a specific session, returning the result.
    ///
    /// Performs the registry lookup, bridge write, and error classification.
    /// Does not perform any state transitions -- that is the caller's responsibility.
    ///
    /// - Parameters:
    ///   - message: The message text to deliver.
    ///   - session: The target session name.
    /// - Returns: A `DeliveryResult` indicating the outcome.
    public func deliver(_ message: String, to session: String) async -> DeliveryResult {
        guard let sessionState = await registry.session(named: session) else {
            Self.logger.error("Session '\(session)' not found in registry for delivery")
            return .sessionNotFound(session: session)
        }

        do {
            let success = try await terminalBridge.writeToSession(
                identifier: sessionState.identifier,
                text: message
            )
            guard success else {
                Self.logger.error("writeToSession returned false for \(session)")
                return .writeFailed(session: session)
            }
            Self.logger.info("Delivered message to \(session)")
            feedback.play(.delivered)
            return .success(message: message, session: session)
        } catch ITermBridgeError.itermNotRunning {
            Self.logger.error("Terminal not running during delivery to \(session)")
            return .terminalNotRunning(session: session)
        } catch GhosttyBridgeError.ghosttyNotRunning {
            Self.logger.error("Terminal not running during delivery to \(session)")
            return .terminalNotRunning(session: session)
        } catch {
            Self.logger.error("Delivery failed for \(session): \(error)")
            return .deliveryError(session: session)
        }
    }

    /// Deliver a message directly without state machine coordination.
    ///
    /// Used for multi-send scenarios (e.g., "send to all" during clarification).
    /// Records the route on success but does not play feedback tones.
    ///
    /// - Parameters:
    ///   - message: The message text to deliver.
    ///   - session: The target session name.
    /// - Returns: `true` if delivery succeeded.
    @discardableResult
    public func deliverDirect(_ message: String, to session: String) async -> Bool {
        guard let sessionState = await registry.session(named: session) else {
            Self.logger.warning("Session '\(session)' not found for direct delivery")
            return false
        }
        do {
            _ = try await terminalBridge.writeToSession(
                identifier: sessionState.identifier,
                text: message
            )
            Self.logger.info("Direct delivery to \(session) succeeded")
            return true
        } catch {
            Self.logger.error("Direct delivery to \(session) failed: \(error)")
            return false
        }
    }
}

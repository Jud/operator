import Foundation

/// A single entry in the routing history, recording which session received which message.
///
/// Used to provide claude -p with recent routing context for more accurate routing decisions.
/// Stored as a struct (not a tuple) for Codable conformance and Sendable safety.
///
/// Reference: technical-spec.md, Session State Model (RoutingState.routedMessages)
public struct RoutedMessage: Codable, Sendable {
    /// The message text that was routed.
    public let text: String
    /// The target session name.
    public let session: String
    /// When the message was routed.
    public let timestamp: Date
}

/// Tracks the state of an interrupted audio playback for intelligent resumption.
///
/// When the user presses push-to-talk while an agent's response is being spoken,
/// SpeechManager captures exactly which text was heard and which was not. This
/// information is used by InterruptionHandler (fast-path keywords) or claude -p
/// (ambiguous cases) to decide whether to replay, skip, or route the new message.
///
/// Reference: technical-spec.md, Component 3, "Interruption Handler";
/// design.md Section 3.1.7
public struct PendingInterruption: Sendable {
    /// The portion of the message that was spoken before interruption.
    public let heardText: String
    /// The remaining portion that was not yet spoken.
    public let unheardText: String
    /// The session whose response was interrupted.
    public let session: String
}

/// Tracks state during an ambiguous routing clarification flow.
///
/// When routing produces multiple candidate sessions with insufficient confidence,
/// Operator speaks a clarification question ("Was that for sudo or frontend?") and
/// enters the CLARIFYING state. This struct holds the candidate list and the original
/// user text so the clarification response can complete the delivery.
///
/// Reference: technical-spec.md, Component 3, "Ambiguous fallback";
/// design.md Section 3.1.6
public struct PendingClarification: Sendable {
    /// The session names that are potential targets.
    public let candidates: [String]
    /// The user's original message text to deliver once resolved.
    public let originalText: String
}

/// Routing history, session affinity, and pending state for the message router.
///
/// Owned by the StateMachine (@MainActor) and read by MessageRouter for routing
/// decisions. This is a pure value type with no external dependencies -- all state
/// mutations happen through explicit method calls.
///
/// Session affinity: when the user sends rapid-fire messages to the same agent,
/// subsequent messages within a 15-second window skip the routing chain and go
/// directly to the last-routed session (unless the user explicitly names a different
/// agent). This eliminates 1-2s of claude -p latency for follow-up messages.
///
/// Reference: technical-spec.md, Session State Model (RoutingState struct);
/// design.md Section 3.1.11; Key Design Decision #6 (15s affinity window)
public struct RoutingState: Sendable {
    /// Maximum number of routing history entries retained for claude -p context.
    private static let maxRoutedMessages = 10

    /// Session affinity timeout in seconds.
    ///
    /// Messages sent within this window after the last routed message will go to
    /// the same session without re-routing, unless the user explicitly names a
    /// different agent.
    ///
    /// Reference: technical-spec.md, Component 3; charter.md "Will Do" item 17;
    /// requirements.md AC-03.4
    public static let affinityTimeoutSeconds: TimeInterval = 15

    /// Recent routing history: last 10 messages with their target session and timestamp.
    ///
    /// Provided to claude -p as context for routing decisions.
    public private(set) var routedMessages: [RoutedMessage] = []

    /// State of an interrupted audio playback, if any.
    ///
    /// Set when the user activates push-to-talk while an agent's response is being
    /// spoken. Cleared after the interruption is resolved (skip, replay, or route).
    public var pendingInterruption: PendingInterruption?

    /// State of an in-progress clarification, if any.
    ///
    /// Set when routing produces ambiguous results and Operator asks the user to
    /// clarify. Cleared after the clarification resolves (user picks a target,
    /// times out, or cancels).
    public var pendingClarification: PendingClarification?

    /// The session name that received the most recent routed message.
    ///
    /// Used together with lastRoutedTimestamp for session affinity.
    public private(set) var lastRoutedSession: String?

    /// Timestamp of the most recent routed message.
    ///
    /// Used together with lastRoutedSession for session affinity.
    public private(set) var lastRoutedTimestamp: Date?

    /// Creates a new empty routing state.
    public init() {}

    // MARK: - Routing History

    /// Record a successfully routed message in the history.
    ///
    /// - Parameters:
    ///   - text: The message text that was routed.
    ///   - session: The target session name.
    ///   - timestamp: When the message was routed. Defaults to now.
    public mutating func recordRoute(text: String, session: String, timestamp: Date = Date()) {
        let entry = RoutedMessage(text: text, session: session, timestamp: timestamp)
        routedMessages.append(entry)

        if routedMessages.count > Self.maxRoutedMessages {
            routedMessages.removeFirst(routedMessages.count - Self.maxRoutedMessages)
        }

        lastRoutedSession = session
        lastRoutedTimestamp = timestamp
    }

    // MARK: - Session Affinity

    /// Check whether session affinity is currently active.
    ///
    /// - Parameter now: The current time. Defaults to Date(). Parameterized for testing.
    /// - Returns: `true` if the last routed message was within the affinity window.
    public func isAffinityActive(now: Date = Date()) -> Bool {
        guard let timestamp = lastRoutedTimestamp else {
            return false
        }
        return now.timeIntervalSince(timestamp) < Self.affinityTimeoutSeconds
    }

    /// Get the affinity target session, if affinity is currently active.
    ///
    /// - Parameter now: The current time. Defaults to Date(). Parameterized for testing.
    /// - Returns: The session name to use for affinity routing, or nil.
    public func affinityTarget(now: Date = Date()) -> String? {
        guard isAffinityActive(now: now) else {
            return nil
        }
        return lastRoutedSession
    }

    // MARK: - Pending State Management

    /// Store an interruption for later resolution by InterruptionHandler.
    ///
    /// - Parameters:
    ///   - heardText: The portion of the message that was spoken before interruption.
    ///   - unheardText: The remaining portion that was not yet spoken.
    ///   - session: The session whose response was interrupted.
    public mutating func storeInterruption(heardText: String, unheardText: String, session: String) {
        pendingInterruption = PendingInterruption(
            heardText: heardText,
            unheardText: unheardText,
            session: session
        )
    }

    /// Clear the pending interruption after it has been resolved.
    public mutating func clearInterruption() {
        pendingInterruption = nil
    }

    /// Store a clarification request for later resolution.
    ///
    /// - Parameters:
    ///   - candidates: The session names that are potential targets.
    ///   - originalText: The user's original message text to deliver once resolved.
    public mutating func storeClarification(candidates: [String], originalText: String) {
        pendingClarification = PendingClarification(
            candidates: candidates,
            originalText: originalText
        )
    }

    /// Clear the pending clarification after it has been resolved or timed out.
    public mutating func clearClarification() {
        pendingClarification = nil
    }

    /// Clear all pending state (both interruption and clarification).
    ///
    /// Called when entering LISTENING state ("latest utterance wins" rule) to
    /// discard any unresolved pending state from the previous interaction.
    public mutating func clearAllPending() {
        pendingInterruption = nil
        pendingClarification = nil
    }
}

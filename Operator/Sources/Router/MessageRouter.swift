import Foundation
import OSLog

/// Result of routing a user's transcribed message through the priority chain.
///
/// Each case represents a different outcome from the routing process, consumed
/// by the StateMachine to determine the next state transition and action.
///
/// Reference: technical-spec.md Component 3, "Routing Priority"
public enum RoutingResult: Sendable {
    /// The message is an operator command (status, list, replay, help, etc.)
    /// handled directly by the daemon without routing to any Claude Code session.
    case operatorCommand(OperatorCommand)

    /// The message should be delivered to a specific session with confidence.
    /// Includes the resolved session name and the message text to deliver
    /// (which may differ from the original if keyword extraction stripped a prefix).
    case route(session: String, message: String)

    /// Routing is ambiguous. The StateMachine should enter CLARIFYING state,
    /// speak the question, and wait for the user's clarification response.
    case clarify(candidates: [String], question: String, originalText: String)

    /// No sessions are registered; the message cannot be routed.
    case noSessions

    /// The `claude` CLI binary was not found. Smart routing is unavailable.
    case cliNotFound
}

/// Recognized operator voice commands handled directly by the daemon.
///
/// Reference: technical-spec.md Component 7
public enum OperatorCommand: Sendable {
    /// "operator status" / "what's everyone working on"
    case status

    /// "list agents" / "who's running"
    case listAgents

    /// "operator replay" -- re-speak last agent message
    case replay

    /// "what did [name] say" -- re-speak last message from a specific agent
    case replayAgent(name: String)

    /// "what did I miss"
    case whatDidIMiss

    /// "operator help"
    case help
}

/// Result of processing a user's clarification response during the CLARIFYING state.
///
/// The StateMachine uses this to determine whether to deliver, re-ask, or abort.
public enum ClarificationResult: Sendable {
    /// The user identified a target session. Deliver the original message to it.
    case resolved(session: String)

    /// The user wants to send to all candidates.
    case sendToAll

    /// The user cancelled ("never mind", "cancel").
    case cancelled

    /// The user asked to hear the question again ("repeat").
    case repeatQuestion

    /// Could not determine target; re-ask if attempts remain.
    case unresolved
}

/// Routes user messages to the correct Claude Code session using a priority chain.
///
/// The routing priority chain is checked in order:
/// 1. Operator commands (status, list, replay, help) -- handled directly, no session routing
/// 2. Keyword extraction (/(?:tell |hey |@)(\w+)[,:]?\s*(.*)/i) -- explicit agent targeting
/// 3. Single-session bypass -- if only one session registered, route directly
/// 4. Session affinity -- if last routed <15s ago and no explicit agent name, reuse target
/// 5. claude -p smart routing -- pipe session context + message, parse JSON response
/// 6. Ambiguous fallback -- return clarification request with candidate list
///
/// Clarification handling (for CLARIFYING state responses):
/// - Keyword match first (session name in response -> route immediately)
/// - Meta-intents: cancel/never mind, both/all, ordinal (first/second), repeat
/// - claude -p fallback for unrecognized responses
/// - 8-second timeout and max 2 attempts are enforced by the StateMachine, not here
///
/// Reference: technical-spec.md Component 3; design.md Section 3.1.6;
/// requirements.md FR-03, FR-09, FR-13
public class MessageRouter: @unchecked Sendable {
    static let logger = Logger(subsystem: "com.operator.app", category: "MessageRouter")

    /// The session registry providing current session state for routing decisions.
    let registry: SessionRegistry

    /// Creates a new message router with the given session registry.
    public init(registry: SessionRegistry) {
        self.registry = registry
    }
}

// MARK: - Primary Routing

extension MessageRouter {
    /// Route a transcribed user message through the full priority chain.
    ///
    /// - Parameters:
    ///   - text: The transcribed user message.
    ///   - routingState: Current routing state with affinity and history data.
    /// - Returns: A `RoutingResult` indicating how the message should be handled.
    public func route(text: String, routingState: RoutingState) async -> RoutingResult {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let lowered = trimmed.lowercased()

        Self.logger.info("Routing message: \"\(trimmed.prefix(80))\"")

        if let command = matchOperatorCommand(lowered) {
            Self.logger.info("Matched operator command: \(String(describing: command))")
            return .operatorCommand(command)
        }

        let sessions = await registry.allSessions()

        if sessions.isEmpty {
            Self.logger.warning("No sessions registered; cannot route")
            return .noSessions
        }

        let sessionNames = sessions.map { $0.name }

        if let (target, message) = extractKeyword(from: trimmed, sessionNames: sessionNames) {
            Self.logger.info("Keyword match: routing to '\(target)'")
            return .route(session: target, message: message)
        }

        if sessions.count == 1 {
            let session = sessions[0]
            Self.logger.info("Single session bypass: routing to '\(session.name)'")
            return .route(session: session.name, message: trimmed)
        }

        if let affinityTarget = routingState.affinityTarget() {
            if sessionNames.contains(where: { $0 == affinityTarget }) {
                Self.logger.info("Session affinity: routing to '\(affinityTarget)'")
                return .route(session: affinityTarget, message: trimmed)
            }
            Self.logger.debug("Affinity target '\(affinityTarget)' no longer registered; falling through")
        }

        Self.logger.info("Invoking claude -p for smart routing")

        return await claudePipeRoute(
            text: trimmed,
            sessions: sessions,
            routingState: routingState
        )
    }
}

// MARK: - Operator Command Matching

extension MessageRouter {
    /// Regex pattern for "what did [name] say" operator command.
    private static let whatDidSayPattern: NSRegularExpression? = {
        try? NSRegularExpression(
            pattern: #"what did (\w+) say"#,
            options: [.caseInsensitive]
        )
    }()

    /// Operator command patterns matched against the lowercased user utterance.
    private static let operatorCommandMatchers: [(pattern: String, command: OperatorCommand)] = [
        ("operator status", .status),
        ("operator what", .status),
        ("what's everyone working on", .status),
        ("whats everyone working on", .status),
        ("what is everyone working on", .status),
        ("list agents", .listAgents),
        ("list sessions", .listAgents),
        ("who's running", .listAgents),
        ("whos running", .listAgents),
        ("who is running", .listAgents),
        ("operator replay", .replay),
        ("what did i miss", .whatDidIMiss),
        ("operator help", .help)
    ]

    /// Match the user's lowercased utterance against known operator command patterns.
    private func matchOperatorCommand(_ lowered: String) -> OperatorCommand? {
        let range = NSRange(lowered.startIndex..., in: lowered)
        if let pattern = Self.whatDidSayPattern,
            let match = pattern.firstMatch(in: lowered, range: range),
            let nameRange = Range(match.range(at: 1), in: lowered)
        {
            let agentName = String(lowered[nameRange])
            if agentName.lowercased() != "i" {
                return .replayAgent(name: agentName)
            }
        }

        for matcher in Self.operatorCommandMatchers where lowered.contains(matcher.pattern) {
            return matcher.command
        }
        return nil
    }
}

// MARK: - Keyword Extraction

extension MessageRouter {
    /// Regex pattern for explicit agent targeting in user utterances.
    private static let keywordPattern: NSRegularExpression? = {
        try? NSRegularExpression(
            pattern: #"(?:tell |hey |@)(\w+)[,:]?\s*(.*)"#,
            options: [.caseInsensitive]
        )
    }()

    /// Extract an explicit agent name and message from the user's utterance.
    private func extractKeyword(
        from text: String,
        sessionNames: [String]
    ) -> (session: String, message: String)? {
        guard let pattern = Self.keywordPattern else {
            return nil
        }

        let range = NSRange(text.startIndex..., in: text)
        guard let match = pattern.firstMatch(in: text, range: range) else {
            return nil
        }

        guard let nameRange = Range(match.range(at: 1), in: text) else {
            return nil
        }

        let extractedName = String(text[nameRange]).lowercased()

        guard let canonical = sessionNames.first(where: { $0.lowercased() == extractedName }) else {
            return nil
        }

        if let messageRange = Range(match.range(at: 2), in: text) {
            let message = String(text[messageRange]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !message.isEmpty {
                return (session: canonical, message: message)
            }
        }

        return (session: canonical, message: text)
    }
}

// MARK: - claude -p Smart Routing

extension MessageRouter {
    /// Format a time interval as a human-readable "N min ago" or "N sec ago" string.
    private static func timeAgoString(from past: Date, to now: Date) -> String {
        let seconds = Int(now.timeIntervalSince(past))
        guard seconds < 60 else {
            let minutes = seconds / 60
            return "\(minutes) min ago"
        }
        return "\(seconds) sec ago"
    }

    /// Invoke claude -p for smart routing with session context and routing history.
    private func claudePipeRoute(
        text: String,
        sessions: [SessionState],
        routingState: RoutingState
    ) async -> RoutingResult {
        let prompt = buildRoutingPrompt(text: text, sessions: sessions, routingState: routingState)
        let sessionNames = sessions.map { $0.name }

        do {
            let json = try await ClaudePipe.run(prompt: prompt)

            let confident = json["confident"] as? Bool ?? false
            let sessionName = json["session"] as? String

            if confident, let session = sessionName,
                sessionNames.contains(where: { $0.lowercased() == session.lowercased() })
            {
                let canonical = sessionNames.first { $0.lowercased() == session.lowercased() } ?? session
                return .route(session: canonical, message: text)
            }

            let candidates: [String]
            if let rawCandidates = json["candidates"] as? [String] {
                candidates = rawCandidates.filter { name in
                    sessionNames.contains { $0.lowercased() == name.lowercased() }
                }
            } else {
                candidates = sessionNames
            }

            let question =
                json["question"] as? String
                ?? "Was that for \(sessionNames.joined(separator: " or "))?"

            return .clarify(candidates: candidates, question: question, originalText: text)
        } catch let error as ClaudePipeError {
            if case .cliNotFound = error {
                Self.logger.error("claude CLI not found; smart routing unavailable")
                return .cliNotFound
            }
            Self.logger.error("claude -p routing failed: \(error.description)")

            let question =
                "I couldn't determine routing. Which agent is this for? "
                + "\(sessionNames.joined(separator: " or "))?"

            return .clarify(candidates: sessionNames, question: question, originalText: text)
        } catch {
            Self.logger.error("Unexpected error in claude -p routing: \(error)")
            let question = "Which agent is this for? \(sessionNames.joined(separator: " or "))?"
            return .clarify(candidates: sessionNames, question: question, originalText: text)
        }
    }

    /// Build the prompt for claude -p smart routing.
    private func buildRoutingPrompt(
        text: String,
        sessions: [SessionState],
        routingState: RoutingState
    ) -> String {
        var lines: [String] = [
            "You are a message router. Active Claude Code sessions:"
        ]

        for (index, session) in sessions.enumerated() {
            var sessionLine = "\(index + 1). \"\(session.name)\" (\(session.cwd))"
            if !session.context.isEmpty {
                sessionLine += " - \(session.context)"
            }
            if let lastMsg = session.recentMessages.last {
                sessionLine += ", last msg: \"\(lastMsg.text.prefix(80))\""
            }
            lines.append(sessionLine)
        }

        if !routingState.routedMessages.isEmpty {
            lines.append("")
            lines.append("Recent routing history:")

            let now = Date()

            for entry in routingState.routedMessages.suffix(5) {
                let ago = Self.timeAgoString(from: entry.timestamp, to: now)
                lines.append("- \"\(entry.text.prefix(60))\" -> \(entry.session) (\(ago))")
            }
        }

        lines.append("")
        lines.append("User said: \"\(text)\"")
        lines.append("")
        let sessionNames = sessions.map { "\"\($0.name)\"" }.joined(separator: ", ")
        lines.append(
            "Reply ONLY as JSON: {\"session\": \"name\", \"confident\": true}"
        )
        lines.append(
            "Or if ambiguous: {\"session\": null, \"confident\": false, "
                + "\"candidates\": [\(sessionNames)], \"question\": \"Was that for X or Y?\"}"
        )

        return lines.joined(separator: "\n")
    }
}

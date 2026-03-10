import Foundation

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

    /// Heuristic routing was not confident enough to select a target.
    /// The original transcribed text is included for downstream dictation.
    case notConfident(String)
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
public final class MessageRouter: Sendable {
    static let logger = Log.logger(for: "MessageRouter")

    /// The session registry providing current session state for routing decisions.
    let registry: SessionRegistry

    /// The routing engine used for claude -p smart routing and clarification fallback.
    let engine: any RoutingEngine

    /// Creates a new message router with the given session registry and routing engine.
    ///
    /// - Parameters:
    ///   - registry: The session registry providing current session state.
    ///   - engine: The routing engine for claude -p invocations. Defaults to `ClaudePipeRoutingEngine`.
    public init(registry: SessionRegistry, engine: any RoutingEngine = ClaudePipeRoutingEngine()) {
        self.registry = registry
        self.engine = engine
    }
}

// MARK: - Deterministic Routing Chain

extension MessageRouter {
    /// Intermediate result from the deterministic priority chain shared by
    /// `route()` and `routeSkipEngine()`.
    enum DeterministicResult {
        /// A step in the chain produced a confident routing result.
        case resolved(RoutingResult)
        /// No deterministic step matched; callers decide what to do next.
        case unresolved(trimmed: String, sessions: [SessionState])
    }

    /// Run the deterministic routing steps: operator commands, keyword extraction,
    /// single-session bypass, session affinity, and heuristic scoring.
    ///
    /// Both `route()` and `routeSkipEngine()` delegate here, diverging only on
    /// the unresolved path (routing engine vs `.notConfident`).
    ///
    /// - Parameters:
    ///   - text: The transcribed user message.
    ///   - routingState: Current routing state with affinity and history data.
    ///   - prefetchedSessions: Pre-fetched sessions to avoid a redundant registry call.
    ///     When nil, sessions are fetched from the registry.
    /// - Returns: A resolved routing result or an unresolved state for callers to handle.
    func deterministicChain(
        text: String,
        routingState: RoutingState,
        prefetchedSessions: [SessionState]? = nil
    ) async -> DeterministicResult {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let lowered = trimmed.lowercased()

        if let command = matchOperatorCommand(lowered) {
            Self.logger.info("Matched operator command: \(String(describing: command))")
            return .resolved(.operatorCommand(command))
        }

        let sessions: [SessionState]
        if let prefetched = prefetchedSessions {
            sessions = prefetched
        } else {
            sessions = await registry.allSessions()
        }
        let sessionNames = sessions.map(\.name)

        if let match = AgentNameMatcher.match(in: trimmed, sessionNames: sessionNames) {
            Self.logger.info("Keyword match: routing to '\(match.session)'")
            return .resolved(.route(session: match.session, message: match.message))
        }

        if sessions.count == 1 {
            let session = sessions[0]
            Self.logger.info("Single session bypass: routing to '\(session.name)'")
            return .resolved(.route(session: session.name, message: trimmed))
        }

        if let affinityTarget = routingState.affinityTarget() {
            if sessionNames.contains(affinityTarget) {
                Self.logger.info("Session affinity: routing to '\(affinityTarget)'")
                return .resolved(.route(session: affinityTarget, message: trimmed))
            }
            Self.logger.debug(
                "Affinity target '\(affinityTarget)' no longer registered; falling through"
            )
        }

        if let heuristicTarget = heuristicRoute(text: trimmed, sessions: sessions) {
            Self.logger.info("Context heuristic: routing to '\(heuristicTarget)'")
            return .resolved(.route(session: heuristicTarget, message: trimmed))
        }

        return .unresolved(trimmed: trimmed, sessions: sessions)
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
        Self.logger.info("Routing message: \"\(text.prefix(80))\"")

        switch await deterministicChain(text: text, routingState: routingState) {
        case .resolved(let result):
            return result

        case .unresolved(let trimmed, let sessions):
            if sessions.isEmpty {
                Self.logger.warning("No sessions registered; cannot route")
                return .noSessions
            }
            Self.logger.info("Invoking routing engine for smart routing")
            return await claudePipeRoute(
                text: trimmed,
                sessions: sessions,
                routingState: routingState
            )
        }
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
    func matchOperatorCommand(_ lowered: String) -> OperatorCommand? {
        let range = NSRange(lowered.startIndex..., in: lowered)
        if let pattern = Self.whatDidSayPattern,
            let match = pattern.firstMatch(in: lowered, range: range),
            let nameRange = Range(match.range(at: 1), in: lowered)
        {
            let agentName = String(lowered[nameRange])
            if agentName != "i" {
                return .replayAgent(name: agentName)
            }
        }

        for matcher in Self.operatorCommandMatchers where lowered.contains(matcher.pattern) {
            return matcher.command
        }
        return nil
    }
}

// MARK: - Case-Insensitive Session Lookup

extension MessageRouter {
    /// Find the canonical session name matching `name` case-insensitively.
    static func canonicalSession(
        _ name: String,
        in sessionNames: [String]
    ) -> String? {
        let lowered = name.lowercased()
        return sessionNames.first { $0.lowercased() == lowered }
    }
}

// MARK: - claude -p Smart Routing

extension MessageRouter {
    /// Invoke claude -p for smart routing with session context and routing history.
    private func claudePipeRoute(
        text: String,
        sessions: [SessionState],
        routingState: RoutingState
    ) async -> RoutingResult {
        let prompt = RoutingPrompt.buildPrompt(
            text: text,
            sessions: sessions,
            routingState: routingState
        )
        let sessionNames = sessions.map(\.name)

        do {
            let json = try await engine.run(prompt: prompt)

            let confident = json["confident"] as? Bool ?? false
            let sessionName = json["session"] as? String

            if confident, let session = sessionName,
                let canonical = Self.canonicalSession(session, in: sessionNames)
            {
                return .route(session: canonical, message: text)
            }

            let candidates: [String]
            if let rawCandidates = json["candidates"] as? [String] {
                candidates = rawCandidates.filter { name in
                    Self.canonicalSession(name, in: sessionNames) != nil
                }
            } else {
                candidates = sessionNames
            }

            let question =
                json["question"] as? String
                ?? "Was that for \(sessionNames.joined(separator: " or "))?"

            return .clarify(candidates: candidates, question: question, originalText: text)
        } catch {
            return handleRoutingError(error, sessionNames: sessionNames, text: text)
        }
    }

    private func handleRoutingError(
        _ error: any Error,
        sessionNames: [String],
        text: String
    ) -> RoutingResult {
        if let pipeError = error as? ClaudePipeError {
            if case .cliNotFound = pipeError {
                Self.logger.error("claude CLI not found; smart routing unavailable")
                return .cliNotFound
            }
            Self.logger.error("claude -p routing failed: \(pipeError.description)")
        } else {
            Self.logger.error("Unexpected error in claude -p routing: \(error)")
        }

        let question = "Which agent is this for? \(sessionNames.joined(separator: " or "))?"
        return .clarify(candidates: sessionNames, question: question, originalText: text)
    }
}

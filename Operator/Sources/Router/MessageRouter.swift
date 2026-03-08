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

// MARK: - Primary Routing

extension MessageRouter {
    private struct HeuristicCandidate {
        let session: String
        let score: Int
        let directHits: Int
    }

    private static let heuristicStopWords: Set<String> = [
        "a", "an", "and", "the", "to", "for", "of", "on", "in", "at", "with", "from", "new",
        "fix", "add", "update", "make", "check", "look", "into", "that", "this", "it", "my",
        "your", "their", "our", "please", "need", "still", "just", "agent", "session"
    ]

    private static let heuristicConceptGroups: [[String]] = [
        ["react", "tsx", "typescript", "frontend", "client", "ui", "css", "layout", "dashboard", "shell", "button", "spacing", "skeleton", "spinner", "loading", "component"],
        ["api", "backend", "server", "endpoint", "rest", "route", "service", "python", "py", "sql", "postgres", "database", "db", "query", "middleware", "auth", "billing", "refund", "profile", "sync", "requirements"],
        ["infra", "terraform", "tf", "deploy", "deployment", "staging", "prod", "production", "cluster", "eks", "vpc", "subnet", "network", "iam", "policy", "s3", "logging", "rollout"]
    ]

    private static let heuristicConceptMap: [String: Set<String>] = {
        var map: [String: Set<String>] = [:]
        for group in heuristicConceptGroups {
            let expanded = Set(group)
            for token in group {
                map[token] = expanded
            }
        }
        return map
    }()

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

        if let heuristicTarget = heuristicRoute(text: trimmed, sessions: sessions) {
            Self.logger.info("Context heuristic: routing to '\(heuristicTarget)'")
            return .route(session: heuristicTarget, message: trimmed)
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
    /// Extract an explicit agent name and message from the user's utterance.
    private func extractKeyword(
        from text: String,
        sessionNames: [String]
    ) -> (session: String, message: String)? {
        guard let result = AgentNameMatcher.match(in: text, sessionNames: sessionNames) else {
            return nil
        }
        return (session: result.session, message: result.message)
    }
}

// MARK: - claude -p Smart Routing

extension MessageRouter {
    private func heuristicRoute(
        text: String,
        sessions: [SessionState]
    ) -> String? {
        let messageTokens = Self.heuristicMessageTokens(from: text)
        guard !messageTokens.isEmpty else {
            return nil
        }

        let ranked = sessions
            .map { session in
                Self.scoreHeuristicRoute(messageTokens: messageTokens, session: session)
            }
            .filter { $0.score > 0 }
            .sorted {
                if $0.score == $1.score {
                    return $0.directHits > $1.directHits
                }
                return $0.score > $1.score
            }

        guard let best = ranked.first else {
            return nil
        }

        let runnerUpScore = ranked.dropFirst().first?.score ?? 0
        let strongDirectMatch = best.directHits >= 2 && best.score >= 10
        let strongMargin = best.score >= 12 && best.score >= runnerUpScore + 4
        guard strongDirectMatch || strongMargin else {
            return nil
        }

        return best.session
    }

    private static func scoreHeuristicRoute(
        messageTokens: Set<String>,
        session: SessionState
    ) -> HeuristicCandidate {
        let nameTokens = heuristicExpandedTokens(
            from: session.name,
            dropStopWords: false
        )
        let cwdTokens = heuristicExpandedTokens(from: session.cwd)
        let contextTokens = heuristicExpandedTokens(from: session.context)
        let recentTokens = heuristicExpandedTokens(
            from: session.recentMessages.map(\.text).joined(separator: " ")
        )

        let tokenWeights = [
            (nameTokens, 8),
            (cwdTokens, 6),
            (contextTokens, 4),
            (recentTokens, 3)
        ]

        var score = 0
        var directHits = 0

        for token in messageTokens {
            var matched = false
            for (sourceTokens, weight) in tokenWeights where sourceTokens.contains(token) {
                score += weight
                matched = true
            }

            if matched {
                directHits += 1
            }
        }

        return HeuristicCandidate(session: session.name, score: score, directHits: directHits)
    }

    private static func heuristicMessageTokens(from text: String) -> Set<String> {
        heuristicExpandedTokens(from: text)
    }

    private static func heuristicExpandedTokens(
        from text: String,
        dropStopWords: Bool = true
    ) -> Set<String> {
        let baseTokens = heuristicBaseTokens(from: text)
        var expanded: Set<String> = []

        for token in baseTokens {
            if dropStopWords && heuristicStopWords.contains(token) {
                continue
            }
            expanded.insert(token)
            if let conceptTokens = heuristicConceptMap[token] {
                expanded.formUnion(conceptTokens)
            }
        }

        return expanded
    }

    private static func heuristicBaseTokens(from text: String) -> Set<String> {
        let separators = CharacterSet.alphanumerics.inverted
        let lowered = text
            .replacingOccurrences(
                of: "([a-z0-9])([A-Z])",
                with: "$1 $2",
                options: .regularExpression
            )
            .lowercased()

        let rawTokens = lowered
            .components(separatedBy: separators)
            .filter { !$0.isEmpty }

        var tokens = Set(rawTokens)
        for token in rawTokens where token.count > 3 && token.hasSuffix("s") {
            tokens.insert(String(token.dropLast()))
        }
        return tokens
    }

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
        let sessionNames = sessions.map { $0.name }

        do {
            let json = try await engine.run(prompt: prompt)

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
}

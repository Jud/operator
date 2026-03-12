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

    /// Optional routing engine for smart routing fallback.
    ///
    /// When nil, ambiguous messages go directly to clarification.
    /// When set, invoked after heuristics fail to attempt classification.
    let engine: (any RoutingEngine)?

    /// Optional trace store for persisting routing decisions.
    ///
    /// When set, every routing decision is captured for offline analysis.
    let traceStore: RoutingTraceStore?

    /// Creates a new message router with the given session registry.
    ///
    /// - Parameters:
    ///   - registry: The session registry providing current session state.
    ///   - engine: Optional routing engine for smart routing. Defaults to nil (clarification only).
    ///   - traceStore: Optional store for persisting routing traces. Defaults to nil.
    public init(
        registry: SessionRegistry,
        engine: (any RoutingEngine)? = nil,
        traceStore: RoutingTraceStore? = nil
    ) {
        self.registry = registry
        self.engine = engine
        self.traceStore = traceStore
    }
}

// MARK: - Deterministic Routing Chain

extension MessageRouter {
    /// Intermediate result from the deterministic priority chain shared by
    /// `route()` and `routeSkipEngine()`.
    enum DeterministicResult {
        /// A step in the chain produced a confident routing result.
        case resolved(RoutingResult, step: RoutingTrace.RoutingStep, sessions: [SessionState])
        /// No deterministic step matched; callers decide what to do next.
        case unresolved(trimmed: String, sessions: [SessionState])
    }

    /// Run the deterministic routing steps: operator commands, keyword extraction,
    /// single-session bypass, session affinity, and heuristic scoring.
    func deterministicChain(
        text: String,
        routingState: RoutingState,
        prefetchedSessions: [SessionState]? = nil
    ) async -> DeterministicResult {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let lowered = trimmed.lowercased()

        if let command = matchOperatorCommand(lowered) {
            Self.logger.info("Matched operator command: \(String(describing: command))")
            return .resolved(.operatorCommand(command), step: .operatorCommand, sessions: [])
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
            return .resolved(
                .route(session: match.session, message: match.message),
                step: .keywordExtraction,
                sessions: sessions
            )
        }

        if sessions.count == 1 {
            let session = sessions[0]
            Self.logger.info("Single session bypass: routing to '\(session.name)'")
            return .resolved(
                .route(session: session.name, message: trimmed),
                step: .singleSessionBypass,
                sessions: sessions
            )
        }

        if let result = checkAffinityAndHeuristic(
            trimmed: trimmed,
            sessions: sessions,
            sessionNames: sessionNames,
            routingState: routingState
        ) {
            return result
        }

        return .unresolved(trimmed: trimmed, sessions: sessions)
    }

    /// Check session affinity and heuristic scoring steps.
    private func checkAffinityAndHeuristic(
        trimmed: String,
        sessions: [SessionState],
        sessionNames: [String],
        routingState: RoutingState
    ) -> DeterministicResult? {
        if let affinityTarget = routingState.affinityTarget() {
            if sessionNames.contains(affinityTarget) {
                Self.logger.info("Session affinity: routing to '\(affinityTarget)'")
                return .resolved(
                    .route(session: affinityTarget, message: trimmed),
                    step: .sessionAffinity,
                    sessions: sessions
                )
            }
            Self.logger.debug(
                "Affinity target '\(affinityTarget)' no longer registered; falling through"
            )
        }

        if let heuristicTarget = heuristicRoute(text: trimmed, sessions: sessions) {
            Self.logger.info("Context heuristic: routing to '\(heuristicTarget)'")
            return .resolved(
                .route(session: heuristicTarget, message: trimmed),
                step: .heuristicScoring,
                sessions: sessions
            )
        }

        return nil
    }
}

// MARK: - Primary Routing

extension MessageRouter {
    /// Route a transcribed user message through the full priority chain.
    public func route(
        text: String,
        routingState: RoutingState,
        audioFile: String? = nil
    ) async -> RoutingResult {
        Self.logger.info("Routing message: \"\(text.prefix(80))\"")

        let result: RoutingResult
        let step: RoutingTrace.RoutingStep
        let sessions: [SessionState]

        switch await deterministicChain(text: text, routingState: routingState) {
        case .resolved(let routeResult, let resolvedStep, let resolvedSessions):
            result = routeResult
            step = resolvedStep
            sessions = resolvedSessions

        case .unresolved(let trimmed, let resolvedSessions):
            sessions = resolvedSessions
            if resolvedSessions.isEmpty {
                Self.logger.warning("No sessions registered; cannot route")
                result = .noSessions
                step = .noSessions
            } else if engine != nil {
                Self.logger.info("Invoking routing engine for smart routing")
                result = await claudePipeRoute(
                    text: trimmed,
                    sessions: resolvedSessions,
                    routingState: routingState
                )
                step = .engineRouting
            } else {
                let sessionNames = resolvedSessions.map(\.name)
                Self.logger.info("Heuristics not confident; asking for clarification")
                result = Self.clarifyResult(candidates: sessionNames, originalText: trimmed)
                step = .clarification
            }
        }

        await postTrace(
            text: text,
            sessions: sessions,
            step: step,
            result: result,
            routingState: routingState,
            audioFile: audioFile
        )
        return result
    }
}

// MARK: - claude -p Smart Routing

extension MessageRouter {
    /// Invoke claude -p for smart routing with session context and routing history.
    func claudePipeRoute(
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

        guard let engine else {
            return Self.clarifyResult(candidates: sessionNames, originalText: text)
        }

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
                candidates = rawCandidates.compactMap { name in
                    Self.canonicalSession(name, in: sessionNames)
                }
            } else {
                candidates = sessionNames
            }

            let question =
                json["question"] as? String
                ?? "Was that for \(sessionNames.joined(separator: " or "))?"

            return .clarify(candidates: candidates, question: question, originalText: text)
        } catch {
            if let pipeError = error as? ClaudePipeError, case .cliNotFound = pipeError {
                Self.logger.error("claude CLI not found; smart routing unavailable")
                return .cliNotFound
            }
            Self.logger.error("claude -p routing failed: \(error)")
            return Self.clarifyResult(candidates: sessionNames, originalText: text)
        }
    }
}

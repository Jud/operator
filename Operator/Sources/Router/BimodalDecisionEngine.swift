import Foundation

/// Decision from the bimodal engine.
///
/// Determines whether voice input should be inserted at the cursor (dictation,
/// the default) or routed to a Claude Code agent session (the exception).
public enum BimodalDecision: Sendable {
    /// Insert text at the cursor via dictation.
    case dictate(String)
    /// Route to an agent via the existing pipeline.
    case routeToAgent(String)
    /// No text field detected in frontmost app.
    case noTextField
    /// Accessibility permission not granted.
    case permissionRequired
}

/// Evaluates context signals to decide between dictation and agent routing.
///
/// Runs a 4-step priority chain after each transcription:
/// 1. Active affinity, pending interruption, or pending clarification -> agent routing
/// 2. Keyword extraction match ("tell X...", "hey X...", "@X") -> agent routing
/// 3. Terminal frontmost -> confidence-gated routing (deterministic signals only)
/// 4. Non-terminal frontmost -> dictation (if text field) or error (if not)
///
/// For step 3, only deterministic routing signals (operator commands, single-session
/// bypass) are trusted. Multi-session heuristic scoring results are discarded because
/// concept expansion inflates scores from a single domain-adjacent word to 300-450
/// points, producing an 87.5% false positive rate (HYP-001, REJECTED). Users wanting
/// agent routing in a multi-session terminal context use keyword addressing.
///
/// Reference: design.md Section 3.3; requirements.md FR-001
public struct BimodalDecisionEngine: Sendable {
    private static let logger = Log.logger(for: "BimodalDecisionEngine")

    let accessibilityQuery: any AccessibilityQuerying
    let router: MessageRouter
    let registry: SessionRegistry

    /// Creates a bimodal decision engine with the given dependencies.
    ///
    /// - Parameters:
    ///   - accessibilityQuery: Service for frontmost app and text field detection.
    ///   - router: Message router for confidence-gated routing in terminal context.
    ///   - registry: Session registry for querying registered agent sessions.
    public init(
        accessibilityQuery: any AccessibilityQuerying,
        router: MessageRouter,
        registry: SessionRegistry
    ) {
        self.accessibilityQuery = accessibilityQuery
        self.router = router
        self.registry = registry
    }

    /// Decide whether to dictate at the cursor or route to an agent.
    ///
    /// - Parameters:
    ///   - text: The transcribed user message.
    ///   - routingState: Current routing state with affinity and history data.
    /// - Returns: A `BimodalDecision` indicating the appropriate action.
    public func decide(text: String, routingState: RoutingState) async -> BimodalDecision {
        // Step 1: Active conversation takes priority regardless of frontmost app.
        if routingState.pendingInterruption != nil {
            Self.logger.info("Bimodal: pending interruption -> routeToAgent")
            return .routeToAgent(text)
        }
        if routingState.pendingClarification != nil {
            Self.logger.info("Bimodal: pending clarification -> routeToAgent")
            return .routeToAgent(text)
        }
        if routingState.isAffinityActive() {
            Self.logger.info("Bimodal: affinity active -> routeToAgent")
            return .routeToAgent(text)
        }

        // Step 2: Keyword extraction ("tell X", "hey X", "@X") routes to agent
        // regardless of which app is frontmost.
        let sessions = await registry.allSessions()
        let sessionNames = sessions.map(\.name)
        if !sessionNames.isEmpty,
            AgentNameMatcher.match(in: text, sessionNames: sessionNames) != nil
        {
            Self.logger.info("Bimodal: keyword match -> routeToAgent")
            return .routeToAgent(text)
        }

        // Query frontmost app and focused element context.
        let context = accessibilityQuery.queryContext()

        if !context.hasPermission {
            Self.logger.warning("Bimodal: accessibility permission not granted")
            return .permissionRequired
        }

        // Step 3: Terminal frontmost -- confidence-gated routing using only
        // deterministic signals from routeSkipEngine.
        if context.isTerminal {
            return await decideForTerminal(
                text: text,
                routingState: routingState,
                sessionCount: sessions.count,
                isTextField: context.isTextField
            )
        }

        // Step 4: Non-terminal frontmost -- dictation if text field available.
        if context.isTextField {
            Self.logger.info("Bimodal: non-terminal + text field -> dictate")
            return .dictate(text)
        }
        Self.logger.info("Bimodal: non-terminal + no text field -> noTextField")
        return .noTextField
    }

    /// Evaluate routing confidence when a terminal is the frontmost app.
    ///
    /// Calls `routeSkipEngine` to leverage the existing priority chain but only
    /// trusts deterministic results: operator commands and single-session bypass.
    /// Multi-session `.route` results are discarded because they originate from
    /// heuristic scoring, which concept expansion renders unreliable for bimodal
    /// gating (HYP-001: 87.5% false positive rate on everyday English).
    private func decideForTerminal(
        text: String,
        routingState: RoutingState,
        sessionCount: Int,
        isTextField: Bool
    ) async -> BimodalDecision {
        let result = await router.routeSkipEngine(text: text, routingState: routingState)

        switch result {
        case .operatorCommand:
            Self.logger.info("Bimodal: terminal + operator command -> routeToAgent")
            return .routeToAgent(text)

        case .route where sessionCount == 1:
            Self.logger.info("Bimodal: terminal + single session -> routeToAgent")
            return .routeToAgent(text)

        case .route:
            // Multi-session .route from routeSkipEngine originates from heuristic
            // scoring after keyword, single-session, and affinity checks (all already
            // handled by steps 1-2 above). Concept expansion inflates a single
            // domain-adjacent word ("server", "dashboard", "profile") into scores of
            // 300-450, trivially passing thresholds designed for same-modality routing.
            // Discard the heuristic result and fall through to dictation.
            Self.logger.info("Bimodal: terminal + heuristic (untrusted per HYP-001) -> dictate")

        case .notConfident:
            Self.logger.info("Bimodal: terminal + not confident -> dictate")

        default:
            Self.logger.info("Bimodal: terminal + unhandled result -> dictate")
        }

        if isTextField {
            return .dictate(text)
        }
        return .noTextField
    }
}

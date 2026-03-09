import Foundation

// MARK: - Confidence-Gated Routing (Skip Engine)

extension MessageRouter {
    /// Route using only heuristic signals -- no routing engine, no clarification.
    ///
    /// Runs the same priority chain as `route(text:routingState:)` but stops
    /// before invoking `claude -p`. When no step produces a confident result,
    /// returns `.notConfident(text)` instead of proceeding to clarification.
    ///
    /// Used by the bimodal decision engine for terminal-frontmost confidence
    /// gating where routing engine latency is unacceptable.
    ///
    /// - Parameters:
    ///   - text: The transcribed user message.
    ///   - routingState: Current routing state with affinity and history data.
    /// - Returns: A `RoutingResult` -- either a confident route or `.notConfident`.
    public func routeSkipEngine(text: String, routingState: RoutingState) async -> RoutingResult {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let lowered = trimmed.lowercased()

        Self.logger.info("Skip-engine routing: \"\(trimmed.prefix(80))\"")

        if let command = matchOperatorCommand(lowered) {
            Self.logger.info("Skip-engine: matched operator command: \(String(describing: command))")
            return .operatorCommand(command)
        }

        let sessions = await registry.allSessions()

        if sessions.isEmpty {
            Self.logger.info("Skip-engine: no sessions; returning notConfident")
            return .notConfident(trimmed)
        }

        let sessionNames = sessions.map { $0.name }

        if let (target, message) = extractKeyword(from: trimmed, sessionNames: sessionNames) {
            Self.logger.info("Skip-engine: keyword match to '\(target)'")
            return .route(session: target, message: message)
        }

        if sessions.count == 1 {
            let session = sessions[0]
            Self.logger.info("Skip-engine: single session bypass to '\(session.name)'")
            return .route(session: session.name, message: trimmed)
        }

        if let affinityTarget = routingState.affinityTarget() {
            if sessionNames.contains(where: { $0 == affinityTarget }) {
                Self.logger.info("Skip-engine: affinity to '\(affinityTarget)'")
                return .route(session: affinityTarget, message: trimmed)
            }
            Self.logger.debug("Skip-engine: affinity target '\(affinityTarget)' no longer registered")
        }

        if let heuristicTarget = heuristicRoute(text: trimmed, sessions: sessions) {
            Self.logger.info("Skip-engine: heuristic match to '\(heuristicTarget)'")
            return .route(session: heuristicTarget, message: trimmed)
        }

        Self.logger.info("Skip-engine: no confident match; returning notConfident")
        return .notConfident(trimmed)
    }
}

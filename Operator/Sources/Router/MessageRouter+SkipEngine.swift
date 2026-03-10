import Foundation

// MARK: - Confidence-Gated Routing (Skip Engine)

extension MessageRouter {
    /// Route using only deterministic signals — no routing engine, no clarification.
    ///
    /// Runs the same deterministic priority chain as `route(text:routingState:)`
    /// but returns `.notConfident(text)` instead of invoking the routing engine.
    ///
    /// Used by the bimodal decision engine for terminal-frontmost confidence
    /// gating where routing engine latency is unacceptable.
    ///
    /// - Parameters:
    ///   - text: The transcribed user message.
    ///   - routingState: Current routing state with affinity and history data.
    /// - Returns: A `RoutingResult` — either a confident route or `.notConfident`.
    public func routeSkipEngine(text: String, routingState: RoutingState) async -> RoutingResult {
        Self.logger.info("Skip-engine routing: \"\(text.prefix(80))\"")

        switch await deterministicChain(text: text, routingState: routingState) {
        case .resolved(let result):
            return result

        case .unresolved(let trimmed, let sessions):
            if sessions.isEmpty {
                Self.logger.info("Skip-engine: no sessions; returning noSessions")
                return .noSessions
            }
            Self.logger.info("Skip-engine: no confident match; returning notConfident")
            return .notConfident(trimmed)
        }
    }
}

import Foundation

// MARK: - Trace Construction

extension RoutingResult {
    /// Convert a routing result to the trace-serializable equivalent.
    var traceResult: RoutingTrace.RoutingTraceResult {
        switch self {
        case .route(let session, let message):
            .routed(session: session, message: message)

        case .clarify(let candidates, _, _):
            .clarify(candidates: candidates)

        case .operatorCommand(let cmd):
            .operatorCommand(String(describing: cmd))

        case .noSessions:
            .noSessions

        case .notConfident:
            .notConfident

        case .cliNotFound:
            .cliNotFound
        }
    }
}

extension MessageRouter {
    /// Build a routing trace capturing the full decision context.
    func buildTrace(
        text: String,
        sessions: [SessionState],
        step: RoutingTrace.RoutingStep,
        result: RoutingResult,
        routingState: RoutingState
    ) -> RoutingTrace {
        let snapshots = sessions.map { session in
            RoutingTrace.SessionSnapshot(
                name: session.name,
                cwd: session.cwd,
                context: session.context,
                recentMessages: session.recentMessages.map {
                    .init(role: $0.role, text: $0.text)
                }
            )
        }

        let messageTokens = Self.heuristicExpandedTokens(from: text)
        let scores: [RoutingTrace.HeuristicScore] = sessions.map { session in
            let candidate = Self.scoreHeuristicRoute(
                messageTokens: messageTokens,
                session: session
            )
            return RoutingTrace.HeuristicScore(
                session: candidate.session,
                score: candidate.score,
                directHits: candidate.directHits
            )
        }

        return RoutingTrace(
            transcribedText: text,
            sessions: snapshots,
            resolvedBy: step,
            heuristicScores: scores,
            result: result.traceResult,
            affinityActive: routingState.isAffinityActive(),
            affinityTarget: routingState.affinityTarget()
        )
    }

    /// Post a routing trace to the trace store and notify observers.
    func postTrace(
        text: String,
        sessions: [SessionState],
        step: RoutingTrace.RoutingStep,
        result: RoutingResult,
        routingState: RoutingState
    ) async {
        guard let traceStore else {
            return
        }
        let trace = buildTrace(
            text: text,
            sessions: sessions,
            step: step,
            result: result,
            routingState: routingState
        )
        await traceStore.append(trace)
        let names = sessions.map(\.name)
        await MainActor.run {
            NotificationCenter.default.post(
                name: .routingTraceAppended,
                object: nil,
                userInfo: ["trace": trace, "sessionNames": names]
            )
        }
    }
}

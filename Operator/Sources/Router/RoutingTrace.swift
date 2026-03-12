import Foundation

extension Notification.Name {
    /// Posted on the main thread after a routing trace is appended to the store.
    /// `userInfo` contains "trace" (RoutingTrace) and "sessionNames" ([String]).
    public static let routingTraceAppended = Notification.Name("RoutingTraceAppended")
}

/// A complete snapshot of a single routing decision for replay testing and feedback.
///
/// Captures the full context at the moment of routing: the user's transcribed text,
/// session state snapshots, which step in the priority chain resolved, heuristic
/// scores for each candidate, and the final result. Persisted as JSONL for offline
/// analysis and test suite generation.
public struct RoutingTrace: Codable, Sendable {
    /// Lightweight session snapshot for trace context.
    public struct SessionSnapshot: Codable, Sendable {
        /// A single message in the session's recent history.
        public struct Message: Codable, Sendable {
            /// The role of the message sender.
            public let role: String
            /// The text content of the message.
            public let text: String
        }

        /// The session's human-readable name.
        public let name: String
        /// The session's working directory.
        public let cwd: String
        /// The session's context summary.
        public let context: String
        /// Recent messages for routing context.
        public let recentMessages: [Message]
    }

    /// Which step in the routing priority chain produced the final result.
    public enum RoutingStep: String, Codable, Sendable {
        case operatorCommand = "operator_command"
        case keywordExtraction = "keyword_extraction"
        case singleSessionBypass = "single_session_bypass"
        case sessionAffinity = "session_affinity"
        case heuristicScoring = "heuristic_scoring"
        case engineRouting = "engine_routing"
        case clarification
        case noSessions = "no_sessions"
    }

    /// Heuristic score for a single session candidate.
    public struct HeuristicScore: Codable, Sendable {
        /// The session name this score applies to.
        public let session: String
        /// The weighted heuristic score.
        public let score: Int
        /// The number of direct token matches.
        public let directHits: Int
    }

    /// The routing outcome, serializable for replay.
    public enum RoutingTraceResult: Codable, Sendable {
        case routed(session: String, message: String)
        case clarify(candidates: [String])
        case operatorCommand(String)
        case noSessions
        case notConfident
        case cliNotFound
    }

    /// User-provided feedback on routing quality.
    ///
    /// Set via the debug HUD's thumbs-up/down buttons.
    public struct Annotation: Codable, Sendable {
        /// Whether the routing was correct.
        public let correct: Bool
        /// If incorrect, which session should have been chosen.
        public let correctedSession: String?
        /// When the annotation was made.
        public let annotatedAt: Date

        /// Creates a new annotation.
        public init(correct: Bool, correctedSession: String? = nil, annotatedAt: Date = Date()) {
            self.correct = correct
            self.correctedSession = correctedSession
            self.annotatedAt = annotatedAt
        }
    }

    /// Unique identifier for this trace.
    public let id: UUID
    /// When the routing decision was made.
    public let timestamp: Date
    /// The user's transcribed text that was routed.
    public let transcribedText: String
    /// Snapshot of each registered session at routing time.
    public let sessions: [SessionSnapshot]
    /// Which step in the priority chain produced the result.
    public let resolvedBy: RoutingStep
    /// Heuristic scores for each session (empty if resolved before heuristics).
    public let heuristicScores: [HeuristicScore]
    /// The routing result.
    public let result: RoutingTraceResult
    /// Whether affinity was active at routing time.
    public let affinityActive: Bool
    /// The affinity target session, if any.
    public let affinityTarget: String?
    /// User annotation: was the routing correct?
    public var annotation: Annotation?

    /// Creates a new routing trace with an auto-generated ID and timestamp.
    public init(
        transcribedText: String,
        sessions: [SessionSnapshot],
        resolvedBy: RoutingStep,
        heuristicScores: [HeuristicScore],
        result: RoutingTraceResult,
        affinityActive: Bool,
        affinityTarget: String?,
        annotation: Annotation? = nil
    ) {
        self.id = UUID()
        self.timestamp = Date()
        self.transcribedText = transcribedText
        self.sessions = sessions
        self.resolvedBy = resolvedBy
        self.heuristicScores = heuristicScores
        self.result = result
        self.affinityActive = affinityActive
        self.affinityTarget = affinityTarget
        self.annotation = annotation
    }
}

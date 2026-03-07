import AVFoundation
import Foundation

/// Status of a registered Claude Code session.
///
/// Tracks the session's current operational state for routing context
/// and status reporting via the HTTP /state endpoint.
public enum SessionStatus: String, Codable, Sendable {
    case idle
    case thinking
    case toolCalling = "toolCalling"  // swiftlint:disable:this redundant_string_enum_value
}

/// A single message exchanged in a session, used for routing context.
///
/// Stored as part of SessionState.recentMessages to provide claude -p
/// with recent conversation history for routing decisions. Uses a struct
/// instead of a tuple so it can be Codable for the HTTP /state response.
public struct SessionMessage: Codable, Sendable {
    /// The role of the message sender (e.g., "user" or "assistant").
    public let role: String
    /// The text content of the message.
    public let text: String

    /// Creates a new session message.
    public init(role: String, text: String) {
        self.role = role
        self.text = text
    }
}

/// Full state of a registered Claude Code session.
///
/// Created at registration time with voice assignment from VoiceManager.
/// Updated via updateContext() when the MCP server pushes context changes.
/// Checked during discovery polling for stale removal.
///
/// Reference: technical-spec.md, Session State Model
public struct SessionState: Sendable {
    /// The human-readable session name.
    public let name: String
    /// The TTY device path identifying this session.
    public let tty: String
    /// The working directory of the session.
    public let cwd: String
    /// A summary of what the session is working on.
    public var context: String
    /// Recent conversation messages for routing context.
    public var recentMessages: [SessionMessage]
    /// The current operational status of the session.
    public var status: SessionStatus
    /// When the session last had activity.
    public var lastActivity: Date
    /// The assigned speech synthesis voice.
    public let voice: AVSpeechSynthesisVoice
    /// The pitch multiplier for auditory differentiation.
    public let pitchMultiplier: Float
    /// The Claude Code session identifier, set via hook integration.
    public var sessionId: String?

    /// Creates a new session state.
    public init(
        name: String,
        tty: String,
        cwd: String,
        context: String,
        recentMessages: [SessionMessage],
        status: SessionStatus,
        lastActivity: Date,
        voice: AVSpeechSynthesisVoice,
        pitchMultiplier: Float,
        sessionId: String? = nil
    ) {
        self.name = name
        self.tty = tty
        self.cwd = cwd
        self.context = context
        self.recentMessages = recentMessages
        self.status = status
        self.lastActivity = lastActivity
        self.voice = voice
        self.pitchMultiplier = pitchMultiplier
        self.sessionId = sessionId
    }
}

/// Codable snapshot of a session for the HTTP /state endpoint.
///
/// Excludes AVSpeechSynthesisVoice (not Codable) and includes only
/// the fields needed by external consumers querying GET /state.
public struct SessionSnapshot: Codable, Sendable {
    /// The human-readable session name.
    public let name: String
    /// The TTY device path identifying this session.
    public let tty: String
    /// The working directory of the session.
    public let cwd: String
    /// A summary of what the session is working on.
    public let context: String
    /// Recent conversation messages for routing context.
    public let recentMessages: [SessionMessage]
    /// The current operational status of the session.
    public let status: SessionStatus
    /// When the session last had activity.
    public let lastActivity: Date
    /// The pitch multiplier for auditory differentiation.
    public let pitchMultiplier: Float
    /// The Claude Code session identifier, if set via hook integration.
    public let sessionId: String?
}

/// Full registry state returned by GET /state endpoint.
///
/// Contains all registered sessions as snapshots.
public struct RegistrySnapshot: Codable, Sendable {
    /// All registered sessions as snapshots.
    public let sessions: [SessionSnapshot]
    /// The number of registered sessions.
    public let sessionCount: Int

    /// Serialize to JSON string.
    public func toJSON() -> String {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        guard let data = try? encoder.encode(self) else {
            return "{}"
        }
        return String(data: data, encoding: .utf8) ?? "{}"
    }
}

/// Thread-safe registry of active Claude Code sessions.
///
/// Manages session lifecycle: registration, context updates, voice assignment,
/// and state queries for the HTTP API. Terminal-agnostic -- discovery polling
/// is handled by `SessionDiscoveryService`.
///
/// Concurrency: Swift actor ensures all state mutations are serialized without
/// manual locks, per technical-spec.md Key Design Decision #10.
///
/// Reference: technical-spec.md Component 5; design.md Section 3.1.10
public actor SessionRegistry {
    // MARK: - Type Properties

    private static let logger = Log.logger(for: "SessionRegistry")

    // MARK: - Instance Properties

    /// Registered sessions keyed by TTY path (globally unique, stable identifier).
    private var sessions: [String: SessionState] = [:]

    /// Voice manager for assigning distinct voices to each agent.
    private let voiceManager: VoiceManager

    /// Get the count of registered sessions.
    public var sessionCount: Int {
        sessions.count
    }

    // MARK: - Initialization

    /// - Parameter voiceManager: Manager for assigning per-agent voices with pitch variation.
    public init(voiceManager: VoiceManager) {
        self.voiceManager = voiceManager
        Self.logger.info("SessionRegistry initialized")
    }

    // MARK: - Registration

    /// Register a new Claude Code session with the given metadata.
    ///
    /// Assigns a distinct voice (from VoiceManager) with a unique pitch multiplier
    /// so the user can auditorily distinguish agents. Rejects the reserved name
    /// "operator" (case-insensitive) per technical-spec.md Component 4.
    ///
    /// - Parameters:
    ///   - name: Human-readable session name (e.g., "sudo", "frontend").
    ///   - tty: TTY device path (e.g., "/dev/ttys004"). Globally unique identifier.
    ///   - cwd: Working directory of the session.
    ///   - context: Initial context summary describing what the session is working on.
    /// - Returns: `true` if registration succeeded, `false` if the name is reserved.
    @discardableResult
    public func register(name: String, tty: String, cwd: String, context: String?) -> Bool {
        if name.lowercased() == "operator" {
            Self.logger.warning("Rejected registration: name 'operator' is reserved")
            return false
        }

        let pitch = voiceManager.nextAgentPitchMultiplier()
        let voice = voiceManager.defaultAgentVoice

        let state = SessionState(
            name: name,
            tty: tty,
            cwd: cwd,
            context: context ?? "",
            recentMessages: [],
            status: .idle,
            lastActivity: Date(),
            voice: voice,
            pitchMultiplier: pitch
        )

        let isReregistration = sessions[tty] != nil
        sessions[tty] = state

        if isReregistration {
            Self.logger.info("Re-registered session '\(name)' at TTY: \(tty)")
        } else {
            Self.logger.info("Registered session '\(name)' at TTY: \(tty) (pitch: \(pitch))")
        }

        return true
    }

    /// Remove a session by TTY path.
    ///
    /// - Parameter tty: The TTY path of the session to remove.
    /// - Returns: The name of the removed session, or nil if not found.
    @discardableResult
    public func deregister(tty: String) -> String? {
        guard let state = sessions.removeValue(forKey: tty) else {
            Self.logger.debug("Deregister called for unknown TTY: \(tty)")
            return nil
        }
        Self.logger.info("Deregistered session '\(state.name)' from TTY: \(tty)")
        return state.name
    }

    // MARK: - Context Updates

    /// Update the routing context for a session identified by TTY.
    ///
    /// - Parameters:
    ///   - tty: TTY path identifying the session to update.
    ///   - summary: Updated context summary.
    ///   - recentMessages: Recent conversation messages for routing context.
    public func updateContext(tty: String, summary: String, recentMessages: [SessionMessage]) {
        guard sessions[tty] != nil else {
            Self.logger.warning("updateContext called for unregistered TTY: \(tty)")
            return
        }
        sessions[tty]?.context = summary
        sessions[tty]?.recentMessages = recentMessages
        sessions[tty]?.lastActivity = Date()
        Self.logger.debug("Updated context for TTY: \(tty)")
    }

    // MARK: - Voice Lookup

    /// Look up the assigned voice for a session by name.
    ///
    /// - Parameter session: The session name to look up.
    /// - Returns: The AVSpeechSynthesisVoice assigned to the session.
    public func voiceFor(session: String?) -> AVSpeechSynthesisVoice {
        guard let name = session else {
            return voiceManager.defaultAgentVoice
        }
        let match = sessions.values.first { $0.name == name }
        return match?.voice ?? voiceManager.defaultAgentVoice
    }

    /// Look up the pitch multiplier for a session by name.
    ///
    /// - Parameter session: The session name to look up.
    /// - Returns: The pitch multiplier, defaulting to 1.0 if not found.
    public func pitchFor(session: String?) -> Float {
        guard let name = session else {
            return 1.0
        }
        let match = sessions.values.first { $0.name == name }
        return match?.pitchMultiplier ?? 1.0
    }

    // MARK: - State Queries

    /// Get a snapshot of all registered sessions for the HTTP /state endpoint.
    public func getState() -> RegistrySnapshot {
        let snapshots = sessions.values.map { state in
            SessionSnapshot(
                name: state.name,
                tty: state.tty,
                cwd: state.cwd,
                context: state.context,
                recentMessages: state.recentMessages,
                status: state.status,
                lastActivity: state.lastActivity,
                pitchMultiplier: state.pitchMultiplier,
                sessionId: state.sessionId
            )
        }
        return RegistrySnapshot(sessions: snapshots, sessionCount: snapshots.count)
    }

    /// Get all currently registered session states.
    public func allSessions() -> [SessionState] {
        Array(sessions.values)
    }

    /// Look up a session state by name.
    ///
    /// - Parameter name: The session name to look up.
    /// - Returns: The SessionState if found.
    public func session(named name: String) -> SessionState? {
        sessions.values.first { $0.name == name }
    }

    /// Look up a session state by TTY.
    ///
    /// - Parameter tty: The TTY path to look up.
    /// - Returns: The SessionState if found.
    public func session(byTTY tty: String) -> SessionState? {
        sessions[tty]
    }
}

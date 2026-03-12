// swiftlint:disable:this file_name
import AVFoundation
import Foundation

/// Status of a registered Claude Code session.
///
/// Tracks the session's current operational state for routing context
/// and status reporting via the HTTP /state endpoint.
public enum SessionStatus: String, Codable, Sendable {
    case idle
    case thinking
    case toolCalling = "tool_calling"
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
    public var name: String
    /// The typed terminal identifier for this session (TTY path or Ghostty terminal ID).
    public let identifier: TerminalIdentifier
    /// The working directory of the session.
    public var cwd: String
    /// A summary of what the session is working on.
    public var context: String
    /// Recent conversation messages for routing context.
    public var recentMessages: [SessionMessage]
    /// The current operational status of the session.
    public var status: SessionStatus
    /// When the session last had activity.
    public var lastActivity: Date
    /// The assigned speech synthesis voice descriptor (Apple + Qwen3-TTS).
    public let voice: VoiceDescriptor
    /// The pitch multiplier for auditory differentiation.
    public let pitchMultiplier: Float
    /// The Claude Code session identifier, set via hook integration.
    public var sessionId: String?

    /// Backward-compatible accessor for the TTY path or terminal description.
    public var tty: String { identifier.legacyTTYString }

    /// Creates a new session state with a typed terminal identifier.
    public init(
        name: String,
        identifier: TerminalIdentifier,
        cwd: String,
        context: String,
        recentMessages: [SessionMessage],
        status: SessionStatus,
        lastActivity: Date,
        voice: VoiceDescriptor,
        pitchMultiplier: Float,
        sessionId: String? = nil
    ) {
        self.name = name
        self.identifier = identifier
        self.cwd = cwd
        self.context = context
        self.recentMessages = recentMessages
        self.status = status
        self.lastActivity = lastActivity
        self.voice = voice
        self.pitchMultiplier = pitchMultiplier
        self.sessionId = sessionId
    }

    /// Convenience initializer for iTerm2 sessions identified by TTY path.
    public init(
        name: String,
        tty: String,
        cwd: String,
        context: String,
        recentMessages: [SessionMessage],
        status: SessionStatus,
        lastActivity: Date,
        voice: VoiceDescriptor,
        pitchMultiplier: Float,
        sessionId: String? = nil
    ) {
        self.init(
            name: name,
            identifier: .tty(tty),
            cwd: cwd,
            context: context,
            recentMessages: recentMessages,
            status: status,
            lastActivity: lastActivity,
            voice: voice,
            pitchMultiplier: pitchMultiplier,
            sessionId: sessionId
        )
    }

    /// Returns a copy of this state with a different terminal identifier.
    public func withIdentifier(_ newIdentifier: TerminalIdentifier) -> Self {
        Self(
            name: name,
            identifier: newIdentifier,
            cwd: cwd,
            context: context,
            recentMessages: recentMessages,
            status: status,
            lastActivity: lastActivity,
            voice: voice,
            pitchMultiplier: pitchMultiplier,
            sessionId: sessionId
        )
    }
}

/// Codable snapshot of a session for the HTTP /state endpoint.
///
/// Excludes VoiceDescriptor (not Codable) and includes only
/// the fields needed by external consumers querying GET /state.
/// The `tty` field is encoded in JSON for backward compatibility
/// with existing consumers; `identifier` provides the typed key.
public struct SessionSnapshot: Codable, Sendable {
    // MARK: - Subtypes

    private enum CodingKeys: String, CodingKey {
        case name, identifier, tty, cwd, context, recentMessages, status, lastActivity, pitchMultiplier, sessionId
    }

    // MARK: - Properties

    /// The human-readable session name.
    public let name: String
    /// The typed terminal identifier for this session.
    public let identifier: TerminalIdentifier
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

    /// Backward-compatible accessor for the TTY path or terminal description.
    public var tty: String { identifier.legacyTTYString }

    // MARK: - Initialization

    /// Creates a new session snapshot with a typed terminal identifier.
    public init(
        name: String,
        identifier: TerminalIdentifier,
        cwd: String,
        context: String,
        recentMessages: [SessionMessage],
        status: SessionStatus,
        lastActivity: Date,
        pitchMultiplier: Float,
        sessionId: String?
    ) {
        self.name = name
        self.identifier = identifier
        self.cwd = cwd
        self.context = context
        self.recentMessages = recentMessages
        self.status = status
        self.lastActivity = lastActivity
        self.pitchMultiplier = pitchMultiplier
        self.sessionId = sessionId
    }

    /// Creates a snapshot from a full session state (excludes non-Codable VoiceDescriptor).
    public init(from state: SessionState) {
        self.init(
            name: state.name,
            identifier: state.identifier,
            cwd: state.cwd,
            context: state.context,
            recentMessages: state.recentMessages,
            status: state.status,
            lastActivity: state.lastActivity,
            pitchMultiplier: state.pitchMultiplier,
            sessionId: state.sessionId
        )
    }

    /// Decodes the snapshot, reading `identifier` if present or falling back to `tty`.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        name = try container.decode(String.self, forKey: .name)
        if let id = try container.decodeIfPresent(TerminalIdentifier.self, forKey: .identifier) {
            identifier = id
        } else {
            let ttyPath = try container.decode(String.self, forKey: .tty)
            identifier = .tty(ttyPath)
        }
        cwd = try container.decode(String.self, forKey: .cwd)
        context = try container.decode(String.self, forKey: .context)
        recentMessages = try container.decode([SessionMessage].self, forKey: .recentMessages)
        status = try container.decode(SessionStatus.self, forKey: .status)
        lastActivity = try container.decode(Date.self, forKey: .lastActivity)
        pitchMultiplier = try container.decode(Float.self, forKey: .pitchMultiplier)
        sessionId = try container.decodeIfPresent(String.self, forKey: .sessionId)
    }

    // MARK: - Codable Encoding

    /// Encodes the snapshot, including both `identifier` and `tty` for backward compatibility.
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(name, forKey: .name)
        try container.encode(identifier, forKey: .identifier)
        try container.encode(tty, forKey: .tty)
        try container.encode(cwd, forKey: .cwd)
        try container.encode(context, forKey: .context)
        try container.encode(recentMessages, forKey: .recentMessages)
        try container.encode(status, forKey: .status)
        try container.encode(lastActivity, forKey: .lastActivity)
        try container.encode(pitchMultiplier, forKey: .pitchMultiplier)
        try container.encodeIfPresent(sessionId, forKey: .sessionId)
    }
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

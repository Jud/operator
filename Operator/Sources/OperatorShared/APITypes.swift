// swiftlint:disable:this file_name
import Foundation

/// JSON body for POST /speak.
///
/// Matches the MCP server's speak() tool output: message text, optional
/// session name for voice lookup, and optional priority level.
public struct SpeakRequest: Codable, Sendable {
    /// The text content to speak through the audio queue.
    public let message: String
    /// Session name for voice lookup (typically the CWD basename).
    public let session: String?
    /// Priority level: "normal" (default) or "urgent" (skips to front of queue).
    public let priority: String?

    /// Creates a new speak request.
    public init(message: String, session: String? = nil, priority: String? = nil) {
        self.message = message
        self.session = session
        self.priority = priority
    }
}

/// JSON response for successful operations.
public struct OkResponse: Codable, Sendable {
    /// Whether the operation succeeded.
    public let ok: Bool

    /// Creates a new ok response.
    public init(ok: Bool) {
        self.ok = ok
    }
}

/// JSON response for POST /speak confirming message was queued.
public struct QueuedResponse: Codable, Sendable {
    /// Whether the message was successfully queued for playback.
    public let queued: Bool

    /// Creates a new queued response.
    public init(queued: Bool) {
        self.queued = queued
    }
}

/// JSON body for POST /hook/session-start.
///
/// Accepts the Claude Code hook payload for session start events.
/// The optional `terminal_type` field identifies the terminal emulator
/// ("ghostty" or "iterm2"). When absent, defaults to iTerm2 behavior
/// for backward compatibility with older MCP server versions.
public struct HookSessionStartRequest: Codable, Sendable {
    enum CodingKeys: String, CodingKey {
        case sessionId = "session_id"
        case tty
        case cwd
        case terminalType = "terminal_type"
    }

    /// Session identifier, typically the CWD basename.
    public let sessionId: String
    /// TTY device path of the parent shell (e.g., "/dev/ttys004").
    public let tty: String
    /// Current working directory of the session.
    public let cwd: String
    /// Terminal emulator type: "ghostty" or "iterm2".
    public let terminalType: String?

    /// Creates a new session-start request.
    public init(sessionId: String, tty: String, cwd: String, terminalType: String? = nil) {
        self.sessionId = sessionId
        self.tty = tty
        self.cwd = cwd
        self.terminalType = terminalType
    }
}

/// JSON response for POST /hook/session-start with `needs_terminal_id` for Ghostty resolution.
public struct HookSessionStartResponse: Codable, Sendable {
    enum CodingKeys: String, CodingKey {
        case ok
        case needsTerminalId = "needs_terminal_id"
    }

    /// Whether the session-start was accepted.
    public let ok: Bool
    /// When true, the MCP server should resolve and POST the Ghostty terminal ID.
    public let needsTerminalId: Bool

    /// Creates a new session-start response.
    public init(ok: Bool, needsTerminalId: Bool) {
        self.ok = ok
        self.needsTerminalId = needsTerminalId
    }
}

/// JSON body for POST /hook/terminal-id mapping a TTY to its Ghostty terminal ID.
public struct HookTerminalIdRequest: Codable, Sendable {
    enum CodingKeys: String, CodingKey {
        case tty
        case ghosttyId = "ghostty_id"
    }

    /// TTY device path of the session.
    public let tty: String
    /// Resolved Ghostty terminal identifier.
    public let ghosttyId: String

    /// Creates a new terminal-id request.
    public init(tty: String, ghosttyId: String) {
        self.tty = tty
        self.ghosttyId = ghosttyId
    }
}

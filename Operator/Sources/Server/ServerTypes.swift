import Foundation
import Hummingbird

/// JSON body for POST /speak.
///
/// Matches the MCP server's speak() tool output: message text, optional
/// session name for voice lookup, and optional priority level.
public struct SpeakRequest: Decodable, Sendable {
    let message: String
    let session: String?
    let priority: String?
}

/// JSON response for successful operations.
public struct OkResponse: ResponseEncodable, Sendable {
    let ok: Bool
}

/// JSON response for POST /speak confirming message was queued.
public struct QueuedResponse: ResponseEncodable, Sendable {
    let queued: Bool
}

/// JSON response for GET /state.
///
/// Returns daemon state, audio queue depth, and all registered sessions.
public struct StateResponse: ResponseEncodable, Sendable {
    let state: String
    let queueLength: Int
    let sessions: [SessionSnapshot]
}

/// JSON body for POST /hook/session-start.
///
/// Accepts the Claude Code hook payload for session start events.
/// The optional `terminal_type` field identifies the terminal emulator
/// ("ghostty" or "iterm2"). When absent, defaults to iTerm2 behavior
/// for backward compatibility with older MCP server versions.
public struct HookSessionStartRequest: Decodable, Sendable {
    enum CodingKeys: String, CodingKey {
        case sessionId = "session_id"
        case tty
        case cwd
        case terminalType = "terminal_type"
    }

    let sessionId: String
    let tty: String
    let cwd: String
    let terminalType: String?
}

/// JSON response for POST /hook/session-start with `needs_terminal_id` for Ghostty resolution.
public struct HookSessionStartResponse: ResponseEncodable, Sendable {
    enum CodingKeys: String, CodingKey {
        case ok
        case needsTerminalId = "needs_terminal_id"
    }

    let ok: Bool
    let needsTerminalId: Bool
}

/// JSON body for POST /hook/terminal-id mapping a TTY to its Ghostty terminal ID.
public struct HookTerminalIdRequest: Decodable, Sendable {
    enum CodingKeys: String, CodingKey {
        case tty
        case ghosttyId = "ghostty_id"
    }

    let tty: String
    let ghosttyId: String
}

/// JSON body for POST /hook/stop.
///
/// Accepts the Claude Code hook payload for stop events, including
/// the last assistant message for context tracking.
public struct HookStopRequest: Decodable, Sendable {
    enum CodingKeys: String, CodingKey {
        case sessionId = "session_id"
        case tty
        case lastAssistantMessage = "last_assistant_message"
    }

    let sessionId: String
    let tty: String
    let lastAssistantMessage: String?
}

/// JSON body for POST /hook/session-end.
///
/// Accepts the Claude Code hook payload for session end events.
public struct HookSessionEndRequest: Decodable, Sendable {
    enum CodingKeys: String, CodingKey {
        case sessionId = "session_id"
        case tty
    }

    let sessionId: String
    let tty: String
}

/// Errors specific to the HTTP server setup.
public enum OperatorHTTPServerError: Error, CustomStringConvertible {
    case tokenFileUnreadable

    case portInUse

    /// A human-readable description of the error.
    public var description: String {
        switch self {
        case .tokenFileUnreadable:
            return "Could not read bearer token from ~/.operator/token"

        case .portInUse:
            return "Port 7420 is already in use. Another Operator instance may be running."
        }
    }
}

import Foundation
import Hummingbird
@_exported import OperatorShared

// MARK: - OperatorShared ResponseEncodable Conformances

extension OkResponse: ResponseEncodable {}
extension QueuedResponse: ResponseEncodable {}
extension HookSessionStartResponse: ResponseEncodable {}

// MARK: - Daemon-Only Types

/// JSON response for GET /state.
///
/// Returns daemon state, audio queue depth, and all registered sessions.
public struct StateResponse: ResponseEncodable, Sendable {
    let state: String
    let queueLength: Int
    let sessions: [SessionSnapshot]
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

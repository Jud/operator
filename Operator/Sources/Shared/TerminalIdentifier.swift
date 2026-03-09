import Foundation

/// The type of terminal emulator hosting a session.
///
/// Used for per-type discovery reconciliation and routing delivery
/// to the correct bridge implementation.
public enum TerminalType: String, Sendable, Codable {
    case iterm
    case ghostty
    case unknown
}

/// A typed session identifier that distinguishes between terminal emulators.
///
/// iTerm2 sessions are identified by their TTY device path (globally unique
/// and stable). Ghostty sessions are identified by the terminal's internal ID
/// from AppleScript (stable for the terminal's lifetime). Using a typed enum
/// prevents accidental mixing and enables the composite bridge to route
/// delivery to the correct terminal-specific bridge.
public enum TerminalIdentifier: Sendable, Hashable, CustomStringConvertible {
    /// An iTerm2 session identified by its TTY device path (e.g., "/dev/ttys004").
    case tty(String)
    /// A Ghostty session identified by its AppleScript terminal ID.
    case ghosttyTerminal(String)

    /// The terminal type for this identifier.
    public var terminalType: TerminalType {
        switch self {
        case .tty: return .iterm
        case .ghosttyTerminal: return .ghostty
        }
    }

    /// A human-readable description of the identifier.
    public var description: String {
        switch self {
        case .tty(let path): return "tty:\(path)"
        case .ghosttyTerminal(let id): return "ghostty:\(id)"
        }
    }
}

// MARK: - Codable

extension TerminalIdentifier: Codable {
    private enum CodingKeys: String, CodingKey {
        case type
        case value
    }

    /// Creates a terminal identifier by decoding from the given decoder.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        let value = try container.decode(String.self, forKey: .value)
        switch type {
        case "tty":
            self = .tty(value)

        case "ghosttyTerminal":
            self = .ghosttyTerminal(value)

        default:
            throw DecodingError.dataCorruptedError(
                forKey: .type,
                in: container,
                debugDescription: "Unknown TerminalIdentifier type: \(type)"
            )
        }
    }

    /// Encodes the terminal identifier into the given encoder.
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .tty(let path):
            try container.encode("tty", forKey: .type)
            try container.encode(path, forKey: .value)

        case .ghosttyTerminal(let id):
            try container.encode("ghosttyTerminal", forKey: .type)
            try container.encode(id, forKey: .value)
        }
    }
}

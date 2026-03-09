import Foundation

/// A terminal session discovered during enumeration.
///
/// Terminal-agnostic representation of a session found by any bridge
/// implementation (iTerm2, Ghostty, etc.). Sessions are identified by
/// their `terminalType` and the appropriate native identifier (TTY for
/// iTerm2, terminal ID for Ghostty).
public struct DiscoveredSession: Codable, Sendable {
    // MARK: - Subtypes

    private enum CodingKeys: String, CodingKey {
        case id, tty, name, isProcessing, workingDirectory, jobName, terminalType
    }

    // MARK: - Properties

    /// The unique session identifier from the terminal emulator.
    public let id: String
    /// The TTY device path for this session (empty string for terminals that don't expose TTY).
    public let tty: String
    /// The display name of the session.
    public let name: String
    /// Whether the session is currently processing a command.
    public let isProcessing: Bool
    /// The current working directory of the session, if available.
    public let workingDirectory: String?
    /// The name of the running job in the session, if available.
    public let jobName: String?
    /// The type of terminal emulator that discovered this session.
    public let terminalType: TerminalType

    // MARK: - Initialization

    /// Creates a new discovered session representation.
    public init(
        id: String,
        tty: String,
        name: String,
        isProcessing: Bool,
        workingDirectory: String?,
        jobName: String?,
        terminalType: TerminalType = .iterm
    ) {
        self.id = id
        self.tty = tty
        self.name = name
        self.isProcessing = isProcessing
        self.workingDirectory = workingDirectory
        self.jobName = jobName
        self.terminalType = terminalType
    }

    /// Custom decoder that defaults `terminalType` to `.iterm` when absent.
    ///
    /// This ensures backward compatibility with JXA discovery output that
    /// does not include a `terminalType` field.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        tty = try container.decode(String.self, forKey: .tty)
        name = try container.decode(String.self, forKey: .name)
        isProcessing = try container.decode(Bool.self, forKey: .isProcessing)
        workingDirectory = try container.decodeIfPresent(String.self, forKey: .workingDirectory)
        jobName = try container.decodeIfPresent(String.self, forKey: .jobName)
        terminalType = try container.decodeIfPresent(TerminalType.self, forKey: .terminalType) ?? .iterm
    }
}

/// Protocol boundary for terminal I/O operations (P1).
///
/// Abstracts session discovery and text delivery so the rest of
/// the system is decoupled from any specific terminal emulator.
public protocol TerminalBridge: Sendable {
    /// Deliver text to a terminal session identified by a typed terminal identifier.
    ///
    /// - Parameters:
    ///   - identifier: The terminal identifier (TTY path or Ghostty terminal ID) for the target session.
    ///   - text: The message text to deliver.
    /// - Returns: `true` if the session was found and text was delivered.
    /// - Throws: An error if the terminal is not running or the write fails.
    func writeToSession(identifier: TerminalIdentifier, text: String) async throws -> Bool

    /// Discover all terminal sessions across all windows and tabs.
    ///
    /// - Returns: Array of `DiscoveredSession` structs representing all discovered sessions.
    func discoverSessions() async throws -> [DiscoveredSession]
}

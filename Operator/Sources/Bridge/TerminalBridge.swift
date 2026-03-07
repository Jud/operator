import Foundation

/// A terminal session discovered during enumeration.
///
/// Terminal-agnostic representation of a session found by any bridge
/// implementation (iTerm2, Terminal.app, etc.). The TTY path is the
/// primary stable identifier.
public struct DiscoveredSession: Codable, Sendable {
    /// The unique session identifier from the terminal emulator.
    public let id: String
    /// The TTY device path for this session.
    public let tty: String
    /// The display name of the session.
    public let name: String
    /// Whether the session is currently processing a command.
    public let isProcessing: Bool
    /// The current working directory of the session, if available.
    public let workingDirectory: String?
    /// The name of the running job in the session, if available.
    public let jobName: String?

    /// Creates a new discovered session representation.
    public init(
        id: String,
        tty: String,
        name: String,
        isProcessing: Bool,
        workingDirectory: String?,
        jobName: String?
    ) {
        self.id = id
        self.tty = tty
        self.name = name
        self.isProcessing = isProcessing
        self.workingDirectory = workingDirectory
        self.jobName = jobName
    }
}

/// Protocol boundary for terminal I/O operations (P1).
///
/// Abstracts session discovery and text delivery so the rest of
/// the system is decoupled from any specific terminal emulator.
public protocol TerminalBridge: Sendable {
    /// Deliver text to a terminal session identified by TTY path.
    ///
    /// - Parameters:
    ///   - tty: The TTY device path (e.g., "/dev/ttys004") identifying the target session.
    ///   - text: The message text to deliver.
    /// - Returns: `true` if the session was found and text was delivered.
    /// - Throws: An error if the terminal is not running or the write fails.
    func writeToSession(tty: String, text: String) async throws -> Bool

    /// Discover all terminal sessions across all windows and tabs.
    ///
    /// - Returns: Array of `DiscoveredSession` structs representing all discovered sessions.
    func discoverSessions() async throws -> [DiscoveredSession]
}

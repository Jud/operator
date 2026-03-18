import Foundation
import OperatorShared

/// Periodic session registration with the Operator daemon.
///
/// Sends a heartbeat POST to `/hook/session-start` immediately on startup,
/// then every 5 seconds. After the first successful heartbeat, sends a spoken
/// greeting and triggers Ghostty terminal ID resolution if requested by the daemon.
/// All daemon communication failures are silently swallowed.
public struct Heartbeat: Sendable {
    /// Daemon HTTP client for heartbeat and greeting POST requests.
    public let client: DaemonClient
    /// Ghostty terminal ID resolver triggered on first heartbeat.
    public let ghosttyResolver: GhosttyResolver
    /// Session name derived from the CWD basename.
    public let sessionName: String
    /// TTY device path of the parent shell.
    public let tty: String
    /// Current working directory of the session.
    public let cwd: String
    /// Terminal emulator type ("ghostty" or "iterm2").
    public let terminalType: String

    /// Creates a new heartbeat instance.
    public init(
        client: DaemonClient,
        ghosttyResolver: GhosttyResolver,
        sessionName: String,
        tty: String,
        cwd: String,
        terminalType: String
    ) {
        self.client = client
        self.ghosttyResolver = ghosttyResolver
        self.sessionName = sessionName
        self.tty = tty
        self.cwd = cwd
        self.terminalType = terminalType
    }

    /// Run the heartbeat registration loop.
    ///
    /// Sends the first heartbeat immediately, then repeats every 5 seconds.
    /// The daemon handles the startup greeting when a new session is registered,
    /// so the MCP server does not send one.
    /// When the daemon response includes `needs_terminal_id: true`, delegates to
    /// the `GhosttyResolver` for one-shot terminal ID resolution.
    ///
    /// This method runs indefinitely until the task is cancelled.
    public func heartbeatLoop() async {
        while !Task.isCancelled {
            _ = await sendHeartbeat()
            try? await Task.sleep(for: .seconds(5))
        }
    }

    /// POST a session-start heartbeat to the daemon.
    ///
    /// Returns `true` if the daemon acknowledged the heartbeat. On success,
    /// also checks whether Ghostty terminal ID resolution is needed.
    private func sendHeartbeat() async -> Bool {
        let request = HookSessionStartRequest(
            sessionId: sessionName,
            tty: tty,
            cwd: cwd,
            terminalType: terminalType
        )

        guard
            let response: HookSessionStartResponse = await client.post(
                path: "/hook/session-start",
                body: request
            )
        else {
            return false
        }

        if response.needsTerminalId {
            await ghosttyResolver.resolveIfNeeded()
        }

        return true
    }
}

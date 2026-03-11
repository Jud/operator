import Foundation
import OperatorShared

/// Periodic session registration with the Operator daemon.
///
/// Sends a heartbeat POST to `/hook/session-start` immediately on startup,
/// then every 5 seconds. After the first successful heartbeat, sends a spoken
/// greeting and triggers Ghostty terminal ID resolution if requested by the daemon.
/// All daemon communication failures are silently swallowed.
internal struct Heartbeat: Sendable {
    let client: DaemonClient
    let ghosttyResolver: GhosttyResolver
    let sessionName: String
    let tty: String
    let cwd: String
    let terminalType: String

    /// Run the heartbeat registration loop.
    ///
    /// Sends the first heartbeat immediately, then repeats every 5 seconds.
    /// After the first successful heartbeat, sends a startup greeting via `/speak`.
    /// When the daemon response includes `needs_terminal_id: true`, delegates to
    /// the `GhosttyResolver` for one-shot terminal ID resolution.
    ///
    /// This method runs indefinitely until the task is cancelled.
    func heartbeatLoop() async {
        var greetingSent = false

        while !Task.isCancelled {
            let succeeded = await sendHeartbeat()

            if succeeded, !greetingSent {
                greetingSent = true
                await sendGreeting()
            }

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

    /// Send the startup greeting message via the daemon's `/speak` endpoint.
    private func sendGreeting() async {
        let request = SpeakRequest(
            message: "\(sessionName) connected.",
            session: sessionName,
            priority: "normal"
        )
        _ = await client.post(path: "/speak", body: request) as Data?
    }
}

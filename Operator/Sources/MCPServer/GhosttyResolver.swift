import Foundation
import OperatorShared
import os

/// Resolves the Ghostty terminal ID for a session by writing an ANSI title
/// marker to the TTY device and enumerating Ghostty windows via JXA.
///
/// Resolution is attempted at most once per session lifetime. If the TTY is
/// `"unknown"` or resolution has already been attempted, the call is a no-op.
/// All errors are silently swallowed and logged to stderr.
public final class GhosttyResolver: Sendable {
    private let client: DaemonClient
    private let tty: String
    private let resolved: OSAllocatedUnfairLock<Bool>

    /// Creates a new Ghostty resolver.
    /// - Parameters:
    ///   - client: Daemon HTTP client for posting the resolved terminal ID.
    ///   - tty: TTY device path of the session.
    public init(client: DaemonClient, tty: String) {
        self.client = client
        self.tty = tty
        self.resolved = OSAllocatedUnfairLock(initialState: false)
    }

    /// Attempt to resolve and register the Ghostty terminal ID.
    ///
    /// Writes a UUID-tagged ANSI title escape to the TTY device file,
    /// waits 500ms for Ghostty to process the title change, then runs
    /// a JXA script to enumerate Ghostty windows/tabs/terminals and
    /// find the marker. On success, POSTs the terminal ID to the daemon.
    func resolveIfNeeded() async {
        guard tty != "unknown" else {
            return
        }

        let alreadyResolved = resolved.withLock { value -> Bool in
            if value {
                return true
            }
            value = true
            return false
        }
        guard !alreadyResolved else {
            return
        }

        do {
            try await resolve()
        } catch {
            MCPLog.write("Ghostty resolution failed: \(error)")
        }
    }

    private func resolve() async throws {
        let marker = "OPERATOR-\(UUID().uuidString)"

        guard let fileHandle = FileHandle(forWritingAtPath: tty) else {
            MCPLog.write("Cannot open TTY device for writing: \(tty)")
            return
        }
        defer { fileHandle.closeFile() }

        let escape = "\u{1b}]2;\(marker)\u{07}"
        guard let escapeData = escape.data(using: .utf8) else {
            return
        }
        fileHandle.write(escapeData)

        try await Task.sleep(for: .milliseconds(500))

        guard let ghosttyId = try runJXAEnumeration(marker: marker) else {
            return
        }

        let request = HookTerminalIdRequest(tty: tty, ghosttyId: ghosttyId)
        _ = await client.postRaw(path: "/hook/terminal-id", body: request)
    }

    private func runJXAEnumeration(marker: String) throws -> String? {
        let script = """
            (function() {
                var app = Application("Ghostty");
                var wins = app.windows();
                for (var i = 0; i < wins.length; i++) {
                    var tabs = wins[i].tabs();
                    for (var j = 0; j < tabs.length; j++) {
                        var terms = tabs[j].terminals();
                        for (var k = 0; k < terms.length; k++) {
                            if (terms[k].name().includes("\(marker)")) {
                                return String(terms[k].id());
                            }
                        }
                    }
                }
                return "not_found";
            })();
            """

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
        process.arguments = ["-l", "JavaScript", "-e", script]

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = Pipe()

        try process.run()
        process.waitUntilExit()

        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let output =
            String(data: data, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !output.isEmpty, output != "not_found" else {
            return nil
        }
        return output
    }
}

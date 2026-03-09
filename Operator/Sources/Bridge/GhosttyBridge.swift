import Foundation

/// Errors that can occur during Ghostty JXA bridge operations.
public enum GhosttyBridgeError: Error, CustomStringConvertible {
    /// The osascript process exited with a non-zero status code.
    case scriptFailed(status: Int32, stderr: String)

    /// Ghostty is not running; JXA cannot connect to the application.
    case ghosttyNotRunning

    /// The target terminal was not found by its Ghostty ID.
    case terminalNotFound(id: String)

    /// Failed to write the message text to a temporary file for file-based delivery.
    case tempFileWriteFailed(underlying: Error)

    /// Failed to decode session discovery JSON output.
    case discoveryDecodeFailed(output: String, underlying: Error)

    /// A human-readable description of the error.
    public var description: String {
        switch self {
        case .scriptFailed(let status, let stderr):
            return "osascript exited with status \(status): \(stderr)"

        case .ghosttyNotRunning:
            return "Ghostty is not running"

        case .terminalNotFound(let id):
            return "Ghostty terminal not found for ID: \(id)"

        case .tempFileWriteFailed(let underlying):
            return "Failed to write temp file for JXA delivery: \(underlying.localizedDescription)"

        case .discoveryDecodeFailed(let output, let underlying):
            return
                "Failed to decode Ghostty discovery output: "
                + "\(underlying.localizedDescription) -- raw: \(output.prefix(200))"
        }
    }
}

/// Bridge to Ghostty via JXA (JavaScript for Automation) for session discovery
/// and text delivery to Claude Code TUI sessions.
///
/// Two primary operations:
///
/// **writeToSession** uses file-based delivery to avoid escaping issues:
/// 1. Write message text to a temp file
/// 2. JXA reads the temp file via ObjC.import('Foundation') + NSString.stringWithContentsOfFile
/// 3. Find terminal by Ghostty ID match
/// 4. Execute two-step sequence: inputText (paste) -> sendKey return (submit)
/// 5. Clean up temp file
///
/// **discoverSessions** enumerates all Ghostty terminals across all windows and tabs,
/// returning session metadata including terminal ID, name, and working directory.
/// Ghostty does not expose TTY, isProcessing, or jobName, so those default to
/// empty string, false, and nil respectively.
///
/// Both operations invoke `/usr/bin/osascript -l JavaScript` via the `runOsascript()` helper.
public class GhosttyBridge: @unchecked Sendable, TerminalBridge {
    private static let logger = Log.logger(for: "GhosttyBridge")

    /// Creates a new Ghostty bridge instance.
    public init() {}

    /// Deliver text to a specific Ghostty terminal identified by a terminal identifier.
    ///
    /// Uses file-based delivery matching the ITermBridge pattern:
    /// text is written to a temp file and read by JXA via ObjC.import('Foundation'),
    /// avoiding all string escaping issues (nested shell + JXA quoting).
    ///
    /// - Parameters:
    ///   - identifier: Must be `.ghosttyTerminal(id)`. Throws `terminalNotFound` for `.tty`.
    ///   - text: The message text to deliver.
    /// - Returns: `true` if the terminal was found and text was delivered.
    /// - Throws: `GhosttyBridgeError` on JXA failure or temp file write failure.
    public func writeToSession(identifier: TerminalIdentifier, text: String) async throws -> Bool {
        guard case .ghosttyTerminal(let ghosttyId) = identifier else {
            throw GhosttyBridgeError.terminalNotFound(id: identifier.description)
        }
        Self.logger.info("Delivering text to Ghostty terminal: \(ghosttyId) (\(text.count) chars)")

        let tempFile = FileManager.default.temporaryDirectory
            .appendingPathComponent("operator-\(UUID().uuidString).txt")

        do {
            try text.write(to: tempFile, atomically: true, encoding: .utf8)
        } catch {
            throw GhosttyBridgeError.tempFileWriteFailed(underlying: error)
        }

        defer {
            try? FileManager.default.removeItem(at: tempFile)
            Self.logger.debug("Cleaned up temp file: \(tempFile.path)")
        }

        let output = try await runWriteScript(ghosttyId: ghosttyId, tempFilePath: tempFile.path)

        if output.contains("terminal_not_found") {
            Self.logger.warning("Ghostty terminal not found for ID: \(ghosttyId)")
            throw GhosttyBridgeError.terminalNotFound(id: ghosttyId)
        }

        let success = output.contains("ok")
        if success {
            Self.logger.info("Successfully delivered text to Ghostty terminal: \(ghosttyId)")
        }
        return success
    }

    /// Build and run the JXA write script for Ghostty delivery.
    private func runWriteScript(ghosttyId: String, tempFilePath: String) async throws -> String {
        let script = """
            (function() {
                ObjC.import('Foundation');
                const textData = $.NSString.stringWithContentsOfFileEncodingError(
                    '\(tempFilePath)', $.NSUTF8StringEncoding, null);
                const text = textData.js;
                const targetId = '\(ghosttyId)';
                const app = Application("Ghostty");
                for (const win of app.windows()) {
                    for (const tab of win.tabs()) {
                        for (const term of tab.terminals()) {
                            if (String(term.id()) === targetId) {
                                term.inputText({text: text});
                                term.sendKey({key: "return"});
                                return "ok";
                            }
                        }
                    }
                }
                return "terminal_not_found";
            })();
            """

        return try await runOsascript(script)
    }

    /// Discover all Ghostty terminals across all windows and tabs.
    ///
    /// Ghostty does not expose TTY paths, isProcessing state, or jobName.
    /// These default to empty string, false, and nil respectively.
    ///
    /// - Returns: Array of `DiscoveredSession` structs representing all discovered terminals.
    /// - Throws: `GhosttyBridgeError` on JXA failure or JSON decode failure.
    public func discoverSessions() async throws -> [DiscoveredSession] {
        Self.logger.debug("Discovering Ghostty sessions")

        let script = """
            const app = Application("Ghostty");
            const sessions = [];
            for (const win of app.windows()) {
                for (const tab of win.tabs()) {
                    for (const term of tab.terminals()) {
                        sessions.push({
                            id: String(term.id()),
                            tty: "",
                            name: term.name(),
                            isProcessing: false,
                            workingDirectory: term.workingDirectory() || null,
                            jobName: null,
                            terminalType: "ghostty"
                        });
                    }
                }
            }
            JSON.stringify(sessions);
            """

        let output = try await runOsascript(script)
        let sessions = try decodeDiscoveryOutput(output)
        Self.logger.info("Discovered \(sessions.count) Ghostty session(s)")
        return sessions
    }

    /// Decode JXA discovery JSON output into DiscoveredSession objects.
    private func decodeDiscoveryOutput(_ output: String) throws -> [DiscoveredSession] {
        guard let data = output.data(using: .utf8) else {
            throw GhosttyBridgeError.discoveryDecodeFailed(
                output: output,
                underlying: NSError(
                    domain: "GhosttyBridge",
                    code: -1,
                    userInfo: [
                        NSLocalizedDescriptionKey: "Failed to convert output to UTF-8 data"
                    ]
                )
            )
        }
        do {
            return try JSONDecoder().decode([DiscoveredSession].self, from: data)
        } catch {
            throw GhosttyBridgeError.discoveryDecodeFailed(output: output, underlying: error)
        }
    }

    /// Execute a JXA (JavaScript for Automation) script via /usr/bin/osascript.
    ///
    /// Detects "not running" / "connection is invalid" patterns in stderr to
    /// throw `ghosttyNotRunning` instead of a generic script failure.
    ///
    /// - Parameter script: The JavaScript source code to execute.
    /// - Returns: The trimmed stdout output from the script.
    /// - Throws: `GhosttyBridgeError.ghosttyNotRunning` if Ghostty is not running,
    ///   or `.scriptFailed` for other osascript failures.
    private func runOsascript(_ script: String) async throws -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
        process.arguments = ["-l", "JavaScript", "-e", script]

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        do {
            try process.run()
        } catch {
            throw GhosttyBridgeError.scriptFailed(
                status: -1,
                stderr: "Failed to launch osascript: \(error.localizedDescription)"
            )
        }

        // Read stdout and stderr before waitUntilExit to avoid deadlock on full pipe buffers
        let stdoutData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
        let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()

        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            let stderrText = String(data: stderrData, encoding: .utf8) ?? ""
            let lower = stderrText.lowercased()
            if lower.contains("is not running") || lower.contains("not running")
                || lower.contains("connection is invalid")
                || lower.contains("application can't be found")
            {
                throw GhosttyBridgeError.ghosttyNotRunning
            }
            throw GhosttyBridgeError.scriptFailed(
                status: process.terminationStatus,
                stderr: stderrText
            )
        }

        let output = String(data: stdoutData, encoding: .utf8) ?? ""
        return output.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

import Foundation
import OSLog

/// Errors that can occur during iTerm2 JXA bridge operations.
public enum ITermBridgeError: Error, CustomStringConvertible {
    /// The osascript process exited with a non-zero status code.
    case jxaFailed(status: Int32, stderr: String)

    /// iTerm2 is not running; JXA cannot connect to the application.
    case itermNotRunning

    /// The target session was not found by TTY path.
    case sessionNotFound(tty: String)

    /// Failed to write the message text to a temporary file for file-based delivery.
    case tempFileWriteFailed(underlying: Error)

    /// Failed to decode session discovery JSON output.
    case discoveryDecodeFailed(output: String, underlying: Error)

    /// A human-readable description of the error.
    public var description: String {
        switch self {
        case .jxaFailed(let status, let stderr):
            return "osascript exited with status \(status): \(stderr)"

        case .itermNotRunning:
            return "iTerm2 is not running"

        case .sessionNotFound(let tty):
            return "iTerm session not found for TTY: \(tty)"

        case .tempFileWriteFailed(let underlying):
            return "Failed to write temp file for JXA delivery: \(underlying.localizedDescription)"

        case .discoveryDecodeFailed(let output, let underlying):
            return
                "Failed to decode session discovery output: "
                + "\(underlying.localizedDescription) -- raw: \(output.prefix(200))"
        }
    }
}

/// Represents a discovered iTerm2 session with its identifying properties.
///
/// Sessions are discovered via JXA enumeration of all windows, tabs, and panes.
/// The TTY path is the primary stable identifier -- it remains constant across
/// tab switches and pane moves (per research doc "Key Insight").
public struct ITermSession: Codable, Sendable {
    /// The unique session identifier from iTerm2.
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

    /// Creates a new iTerm session representation.
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

/// Bridge to iTerm2 via JXA (JavaScript for Automation) for session discovery
/// and text delivery to Claude Code TUI sessions.
///
/// Two primary operations:
///
/// **writeToSession** uses file-based delivery to avoid escaping issues:
/// 1. Write message text to a temp file
/// 2. JXA reads the temp file via ObjC.import('Foundation') + NSString.stringWithContentsOfFile
/// 3. Find session by TTY match (globally unique, stable across tab switches)
/// 4. Execute three-step sequence: Escape (clear) -> 300ms delay -> text -> 200ms delay -> CR (submit)
/// 5. Clean up temp file
///
/// **discoverSessions** enumerates all iTerm2 sessions across all windows and tabs,
/// returning session metadata including TTY, name, processing state, and working directory.
///
/// Both operations invoke `/usr/bin/osascript -l JavaScript` via the `runOsascript()` helper.
///
/// Reference: technical-spec.md Component 5; research doc "What Worked: The Proven Recipe"
public class ITermBridge: @unchecked Sendable {
    private static let logger = Logger(subsystem: "com.operator.app", category: "ITermBridge")

    /// Creates a new iTerm bridge instance.
    public init() {}

    /// Deliver text to a specific iTerm2 session identified by TTY path.
    ///
    /// Uses file-based delivery per technical-spec.md Key Design Decision #3:
    /// text is written to a temp file and read by JXA via ObjC.import('Foundation'),
    /// avoiding all string escaping issues (nested shell + JXA quoting).
    ///
    /// - Parameters:
    ///   - tty: The TTY device path (e.g., "/dev/ttys004") identifying the target session.
    ///   - text: The message text to deliver.
    /// - Returns: `true` if the session was found and text was delivered.
    /// - Throws: `ITermBridgeError` on JXA failure or temp file write failure.
    public func writeToSession(tty: String, text: String) async throws -> Bool {
        Self.logger.info("Delivering text to session at TTY: \(tty) (\(text.count) chars)")

        let tempFile = FileManager.default.temporaryDirectory
            .appendingPathComponent("operator-\(UUID().uuidString).txt")

        do {
            try text.write(to: tempFile, atomically: true, encoding: .utf8)
        } catch {
            throw ITermBridgeError.tempFileWriteFailed(underlying: error)
        }

        defer {
            try? FileManager.default.removeItem(at: tempFile)
            Self.logger.debug("Cleaned up temp file: \(tempFile.path)")
        }

        let output = try await runWriteScript(tty: tty, tempFilePath: tempFile.path)

        if output.contains("session_not_found") {
            Self.logger.warning("Session not found for TTY: \(tty)")
            throw ITermBridgeError.sessionNotFound(tty: tty)
        }

        let success = output.contains("ok")
        if success {
            Self.logger.info("Successfully delivered text to TTY: \(tty)")
        }
        return success
    }

    /// Build and run the JXA write script.
    private func runWriteScript(tty: String, tempFilePath: String) async throws -> String {
        let script = """
            ObjC.import('Foundation');
            const textData = $.NSString.stringWithContentsOfFileEncodingError(
                '\(tempFilePath)', $.NSUTF8StringEncoding, null);
            const text = textData.js;
            const tty = '\(tty)';
            const app = Application("iTerm2");
            for (const win of app.windows()) {
                for (const tab of win.tabs()) {
                    for (const sess of tab.sessions()) {
                        if (sess.tty() === tty) {
                            sess.write({ text: "\\u001b", newline: false });
                            delay(0.3);
                            sess.write({ text: text, newline: false });
                            delay(0.2);
                            sess.write({ text: "\\r", newline: false });
                            return "ok";
                        }
                    }
                }
            }
            return "session_not_found";
            """

        return try await runOsascript(script)
    }

    /// Discover all iTerm2 sessions across all windows and tabs.
    ///
    /// - Returns: Array of `ITermSession` structs representing all discovered sessions.
    /// - Throws: `ITermBridgeError` on JXA failure or JSON decode failure.
    public func discoverSessions() async throws -> [ITermSession] {
        Self.logger.debug("Discovering iTerm sessions")

        let script = """
            const app = Application("iTerm2");
            const sessions = [];
            for (const win of app.windows()) {
                for (const tab of win.tabs()) {
                    for (const sess of tab.sessions()) {
                        sessions.push({
                            id: sess.id(),
                            tty: sess.tty(),
                            name: sess.name(),
                            isProcessing: sess.isProcessing(),
                            workingDirectory: sess.variable({ named: "session.path" }),
                            jobName: sess.variable({ named: "session.jobName" })
                        });
                    }
                }
            }
            JSON.stringify(sessions);
            """

        let output = try await runOsascript(script)

        guard let data = output.data(using: .utf8) else {
            throw ITermBridgeError.discoveryDecodeFailed(
                output: output,
                underlying: NSError(
                    domain: "ITermBridge",
                    code: -1,
                    userInfo: [
                        NSLocalizedDescriptionKey: "Failed to convert output to UTF-8 data"
                    ]
                )
            )
        }

        do {
            let sessions = try JSONDecoder().decode([ITermSession].self, from: data)
            Self.logger.info("Discovered \(sessions.count) iTerm session(s)")
            return sessions
        } catch {
            throw ITermBridgeError.discoveryDecodeFailed(output: output, underlying: error)
        }
    }

    /// Execute a JXA (JavaScript for Automation) script via /usr/bin/osascript.
    ///
    /// - Parameter script: The JavaScript source code to execute.
    /// - Returns: The trimmed stdout output from the script.
    /// - Throws: `ITermBridgeError.jxaFailed` if the process exits with non-zero status.
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
            throw ITermBridgeError.jxaFailed(
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
                throw ITermBridgeError.itermNotRunning
            }
            throw ITermBridgeError.jxaFailed(status: process.terminationStatus, stderr: stderrText)
        }

        let output = String(data: stdoutData, encoding: .utf8) ?? ""
        return output.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

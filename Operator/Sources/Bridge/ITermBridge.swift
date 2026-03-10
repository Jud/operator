import Foundation

/// Bridge to iTerm2 via JXA (JavaScript for Automation) for session discovery
/// and file-based text delivery to Claude Code TUI sessions.
///
/// Delivery uses file-based injection to avoid shell + JXA double-escaping:
/// write message to temp file, JXA reads via ObjC Foundation, then executes
/// Escape (clear) -> text -> CR (submit) on the matched session.
public class ITermBridge: @unchecked Sendable, TerminalBridge {
    private static let logger = Log.logger(for: "ITermBridge")

    /// Creates a new iTerm2 bridge instance.
    public init() {}

    /// Write text to an iTerm2 session identified by its TTY path.
    public func writeToSession(identifier: TerminalIdentifier, text: String) async throws -> Bool {
        guard case .tty(let tty) = identifier else {
            throw TerminalBridgeError.sessionNotFound(identifier: identifier.description)
        }
        Self.logger.info("Delivering text to session at TTY: \(tty) (\(text.count) chars)")

        let tempFile: URL
        do {
            tempFile = try OsascriptRunner.writeTempFile(text)
        } catch {
            throw TerminalBridgeError.tempFileWriteFailed(underlying: error)
        }

        defer {
            OsascriptRunner.cleanupTempFile(tempFile)
            Self.logger.debug("Cleaned up temp file: \(tempFile.path)")
        }

        let output = try await runWriteScript(tty: tty, tempFilePath: tempFile.path)

        if output.contains("session_not_found") {
            Self.logger.warning("Session not found for TTY: \(tty)")
            throw TerminalBridgeError.sessionNotFound(identifier: tty)
        }

        let success = output.contains("ok")
        if success {
            Self.logger.info("Successfully delivered text to TTY: \(tty)")
        }
        return success
    }

    private func runWriteScript(tty: String, tempFilePath: String) async throws -> String {
        let script = """
            (function() {
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
                                sess.write({ text: "\\u0015", newline: false });
                                delay(0.1);
                                sess.write({ text: text, newline: false });
                                delay(0.1);
                                sess.write({ text: "\\r", newline: false });
                                return "ok";
                            }
                        }
                    }
                }
                return "session_not_found";
            })();
            """

        return try await OsascriptRunner.runForTerminal(script, terminal: .iterm)
    }

    /// Discover all sessions in running iTerm2 windows.
    public func discoverSessions() async throws -> [DiscoveredSession] {
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

        let output = try await OsascriptRunner.runForTerminal(script, terminal: .iterm)
        let sessions = try OsascriptRunner.decodeDiscoveryJSON(output)
        Self.logger.info("Discovered \(sessions.count) iTerm session(s)")
        return sessions
    }
}

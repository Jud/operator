import Foundation

/// Bridge to Ghostty via JXA (JavaScript for Automation) for session discovery
/// and file-based text delivery to Claude Code TUI sessions.
///
/// Delivery uses file-based injection matching the ITermBridge pattern:
/// write message to temp file, JXA reads via ObjC Foundation, then executes
/// inputText (paste) -> sendKey return (submit) on the matched terminal.
///
/// Ghostty does not expose TTY, isProcessing, or jobName via AppleScript,
/// so those default to empty string, false, and nil respectively.
public class GhosttyBridge: @unchecked Sendable, TerminalBridge {
    private static let logger = Log.logger(for: "GhosttyBridge")

    public init() {}

    public func writeToSession(identifier: TerminalIdentifier, text: String) async throws -> Bool {
        guard case .ghosttyTerminal(let ghosttyId) = identifier else {
            throw TerminalBridgeError.sessionNotFound(identifier: identifier.description)
        }
        Self.logger.info("Delivering text to Ghostty terminal: \(ghosttyId) (\(text.count) chars)")

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

        let output = try await runWriteScript(ghosttyId: ghosttyId, tempFilePath: tempFile.path)

        if output.contains("terminal_not_found") {
            Self.logger.warning("Ghostty terminal not found for ID: \(ghosttyId)")
            throw TerminalBridgeError.sessionNotFound(identifier: ghosttyId)
        }

        let success = output.contains("ok")
        if success {
            Self.logger.info("Successfully delivered text to Ghostty terminal: \(ghosttyId)")
        }
        return success
    }

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

        return try await OsascriptRunner.runForTerminal(script, terminal: .ghostty)
    }

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

        let output = try await OsascriptRunner.runForTerminal(script, terminal: .ghostty)
        let sessions = try OsascriptRunner.decodeDiscoveryJSON(output)
        Self.logger.info("Discovered \(sessions.count) Ghostty session(s)")
        return sessions
    }
}

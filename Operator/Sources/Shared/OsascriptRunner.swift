import Foundation

public enum OsascriptRunnerError: Error, CustomStringConvertible {
    case scriptFailed(status: Int32, stderr: String)
    case applicationNotRunning
    case launchFailed(underlying: Error)

    public var description: String {
        switch self {
        case .scriptFailed(let status, let stderr):
            return "osascript exited with status \(status): \(stderr)"
        case .applicationNotRunning:
            return "Target application is not running"
        case .launchFailed(let underlying):
            return "Failed to launch osascript: \(underlying.localizedDescription)"
        }
    }
}

/// Shared utility for executing JXA scripts via /usr/bin/osascript and
/// managing temp files used for file-based terminal text delivery.
public enum OsascriptRunner {
    /// Execute a JXA (JavaScript for Automation) script without blocking the
    /// cooperative thread pool. Uses `Process.terminationHandler` with
    /// `CheckedContinuation` instead of the blocking `waitUntilExit()`.
    ///
    /// Detects common "not running" patterns in stderr and throws
    /// `.applicationNotRunning` so callers can map to terminal-specific errors.
    public static func run(_ script: String) async throws -> String {
        try await withCheckedThrowingContinuation { continuation in
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
            process.arguments = ["-l", "JavaScript", "-e", script]

            let stdoutPipe = Pipe()
            let stderrPipe = Pipe()
            process.standardOutput = stdoutPipe
            process.standardError = stderrPipe

            process.terminationHandler = { proc in
                // Process has terminated; pipe write ends are closed so reads
                // return immediately. Reading here (on Process's private dispatch
                // queue) keeps the cooperative thread pool unblocked.
                let stdoutData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
                let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()

                guard proc.terminationStatus == 0 else {
                    let stderrText = String(data: stderrData, encoding: .utf8) ?? ""
                    let lower = stderrText.lowercased()
                    if lower.contains("is not running") || lower.contains("not running")
                        || lower.contains("connection is invalid")
                        || lower.contains("application can't be found")
                    {
                        continuation.resume(throwing: OsascriptRunnerError.applicationNotRunning)
                        return
                    }
                    continuation.resume(
                        throwing: OsascriptRunnerError.scriptFailed(
                            status: proc.terminationStatus, stderr: stderrText
                        ))
                    return
                }

                let output = String(data: stdoutData, encoding: .utf8) ?? ""
                continuation.resume(
                    returning: output.trimmingCharacters(in: .whitespacesAndNewlines))
            }

            do {
                try process.run()
            } catch {
                continuation.resume(throwing: OsascriptRunnerError.launchFailed(underlying: error))
            }
        }
    }

    /// Write text to a uniquely-named temp file for file-based JXA delivery.
    public static func writeTempFile(_ text: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("operator-\(UUID().uuidString).txt")
        try text.write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    /// Remove a temp file created by `writeTempFile`. Failures are silently ignored.
    public static func cleanupTempFile(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
    }

    /// Run a JXA script and map `OsascriptRunnerError` to `TerminalBridgeError`
    /// for the given terminal type.
    public static func runForTerminal(
        _ script: String, terminal: TerminalType
    ) async throws -> String {
        do {
            return try await run(script)
        } catch let error as OsascriptRunnerError {
            throw error.asTerminalBridgeError(terminal: terminal)
        }
    }

    /// Decode JXA JSON output into `[DiscoveredSession]`, wrapping errors
    /// as `TerminalBridgeError.discoveryDecodeFailed`.
    public static func decodeDiscoveryJSON(_ output: String) throws -> [DiscoveredSession] {
        guard let data = output.data(using: .utf8) else {
            throw TerminalBridgeError.discoveryDecodeFailed(
                output: output,
                underlying: NSError(
                    domain: "OsascriptRunner", code: -1,
                    userInfo: [NSLocalizedDescriptionKey: "Output is not valid UTF-8"]
                )
            )
        }
        do {
            return try JSONDecoder().decode([DiscoveredSession].self, from: data)
        } catch {
            throw TerminalBridgeError.discoveryDecodeFailed(output: output, underlying: error)
        }
    }
}

extension OsascriptRunnerError {
    func asTerminalBridgeError(terminal: TerminalType) -> TerminalBridgeError {
        switch self {
        case .applicationNotRunning:
            return .terminalNotRunning(terminal: terminal)
        case .scriptFailed(let status, let stderr):
            return .scriptFailed(terminal: terminal, status: status, stderr: stderr)
        case .launchFailed(let underlying):
            return .scriptFailed(
                terminal: terminal, status: -1,
                stderr: "Failed to launch osascript: \(underlying.localizedDescription)"
            )
        }
    }
}

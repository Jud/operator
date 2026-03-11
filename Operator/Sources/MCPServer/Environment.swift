// swiftlint:disable:this file_name
import Foundation

// MARK: - Config

/// Runtime configuration for the MCP server, loaded from environment variables.
public struct Config: Sendable {
    /// Bearer token for daemon API authentication (from OPERATOR_TOKEN).
    public let token: String
    /// Daemon HTTP port (from OPERATOR_PORT, default 7_420).
    public let port: Int
    /// Base URL for daemon HTTP API.
    public var baseURL: String {
        "http://localhost:\(port)"
    }
}

// MARK: - Environment Detection

/// Normalize raw TTY output from `ps` into a canonical device path.
///
/// Prepends `/dev/` if missing. Returns `"unknown"` for nil or empty input.
internal func normalizeTTY(_ rawOutput: String?) -> String {
    guard let raw = rawOutput?.trimmingCharacters(in: .whitespacesAndNewlines),
        !raw.isEmpty
    else {
        return "unknown"
    }
    return raw.hasPrefix("/dev/") ? raw : "/dev/\(raw)"
}

/// Detect the TTY of the parent shell process.
///
/// Shells out to `ps -o tty= -p {ppid}` and normalizes the result
/// to include a `/dev/` prefix. Returns `"unknown"` on failure.
public func detectTTY() -> String {
    let parentPID = getppid()

    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/ps")
    process.arguments = ["-o", "tty=", "-p", "\(parentPID)"]

    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = Pipe()

    do {
        try process.run()
        process.waitUntilExit()

        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let raw = String(data: data, encoding: .utf8)
        return normalizeTTY(raw)
    } catch {
        return "unknown"
    }
}

/// Detect the terminal emulator type from the TERM_PROGRAM environment variable.
///
/// Returns `"ghostty"` when running in Ghostty, `"iterm2"` for all other terminals.
public func detectTerminalType() -> String {
    let termProgram = ProcessInfo.processInfo.environment["TERM_PROGRAM"]
    return termProgram == "ghostty" ? "ghostty" : "iterm2"
}

/// Derive the session name from the basename of the current working directory.
public func deriveSessionName() -> String {
    URL(fileURLWithPath: FileManager.default.currentDirectoryPath).lastPathComponent
}

/// Load runtime configuration from environment variables.
///
/// Reads `OPERATOR_TOKEN` for the bearer token and `OPERATOR_PORT` for the
/// daemon port (defaults to 7_420 if unset or non-numeric). No file I/O is performed.
public func loadConfig() -> Config {
    let env = ProcessInfo.processInfo.environment
    let token = env["OPERATOR_TOKEN"] ?? ""
    let port = env["OPERATOR_PORT"].flatMap(Int.init) ?? 7_420

    return Config(token: token, port: port)
}

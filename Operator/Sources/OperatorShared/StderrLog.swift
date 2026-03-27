import Foundation

/// Minimal stderr logger for CLI and MCP processes.
///
/// All output goes to stderr so stdout remains available for structured
/// output (JSON-RPC transport in MCP mode, JSON results in CLI mode).
public enum StderrLog {
    /// Write a message to stderr with a prefix tag.
    public static func write(_ message: String, tag: String = "Operator") {
        let line = "[\(tag)] \(message)\n"
        if let data = line.data(using: .utf8) {
            FileHandle.standardError.write(data)
        }
    }
}

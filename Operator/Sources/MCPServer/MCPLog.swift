import Foundation

/// Shared stderr logging for the MCP server process.
///
/// All output goes to stderr so stdout remains clean for JSON-RPC transport.
internal enum MCPLog {
    static func write(_ message: String) {
        let line = "[OperatorMCP] \(message)\n"
        if let data = line.data(using: .utf8) {
            FileHandle.standardError.write(data)
        }
    }
}

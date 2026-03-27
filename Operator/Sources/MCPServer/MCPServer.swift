import Foundation
import OperatorShared

/// MCP server core handling JSON-RPC 2.0 stdin/stdout transport and method dispatch.
///
/// Reads newline-delimited JSON-RPC requests from stdin, dispatches to method
/// handlers, and writes JSON-RPC responses as single-line JSON to stdout.
/// All diagnostic output goes to stderr to keep the MCP transport clean.
public struct MCPServer: Sendable {
    /// Daemon HTTP client for speak tool requests.
    public let client: DaemonClient
    /// Session name derived from the CWD basename.
    public let sessionName: String

    private let encoder: JSONEncoder = {
        let enc = JSONEncoder()
        enc.outputFormatting = [.sortedKeys]
        return enc
    }()

    private let decoder = JSONDecoder()

    /// Creates a new MCP server.
    /// - Parameters:
    ///   - client: Daemon HTTP client for speak tool requests.
    ///   - sessionName: Session name derived from the CWD basename.
    public init(client: DaemonClient, sessionName: String) {
        self.client = client
        self.sessionName = sessionName
    }

    /// Run the stdin read loop, processing one JSON-RPC request per line.
    ///
    /// Blocks on `readLine()` in the calling async context. Returns when
    /// stdin reaches EOF (Claude Code exited or closed the pipe).
    public func stdinLoop() async {
        while let line = readLine(strippingNewline: true) {
            guard !line.isEmpty else {
                continue
            }

            guard let data = line.data(using: .utf8) else {
                MCPLog.write("Failed to convert stdin line to UTF-8")
                continue
            }

            let request: JSONRPCRequest
            do {
                request = try decoder.decode(JSONRPCRequest.self, from: data)
            } catch {
                MCPLog.write("Malformed JSON-RPC input: \(error)")
                continue
            }

            if let response = await handleRequest(request) {
                writeResponse(response)
            }
        }
    }

    // MARK: - Method Dispatch

    func handleRequest(_ request: JSONRPCRequest) async -> JSONRPCResponse? {
        switch request.method {
        case "initialize":
            return handleInitialize(id: request.id)

        case "initialized":
            return nil

        case "tools/list":
            return handleToolsList(id: request.id)

        case "tools/call":
            return .methodNotFound(id: request.id, method: "tools/call")

        default:
            return .methodNotFound(id: request.id, method: request.method)
        }
    }

    // MARK: - Initialize

    private func handleInitialize(id: JSONRPCId?) -> JSONRPCResponse {
        let result: JSONValue = .object([
            "protocolVersion": .string("2024-11-05"),
            "serverInfo": .object([
                "name": .string("operator"),
                "version": .string("1.0.0")
            ]),
            "capabilities": .object([
                "tools": .object([:])
            ])
        ])
        return .success(id: id, result: result)
    }

    // MARK: - Tools List

    /// Returns an empty tools array — the MCP server is toolless.
    ///
    /// Speech output is handled by the `operator speak` CLI instead.
    private func handleToolsList(id: JSONRPCId?) -> JSONRPCResponse {
        let result: JSONValue = .object([
            "tools": .array([])
        ])
        return .success(id: id, result: result)
    }

    private func writeResponse(_ response: JSONRPCResponse) {
        do {
            var data = try encoder.encode(response)
            data.append(0x0A)
            FileHandle.standardOutput.write(data)
        } catch {
            MCPLog.write("Failed to encode JSON-RPC response: \(error)")
        }
    }
}

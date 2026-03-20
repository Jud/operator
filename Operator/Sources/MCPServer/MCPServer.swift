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
            return await handleToolsCall(id: request.id, params: request.params)

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

    private func handleToolsList(id: JSONRPCId?) -> JSONRPCResponse {
        let inputSchema: JSONValue = .object([
            "type": .string("object"),
            "properties": .object([
                "message": .object([
                    "type": .string("string"),
                    "description": .string("Message to speak")
                ]),
                "priority": .object([
                    "type": .string("string"),
                    "enum": .array([.string("normal"), .string("urgent")]),
                    "default": .string("normal"),
                    "description": .string(
                        "Priority level. Urgent messages skip to front of queue."
                    )
                ])
            ]),
            "required": .array([.string("message")])
        ])

        let tool: JSONValue = .object([
            "name": .string("speak"),
            "description": .string(
                "Send a short voice status update to the user. Call at end of every turn and at key milestones."
                    + " One sentence max — like a walkie-talkie, not a presentation."
            ),
            "inputSchema": inputSchema
        ])

        let result: JSONValue = .object([
            "tools": .array([tool])
        ])
        return .success(id: id, result: result)
    }

    // MARK: - Tools Call

    private func handleToolsCall(
        id: JSONRPCId?,
        params: [String: JSONValue]?
    ) async -> JSONRPCResponse {
        guard let name = params?["name"]?.stringValue else {
            return .error(id: id, code: -32_602, message: "Missing tool name")
        }

        guard name == "speak" else {
            return .error(id: id, code: -32_602, message: "Unknown tool: \(name)")
        }

        let arguments = params?["arguments"]?.objectValue ?? [:]
        return await handleSpeak(id: id, arguments: arguments)
    }

    private func handleSpeak(
        id: JSONRPCId?,
        arguments: [String: JSONValue]
    ) async -> JSONRPCResponse {
        let message = arguments["message"]?.stringValue ?? ""
        let priority = arguments["priority"]?.stringValue ?? "normal"

        let speakRequest = SpeakRequest(
            message: message,
            session: sessionName,
            priority: priority
        )

        guard await client.postRaw(path: "/speak", body: speakRequest) != nil else {
            let errorContent = toolResultContent(
                text: "Operator daemon unreachable",
                isError: true
            )
            return .success(id: id, result: errorContent)
        }

        // Return empty content to avoid triggering follow-up from the agent.
        let content = toolResultContent(text: "", isError: false)
        return .success(id: id, result: content)
    }

    // MARK: - Helpers

    private func toolResultContent(text: String, isError: Bool) -> JSONValue {
        var result: [String: JSONValue] = [
            "content": .array([
                .object([
                    "type": .string("text"),
                    "text": .string(text)
                ])
            ])
        ]
        if isError {
            result["isError"] = .bool(true)
        }
        return .object(result)
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

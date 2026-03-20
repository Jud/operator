import Foundation
import Testing

@testable import OperatorMCPCore

// MARK: - Protocol Handshake + Tools List

@Suite("MCPServer - Protocol and Tools List")
internal struct MCPServerProtocolTests {
    private func makeServer() -> MCPServer {
        let client = DaemonClient(baseURL: "http://localhost:0", token: "test-token")
        return MCPServer(client: client, sessionName: "test-session")
    }

    @Test("initialize returns server info with name and version")
    func initializeResponse() async {
        let server = makeServer()
        let request = JSONRPCRequest(
            jsonrpc: "2.0",
            method: "initialize",
            params: nil,
            id: .int(1)
        )

        let response = await server.handleRequest(request)
        #expect(response != nil)

        guard case .object(let result) = response?.result else {
            Issue.record("Expected object result")
            return
        }

        guard case .object(let serverInfo) = result["serverInfo"] else {
            Issue.record("Expected serverInfo object")
            return
        }
        #expect(serverInfo["name"] == .string("operator"))
        #expect(serverInfo["version"] == .string("1.0.0"))
    }

    @Test("initialize includes tools capability")
    func initializeCapabilities() async {
        let server = makeServer()
        let request = JSONRPCRequest(
            jsonrpc: "2.0",
            method: "initialize",
            params: nil,
            id: .int(1)
        )

        let response = await server.handleRequest(request)

        guard case .object(let result) = response?.result else {
            Issue.record("Expected object result")
            return
        }

        guard case .object(let capabilities) = result["capabilities"] else {
            Issue.record("Expected capabilities object")
            return
        }
        guard case .object = capabilities["tools"] else {
            Issue.record("Expected tools capability")
            return
        }
    }

    @Test("initialize includes protocol version")
    func initializeProtocolVersion() async {
        let server = makeServer()
        let request = JSONRPCRequest(
            jsonrpc: "2.0",
            method: "initialize",
            params: nil,
            id: .int(1)
        )

        let response = await server.handleRequest(request)

        guard case .object(let result) = response?.result else {
            Issue.record("Expected object result")
            return
        }
        #expect(result["protocolVersion"] == .string("2024-11-05"))
    }

    @Test("initialized notification returns nil response")
    func initializedNotification() async {
        let server = makeServer()
        let request = JSONRPCRequest(
            jsonrpc: "2.0",
            method: "initialized",
            params: nil,
            id: nil
        )

        let response = await server.handleRequest(request)
        #expect(response == nil)
    }

    @Test("tools/list returns exactly one tool named speak")
    func toolsListSpeakTool() async {
        let server = makeServer()
        let request = JSONRPCRequest(
            jsonrpc: "2.0",
            method: "tools/list",
            params: nil,
            id: .int(2)
        )

        let response = await server.handleRequest(request)

        guard case .object(let result) = response?.result else {
            Issue.record("Expected object result")
            return
        }

        guard case .array(let tools) = result["tools"] else {
            Issue.record("Expected tools array")
            return
        }
        #expect(tools.count == 1)

        guard case .object(let tool) = tools.first else {
            Issue.record("Expected tool object")
            return
        }
        #expect(tool["name"] == .string("speak"))
    }

    @Test("tools/list speak tool has correct description")
    func toolsListDescription() async {
        let server = makeServer()
        let request = JSONRPCRequest(
            jsonrpc: "2.0",
            method: "tools/list",
            params: nil,
            id: .int(2)
        )

        let response = await server.handleRequest(request)

        guard case .object(let result) = response?.result,
            case .array(let tools) = result["tools"],
            case .object(let tool) = tools.first
        else {
            Issue.record("Expected tool object")
            return
        }

        let expectedDesc =
            "Send a short voice status update to the user. Call at end of every turn and at key milestones."
            + " One sentence max — like a walkie-talkie, not a presentation."
        #expect(tool["description"] == .string(expectedDesc))
    }

    @Test("tools/list input schema defines message as required string")
    func toolsListMessageSchema() async {
        let server = makeServer()
        let request = JSONRPCRequest(
            jsonrpc: "2.0",
            method: "tools/list",
            params: nil,
            id: .int(2)
        )

        let response = await server.handleRequest(request)

        guard case .object(let result) = response?.result,
            case .array(let tools) = result["tools"],
            case .object(let tool) = tools.first,
            case .object(let schema) = tool["inputSchema"],
            case .object(let properties) = schema["properties"],
            case .object(let messageProp) = properties["message"],
            case .array(let required) = schema["required"]
        else {
            Issue.record("Expected input schema with message property")
            return
        }

        #expect(messageProp["type"] == .string("string"))
        #expect(required.contains(.string("message")))
    }

    @Test("tools/list input schema defines priority with enum and default")
    func toolsListPrioritySchema() async {
        let server = makeServer()
        let request = JSONRPCRequest(
            jsonrpc: "2.0",
            method: "tools/list",
            params: nil,
            id: .int(2)
        )

        let response = await server.handleRequest(request)

        guard case .object(let result) = response?.result,
            case .array(let tools) = result["tools"],
            case .object(let tool) = tools.first,
            case .object(let schema) = tool["inputSchema"],
            case .object(let properties) = schema["properties"],
            case .object(let priorityProp) = properties["priority"]
        else {
            Issue.record("Expected input schema with priority property")
            return
        }

        #expect(priorityProp["type"] == .string("string"))
        #expect(priorityProp["default"] == .string("normal"))
        #expect(priorityProp["enum"] == .array([.string("normal"), .string("urgent")]))
    }
}

// MARK: - Tools Call + Error Handling

@Suite("MCPServer - Tools Call and Errors")
internal struct MCPServerToolsCallTests {
    private func makeServer() -> MCPServer {
        let client = DaemonClient(baseURL: "http://localhost:0", token: "test-token")
        return MCPServer(client: client, sessionName: "test-session")
    }

    @Test("tools/call with missing tool name returns error")
    func toolsCallMissingName() async {
        let server = makeServer()
        let request = JSONRPCRequest(
            jsonrpc: "2.0",
            method: "tools/call",
            params: [:],
            id: .int(3)
        )

        let response = await server.handleRequest(request)
        #expect(response?.error?.code == -32_602)
        #expect(response?.error?.message == "Missing tool name")
    }

    @Test("tools/call with unknown tool name returns error")
    func toolsCallUnknownTool() async {
        let server = makeServer()
        let request = JSONRPCRequest(
            jsonrpc: "2.0",
            method: "tools/call",
            params: ["name": .string("nonexistent")],
            id: .int(3)
        )

        let response = await server.handleRequest(request)
        #expect(response?.error?.code == -32_602)
        #expect(response?.error?.message == "Unknown tool: nonexistent")
    }

    @Test("tools/call for speak with unreachable daemon returns error content block")
    func toolsCallSpeakDaemonUnreachable() async {
        let server = makeServer()
        let request = JSONRPCRequest(
            jsonrpc: "2.0",
            method: "tools/call",
            params: [
                "name": .string("speak"),
                "arguments": .object([
                    "message": .string("hello"),
                    "priority": .string("normal")
                ])
            ],
            id: .int(4)
        )

        let response = await server.handleRequest(request)
        #expect(response != nil)
        #expect(response?.error == nil)

        guard case .object(let result) = response?.result else {
            Issue.record("Expected object result")
            return
        }

        #expect(result["isError"] == .bool(true))

        guard case .array(let content) = result["content"],
            case .object(let block) = content.first
        else {
            Issue.record("Expected content array with text block")
            return
        }
        #expect(block["type"] == .string("text"))
        #expect(block["text"] == .string("Operator daemon unreachable"))
    }

    @Test("unknown method returns -32601 error")
    func unknownMethodError() async {
        let server = makeServer()
        let request = JSONRPCRequest(
            jsonrpc: "2.0",
            method: "rpc/nonexistent",
            params: nil,
            id: .int(99)
        )

        let response = await server.handleRequest(request)
        #expect(response?.error?.code == -32_601)
        #expect(response?.error?.message == "Method not found: rpc/nonexistent")
    }

    @Test("response preserves request id")
    func responsePreservesId() async {
        let server = makeServer()
        let request = JSONRPCRequest(
            jsonrpc: "2.0",
            method: "initialize",
            params: nil,
            id: .string("req-abc")
        )

        let response = await server.handleRequest(request)
        #expect(response?.id == .string("req-abc"))
    }
}

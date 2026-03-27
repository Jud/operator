import Foundation
import OperatorShared
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

    @Test("tools/list returns empty tools array (toolless MCP)")
    func toolsListEmpty() async {
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
        #expect(tools.isEmpty)
    }
}

// MARK: - Error Handling

@Suite("MCPServer - Errors")
internal struct MCPServerErrorTests {
    private func makeServer() -> MCPServer {
        let client = DaemonClient(baseURL: "http://localhost:0", token: "test-token")
        return MCPServer(client: client, sessionName: "test-session")
    }

    @Test("tools/call returns method-not-found (toolless MCP)")
    func toolsCallMethodNotFound() async {
        let server = makeServer()
        let request = JSONRPCRequest(
            jsonrpc: "2.0",
            method: "tools/call",
            params: ["name": .string("speak")],
            id: .int(3)
        )

        let response = await server.handleRequest(request)
        #expect(response?.error?.code == -32_601)
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

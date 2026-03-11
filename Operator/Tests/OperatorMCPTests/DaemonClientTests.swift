import Foundation
import OperatorShared
import Testing

@testable import OperatorMCPCore

@Suite("DaemonClient - Shared Type Encoding")
internal struct DaemonClientTests {
    private let encoder: JSONEncoder = {
        let enc = JSONEncoder()
        enc.outputFormatting = [.sortedKeys]
        return enc
    }()

    // MARK: - SpeakRequest

    @Test("SpeakRequest encodes with camelCase keys")
    func speakRequestEncoding() throws {
        let request = SpeakRequest(message: "hello", session: "myapp", priority: "urgent")
        let data = try encoder.encode(request)
        let json = try JSONDecoder().decode([String: String].self, from: data)

        #expect(json["message"] == "hello")
        #expect(json["session"] == "myapp")
        #expect(json["priority"] == "urgent")
    }

    @Test("SpeakRequest encodes optional fields as null when nil")
    func speakRequestOptionalFields() throws {
        let request = SpeakRequest(message: "hello")
        let data = try encoder.encode(request)
        let str = try #require(String(data: data, encoding: .utf8))

        #expect(str.contains("\"message\":\"hello\""))
    }

    // MARK: - HookSessionStartRequest

    @Test("HookSessionStartRequest encodes with snake_case keys")
    func hookSessionStartRequestEncoding() throws {
        let request = HookSessionStartRequest(
            sessionId: "myapp",
            tty: "/dev/ttys004",
            cwd: "/Users/jud/Projects/myapp",
            terminalType: "iterm2"
        )
        let data = try encoder.encode(request)
        let json = try JSONDecoder().decode(
            HookSessionStartRequest.self,
            from: data
        )
        #expect(json.sessionId == "myapp")
        #expect(json.terminalType == "iterm2")
        #expect(json.tty == "/dev/ttys004")
        #expect(json.cwd == "/Users/jud/Projects/myapp")

        let str = try #require(String(data: data, encoding: .utf8))
        #expect(str.contains("\"session_id\""))
        #expect(str.contains("\"terminal_type\""))
    }

    @Test("HookSessionStartRequest does not use camelCase session_id key")
    func hookSessionStartRequestNoSessionId() throws {
        let request = HookSessionStartRequest(
            sessionId: "test",
            tty: "/dev/ttys000",
            cwd: "/tmp",
            terminalType: "iterm2"
        )
        let data = try encoder.encode(request)
        let str = try #require(String(data: data, encoding: .utf8))

        #expect(!str.contains("\"sessionId\""))
        #expect(!str.contains("\"terminalType\""))
    }

    // MARK: - HookTerminalIdRequest

    @Test("HookTerminalIdRequest encodes with snake_case ghostty_id key")
    func hookTerminalIdRequestEncoding() throws {
        let request = HookTerminalIdRequest(
            tty: "/dev/ttys004",
            ghosttyId: "term-abc-123"
        )
        let data = try encoder.encode(request)
        let json = try JSONDecoder().decode(
            HookTerminalIdRequest.self,
            from: data
        )
        #expect(json.tty == "/dev/ttys004")
        #expect(json.ghosttyId == "term-abc-123")

        let str = try #require(String(data: data, encoding: .utf8))
        #expect(str.contains("\"ghostty_id\""))
        #expect(!str.contains("\"ghosttyId\""))
    }

    // MARK: - HookSessionStartResponse

    @Test("HookSessionStartResponse decodes snake_case needs_terminal_id")
    func hookSessionStartResponseDecoding() throws {
        let json = """
            {"ok":true,"needs_terminal_id":true}
            """
        let response = try JSONDecoder().decode(
            HookSessionStartResponse.self,
            from: Data(json.utf8)
        )
        #expect(response.ok == true)
        #expect(response.needsTerminalId == true)
    }

    @Test("HookSessionStartResponse decodes when needs_terminal_id is false")
    func hookSessionStartResponseNeedsIdFalse() throws {
        let json = """
            {"ok":true,"needs_terminal_id":false}
            """
        let response = try JSONDecoder().decode(
            HookSessionStartResponse.self,
            from: Data(json.utf8)
        )
        #expect(response.ok == true)
        #expect(response.needsTerminalId == false)
    }

    // MARK: - Round-trip

    @Test("HookSessionStartRequest round-trips through encode/decode")
    func hookSessionStartRoundTrip() throws {
        let original = HookSessionStartRequest(
            sessionId: "myapp",
            tty: "/dev/ttys004",
            cwd: "/Users/jud/Projects/myapp",
            terminalType: "ghostty"
        )
        let data = try encoder.encode(original)
        let decoded = try JSONDecoder().decode(HookSessionStartRequest.self, from: data)

        #expect(decoded.sessionId == original.sessionId)
        #expect(decoded.tty == original.tty)
        #expect(decoded.cwd == original.cwd)
        #expect(decoded.terminalType == original.terminalType)
    }
}

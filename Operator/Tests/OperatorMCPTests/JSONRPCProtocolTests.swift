import Foundation
import Testing

@testable import OperatorMCPCore

// MARK: - JSONValue Tests

@Suite("JSONValue")
internal struct JSONValueTests {
    private let encoder: JSONEncoder = {
        let enc = JSONEncoder()
        enc.outputFormatting = [.sortedKeys]
        return enc
    }()

    private let decoder = JSONDecoder()

    @Test("string round-trips through JSON")
    func stringRoundTrip() throws {
        let original = JSONValue.string("hello")
        let data = try encoder.encode(original)
        let decoded = try decoder.decode(JSONValue.self, from: data)
        #expect(decoded == original)
    }

    @Test("int round-trips through JSON")
    func intRoundTrip() throws {
        let original = JSONValue.int(42)
        let data = try encoder.encode(original)
        let decoded = try decoder.decode(JSONValue.self, from: data)
        #expect(decoded == original)
    }

    @Test("double round-trips through JSON")
    func doubleRoundTrip() throws {
        let original = JSONValue.double(3.14)
        let data = try encoder.encode(original)
        let decoded = try decoder.decode(JSONValue.self, from: data)
        #expect(decoded == original)
    }

    @Test("bool true round-trips through JSON")
    func boolTrueRoundTrip() throws {
        let original = JSONValue.bool(true)
        let data = try encoder.encode(original)
        let decoded = try decoder.decode(JSONValue.self, from: data)
        #expect(decoded == original)
    }

    @Test("bool false round-trips through JSON")
    func boolFalseRoundTrip() throws {
        let original = JSONValue.bool(false)
        let data = try encoder.encode(original)
        let decoded = try decoder.decode(JSONValue.self, from: data)
        #expect(decoded == original)
    }

    @Test("null round-trips through JSON")
    func nullRoundTrip() throws {
        let original = JSONValue.null
        let data = try encoder.encode(original)
        let decoded = try decoder.decode(JSONValue.self, from: data)
        #expect(decoded == original)
    }

    @Test("array round-trips through JSON")
    func arrayRoundTrip() throws {
        let original = JSONValue.array([.string("a"), .int(1), .bool(true)])
        let data = try encoder.encode(original)
        let decoded = try decoder.decode(JSONValue.self, from: data)
        #expect(decoded == original)
    }

    @Test("object round-trips through JSON")
    func objectRoundTrip() throws {
        let original = JSONValue.object(["key": .string("value"), "count": .int(5)])
        let data = try encoder.encode(original)
        let decoded = try decoder.decode(JSONValue.self, from: data)
        #expect(decoded == original)
    }

    @Test("nested object round-trips through JSON")
    func nestedObjectRoundTrip() throws {
        let original = JSONValue.object([
            "outer": .object([
                "inner": .array([.int(1), .null, .string("x")])
            ])
        ])
        let data = try encoder.encode(original)
        let decoded = try decoder.decode(JSONValue.self, from: data)
        #expect(decoded == original)
    }

    @Test("bool is not decoded as int")
    func boolNotDecodedAsInt() throws {
        let json = Data("true".utf8)
        let decoded = try decoder.decode(JSONValue.self, from: json)
        #expect(decoded == .bool(true))
    }

    @Test("stringValue accessor returns string for string case")
    func stringValueAccessor() {
        let value = JSONValue.string("hello")
        #expect(value.stringValue == "hello")
    }

    @Test("stringValue accessor returns nil for non-string case")
    func stringValueNilForNonString() {
        let value = JSONValue.int(42)
        #expect(value.stringValue == nil)
    }

    @Test("objectValue accessor returns dict for object case")
    func objectValueAccessor() {
        let value = JSONValue.object(["k": .string("v")])
        #expect(value.objectValue?["k"] == .string("v"))
    }

    @Test("objectValue accessor returns nil for non-object case")
    func objectValueNilForNonObject() {
        let value = JSONValue.string("not an object")
        #expect(value.objectValue == nil)
    }
}

// MARK: - JSONRPCId Tests

@Suite("JSONRPCId")
internal struct JSONRPCIdTests {
    private let decoder = JSONDecoder()
    private let encoder = JSONEncoder()

    @Test("decodes integer id")
    func decodeIntId() throws {
        let json = Data("42".utf8)
        let id = try decoder.decode(JSONRPCId.self, from: json)
        #expect(id == .int(42))
    }

    @Test("decodes string id")
    func decodeStringId() throws {
        let json = Data("\"abc-123\"".utf8)
        let id = try decoder.decode(JSONRPCId.self, from: json)
        #expect(id == .string("abc-123"))
    }

    @Test("encodes integer id")
    func encodeIntId() throws {
        let data = try encoder.encode(JSONRPCId.int(7))
        let str = String(data: data, encoding: .utf8)
        #expect(str == "7")
    }

    @Test("encodes string id")
    func encodeStringId() throws {
        let data = try encoder.encode(JSONRPCId.string("req-1"))
        let str = String(data: data, encoding: .utf8)
        #expect(str == "\"req-1\"")
    }
}

// MARK: - JSONRPCRequest Tests

@Suite("JSONRPCRequest")
internal struct JSONRPCRequestTests {
    private let decoder = JSONDecoder()

    @Test("parses valid request with int id")
    func parseValidRequestIntId() throws {
        let json = """
            {"jsonrpc":"2.0","method":"initialize","params":{"key":"val"},"id":1}
            """
        let request = try decoder.decode(JSONRPCRequest.self, from: Data(json.utf8))
        #expect(request.jsonrpc == "2.0")
        #expect(request.method == "initialize")
        #expect(request.id == .int(1))
        #expect(request.params?["key"] == .string("val"))
    }

    @Test("parses valid request with string id")
    func parseValidRequestStringId() throws {
        let json = """
            {"jsonrpc":"2.0","method":"tools/list","id":"abc"}
            """
        let request = try decoder.decode(JSONRPCRequest.self, from: Data(json.utf8))
        #expect(request.id == .string("abc"))
        #expect(request.method == "tools/list")
    }

    @Test("parses notification without id")
    func parseNotification() throws {
        let json = """
            {"jsonrpc":"2.0","method":"initialized"}
            """
        let request = try decoder.decode(JSONRPCRequest.self, from: Data(json.utf8))
        #expect(request.method == "initialized")
        #expect(request.id == nil)
        #expect(request.params == nil)
    }

    @Test("malformed JSON throws")
    func malformedJSONThrows() {
        let json = Data("{not valid json".utf8)
        #expect(throws: (any Error).self) {
            try decoder.decode(JSONRPCRequest.self, from: json)
        }
    }

    @Test("missing method field throws")
    func missingMethodThrows() {
        let json = Data("{\"jsonrpc\":\"2.0\",\"id\":1}".utf8)
        #expect(throws: (any Error).self) {
            try decoder.decode(JSONRPCRequest.self, from: json)
        }
    }
}

// MARK: - JSONRPCResponse Tests

@Suite("JSONRPCResponse")
internal struct JSONRPCResponseTests {
    private let encoder: JSONEncoder = {
        let enc = JSONEncoder()
        enc.outputFormatting = [.sortedKeys]
        return enc
    }()

    @Test("success response encodes with result and jsonrpc 2.0")
    func successResponseEncoding() throws {
        let response = JSONRPCResponse.success(
            id: .int(1),
            result: .object(["status": .string("ok")])
        )
        let data = try encoder.encode(response)
        let json = try JSONDecoder().decode([String: JSONValue].self, from: data)

        #expect(json["jsonrpc"] == .string("2.0"))
        #expect(json["id"] == .int(1))

        guard case .object(let resultObj) = json["result"] else {
            Issue.record("Expected object result")
            return
        }
        #expect(resultObj["status"] == .string("ok"))
    }

    @Test("error response encodes with error object")
    func errorResponseEncoding() throws {
        let response = JSONRPCResponse.error(
            id: .int(2),
            code: -32_601,
            message: "Method not found: foo"
        )
        let data = try encoder.encode(response)
        let json = try JSONDecoder().decode([String: JSONValue].self, from: data)

        #expect(json["jsonrpc"] == .string("2.0"))
        #expect(json["id"] == .int(2))

        guard case .object(let errorObj) = json["error"] else {
            Issue.record("Expected error object")
            return
        }
        #expect(errorObj["code"] == .int(-32_601))
        #expect(errorObj["message"] == .string("Method not found: foo"))
    }

    @Test("methodNotFound factory produces correct code and message")
    func methodNotFoundFactory() throws {
        let response = JSONRPCResponse.methodNotFound(id: .int(5), method: "rpc/unknown")
        let data = try encoder.encode(response)
        let json = try JSONDecoder().decode([String: JSONValue].self, from: data)

        guard case .object(let errorObj) = json["error"] else {
            Issue.record("Expected error object")
            return
        }
        #expect(errorObj["code"] == .int(-32_601))
        #expect(errorObj["message"] == .string("Method not found: rpc/unknown"))
    }

    @Test("response with nil result encodes null result")
    func nilResultEncodesNull() throws {
        let response = JSONRPCResponse(id: .int(3), result: nil, error: nil)
        let data = try encoder.encode(response)
        let str = try #require(String(data: data, encoding: .utf8))
        #expect(str.contains("\"result\":null"))
    }

    @Test("response encodes as single line without newlines")
    func singleLineEncoding() throws {
        let response = JSONRPCResponse.success(
            id: .int(1),
            result: .object(["nested": .object(["deep": .string("value")])])
        )
        let data = try encoder.encode(response)
        let str = try #require(String(data: data, encoding: .utf8))
        #expect(!str.contains("\n"))
    }
}

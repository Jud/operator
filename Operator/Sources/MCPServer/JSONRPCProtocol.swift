import Foundation

// MARK: - JSONValue

/// Type-safe representation of an arbitrary JSON value.
///
/// Used for flexible JSON-RPC parameter and result handling without
/// third-party dependencies. Supports full round-trip Codable fidelity.
internal enum JSONValue: Sendable, Equatable {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)
    case null
    case array([Self])
    case object([String: Self])
}

extension JSONValue: Codable {
    internal init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()

        if container.decodeNil() {
            self = .null
            return
        }

        // Bool must come before Int -- Foundation decodes true/false as Int(1/0) otherwise.
        if let value = try? container.decode(Bool.self) {
            self = .bool(value)
            return
        }
        if let value = try? container.decode(Int.self) {
            self = .int(value)
            return
        }
        if let value = try? container.decode(Double.self) {
            self = .double(value)
            return
        }
        if let value = try? container.decode(String.self) {
            self = .string(value)
            return
        }
        if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
            return
        }
        if let value = try? container.decode([String: JSONValue].self) {
            self = .object(value)
            return
        }

        throw DecodingError.typeMismatch(
            JSONValue.self,
            DecodingError.Context(
                codingPath: decoder.codingPath,
                debugDescription: "Unable to decode JSON value"
            )
        )
    }

    internal func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value): try container.encode(value)
        case .int(let value): try container.encode(value)
        case .double(let value): try container.encode(value)
        case .bool(let value): try container.encode(value)
        case .null: try container.encodeNil()
        case .array(let value): try container.encode(value)
        case .object(let value): try container.encode(value)
        }
    }
}

extension JSONValue {
    /// Access a string value, or nil if not a string.
    var stringValue: String? {
        if case .string(let value) = self {
            return value
        }
        return nil
    }

    /// Access an object dictionary, or nil if not an object.
    var objectValue: [String: JSONValue]? {
        if case .object(let value) = self {
            return value
        }
        return nil
    }
}

// MARK: - JSONRPCId

/// JSON-RPC 2.0 request/response identifier.
///
/// Per the spec, id may be a string or integer. Notifications omit the id field entirely.
internal enum JSONRPCId: Sendable, Equatable {
    case string(String)
    case int(Int)
}

extension JSONRPCId: Codable {
    internal init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(Int.self) {
            self = .int(value)
            return
        }
        if let value = try? container.decode(String.self) {
            self = .string(value)
            return
        }
        throw DecodingError.typeMismatch(
            JSONRPCId.self,
            DecodingError.Context(
                codingPath: decoder.codingPath,
                debugDescription: "JSON-RPC id must be a string or integer"
            )
        )
    }

    internal func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value): try container.encode(value)
        case .int(let value): try container.encode(value)
        }
    }
}

// MARK: - JSONRPCError

/// JSON-RPC 2.0 error object.
internal struct JSONRPCError: Codable, Sendable, Equatable {
    let code: Int
    let message: String
}

// MARK: - JSONRPCRequest

/// An incoming JSON-RPC 2.0 request or notification.
///
/// Notifications have a nil `id` and must not receive a response.
internal struct JSONRPCRequest: Sendable {
    let jsonrpc: String
    let method: String
    let params: [String: JSONValue]?
    let id: JSONRPCId?
}

extension JSONRPCRequest: Decodable {
    private enum CodingKeys: String, CodingKey {
        case jsonrpc, method, params, id
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        jsonrpc = try container.decode(String.self, forKey: .jsonrpc)
        method = try container.decode(String.self, forKey: .method)
        params = try container.decodeIfPresent([String: JSONValue].self, forKey: .params)
        id = try container.decodeIfPresent(JSONRPCId.self, forKey: .id)
    }
}

// MARK: - JSONRPCResponse

/// An outgoing JSON-RPC 2.0 response.
///
/// Contains either a `result` or an `error`, never both.
/// The `id` matches the request that produced this response.
internal struct JSONRPCResponse: Sendable {
    let id: JSONRPCId?
    let result: JSONValue?
    let error: JSONRPCError?
}

extension JSONRPCResponse: Encodable {
    private enum CodingKeys: String, CodingKey {
        case jsonrpc, result, error, id
    }

    internal func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode("2.0", forKey: .jsonrpc)

        if let error {
            try container.encode(error, forKey: .error)
        } else if let result {
            try container.encode(result, forKey: .result)
        } else {
            try container.encodeNil(forKey: .result)
        }

        if let id {
            try container.encode(id, forKey: .id)
        } else {
            try container.encodeNil(forKey: .id)
        }
    }
}

extension JSONRPCResponse {
    /// Create a success response with the given result.
    static func success(id: JSONRPCId?, result: JSONValue) -> Self {
        Self(id: id, result: result, error: nil)
    }

    /// Create an error response.
    static func error(id: JSONRPCId?, code: Int, message: String) -> Self {
        Self(id: id, result: nil, error: JSONRPCError(code: code, message: message))
    }

    /// Create a Method Not Found (-32601) error response.
    static func methodNotFound(id: JSONRPCId?, method: String) -> Self {
        .error(id: id, code: -32_601, message: "Method not found: \(method)")
    }
}

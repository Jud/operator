import Foundation
import Testing

@testable import OperatorCore

@Suite("ClaudePipe.extractJSON - Adversarial Inputs")
internal struct ExtractJSONFuzzTests {
    // MARK: - Unclosed / mismatched braces

    @Test("unclosed opening brace returns nil")
    func unclosedBrace() {
        let result = ClaudePipe.extractJSON(from: #"{"session": "sudo""#)
        // If it returns something, it should at least be valid JSON
        if let json = result {
            let valid = try? JSONSerialization.jsonObject(with: Data(json.utf8))
            #expect(valid != nil, "extractJSON returned invalid JSON: \(json)")
        }
    }

    @Test("extra closing brace doesn't corrupt extraction")
    func extraClosingBrace() {
        let result = ClaudePipe.extractJSON(from: #"{"a": 1}}"#)
        #expect(result == #"{"a": 1}"#)
    }

    @Test("only closing braces returns nil or valid JSON")
    func onlyClosingBraces() {
        let result = ClaudePipe.extractJSON(from: "}}}")
        if let json = result {
            let valid = try? JSONSerialization.jsonObject(with: Data(json.utf8))
            #expect(valid != nil, "extractJSON returned invalid JSON: \(json)")
        }
    }

    @Test("deeply nested braces extracts valid JSON")
    func deeplyNested() {
        let json = #"{"a":{"b":{"c":{"d":{"e":1}}}}}"#
        let result = ClaudePipe.extractJSON(from: "preamble \(json) trailing")
        #expect(result == json)
        guard let extracted = result else {
            Issue.record("Expected non-nil result")
            return
        }
        let parsed = try? JSONSerialization.jsonObject(with: Data(extracted.utf8))
        #expect(parsed != nil)
    }

    // MARK: - String edge cases

    @Test("JSON string containing escaped quotes")
    func escapedQuotesInString() {
        let input = #"{"msg": "he said \"hello\" then \"bye\""}"#
        let result = ClaudePipe.extractJSON(from: input)
        guard let extracted = result else {
            Issue.record("Expected non-nil result")
            return
        }
        let parsed = try? JSONSerialization.jsonObject(with: Data(extracted.utf8))
        #expect(parsed != nil, "Result should be valid JSON: \(extracted)")
    }

    @Test("JSON string containing braces inside strings")
    func bracesInsideStrings() {
        let input = #"{"code": "if (x) { return }"}"#
        let result = ClaudePipe.extractJSON(from: input)
        #expect(result == input)
        guard let extracted = result else {
            Issue.record("Expected non-nil result")
            return
        }
        let parsed = try? JSONSerialization.jsonObject(with: Data(extracted.utf8))
        #expect(parsed != nil)
    }

    @Test("JSON string containing backslash at end of string")
    func backslashAtEndOfString() {
        let input = #"{"path": "C:\\Users\\test"}"#
        let result = ClaudePipe.extractJSON(from: input)
        guard let extracted = result else {
            Issue.record("Expected non-nil result")
            return
        }
        let parsed = try? JSONSerialization.jsonObject(with: Data(extracted.utf8))
        #expect(parsed != nil, "Result should be valid JSON: \(extracted)")
    }

    @Test("JSON with unicode escape sequences")
    func unicodeEscapes() {
        let input = #"{"emoji": "\u0048ello"}"#
        let result = ClaudePipe.extractJSON(from: input)
        guard let extracted = result else {
            Issue.record("Expected non-nil result")
            return
        }
        let parsed = try? JSONSerialization.jsonObject(with: Data(extracted.utf8))
        #expect(parsed != nil)
    }

    @Test("string containing escaped backslash then quote")
    func escapedBackslashThenQuote() {
        // This is the tricky case: \\" in JSON means escaped backslash followed by end-of-string
        let input = #"{"val": "end\\"}"#
        let result = ClaudePipe.extractJSON(from: input)
        if let json = result {
            let parsed = try? JSONSerialization.jsonObject(with: Data(json.utf8))
            #expect(parsed != nil, "extractJSON returned invalid JSON: \(json)")
        }
    }

    // MARK: - Preamble / noise variations

    @Test("markdown code fence around JSON")
    func markdownCodeFence() {
        let input = """
            Here's the result:
            ```json
            {"session": "sudo", "confident": true}
            ```
            """
        let result = ClaudePipe.extractJSON(from: input)
        #expect(result == #"{"session": "sudo", "confident": true}"#)
    }

    @Test("multiple JSON objects returns only the first")
    func multipleJsonObjects() {
        let input = #"{"first": 1} {"second": 2}"#
        let result = ClaudePipe.extractJSON(from: input)
        #expect(result == #"{"first": 1}"#)
    }

    @Test("JSON preceded by a lot of preamble text")
    func longPreamble() {
        let preamble = String(repeating: "blah blah blah ", count: 100)
        let json = #"{"session": "sudo"}"#
        let result = ClaudePipe.extractJSON(from: preamble + json)
        #expect(result == json)
    }

    @Test("JSON array is not extracted (only objects)")
    func jsonArrayNotExtracted() {
        let result = ClaudePipe.extractJSON(from: #"[1, 2, 3]"#)
        #expect(result == nil)
    }

    // MARK: - Empty / minimal inputs

    @Test("empty object")
    func emptyObject() {
        let result = ClaudePipe.extractJSON(from: "{}")
        #expect(result == "{}")
    }

    @Test("whitespace only")
    func whitespaceOnly() {
        let result = ClaudePipe.extractJSON(from: "   \n\t  ")
        #expect(result == nil)
    }

    @Test("single character inputs")
    func singleChars() {
        #expect(ClaudePipe.extractJSON(from: "{") == nil)
        #expect(ClaudePipe.extractJSON(from: "}") == nil)
        #expect(ClaudePipe.extractJSON(from: "\"") == nil)
        #expect(ClaudePipe.extractJSON(from: "a") == nil)
    }

    // MARK: - Malicious / adversarial

    @Test("extremely long string value doesn't hang")
    func longStringValue() {
        let longVal = String(repeating: "a", count: 10_000)
        let input = #"{"key": ""# + longVal + #""}"#
        let result = ClaudePipe.extractJSON(from: input)
        #expect(result != nil)
    }

    @Test("many nested empty objects")
    func manyNestedEmpty() {
        // 50 levels deep
        let open = String(repeating: "{\"a\":", count: 50)
        let close = String(repeating: "}", count: 50)
        let input = open + "1" + close
        let result = ClaudePipe.extractJSON(from: input)
        #expect(result != nil)
        guard let extracted = result else {
            Issue.record("Expected non-nil result")
            return
        }
        let parsed = try? JSONSerialization.jsonObject(with: Data(extracted.utf8))
        #expect(parsed != nil)
    }

    @Test("alternating quotes and braces outside any object")
    func alternatingQuotesBraces() {
        let result = ClaudePipe.extractJSON(from: #"" { " } " { " }"#)
        // This is adversarial — the parser may get confused by unmatched quotes
        // Just ensure it doesn't crash and returns nil or valid JSON
        if let json = result {
            let parsed = try? JSONSerialization.jsonObject(with: Data(json.utf8))
            #expect(parsed != nil, "extractJSON returned invalid JSON: \(json)")
        }
    }

    @Test("null bytes in input don't crash")
    func nullBytes() {
        let input = "{\"a\": 1}\0\0\0"
        let result = ClaudePipe.extractJSON(from: input)
        #expect(result == "{\"a\": 1}")
    }

    @Test("newlines inside JSON string values")
    func newlinesInValues() {
        // Real JSON doesn't allow literal newlines in strings, but claude might emit them
        let input = "{\"msg\": \"line1\\nline2\"}"
        let result = ClaudePipe.extractJSON(from: input)
        #expect(result != nil)
    }

    // MARK: - The critical bug hunt: escaped backslash before closing quote

    @Test("double backslash before quote is NOT an escaped quote")
    func doubleBackslashBeforeQuote() {
        // In JSON: "val\\" means the string value is: val\
        // The \\ is an escaped backslash, and the " after it closes the string.
        // A naive parser that just tracks escaped=true on \ would misparse this.
        let input = #"{"path": "C:\\", "ok": true}"#
        let result = ClaudePipe.extractJSON(from: input)
        #expect(result != nil, "Should extract the full object")
        if let json = result {
            let parsed = try? JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any]
            #expect(parsed != nil, "Result should be valid JSON: \(json)")
            if let dict = parsed {
                #expect(dict["ok"] as? Bool == true, "Should have parsed 'ok' field")
            }
        }
    }

    @Test("triple backslash before quote IS an escaped quote")
    func tripleBackslashBeforeQuote() {
        // \\\" = escaped backslash + escaped quote -> string continues
        let input = #"{"val": "a\\\"b"}"#
        let result = ClaudePipe.extractJSON(from: input)
        #expect(result != nil)
        if let json = result {
            let parsed = try? JSONSerialization.jsonObject(with: Data(json.utf8))
            #expect(parsed != nil, "Result should be valid JSON: \(json)")
        }
    }

    @Test("quadruple backslash before quote is NOT an escaped quote")
    func quadBackslashBeforeQuote() {
        // \\\\" = two escaped backslashes, then " closes the string
        let input = #"{"val": "a\\\\", "b": 1}"#
        let result = ClaudePipe.extractJSON(from: input)
        #expect(result != nil)
        if let json = result {
            let parsed = try? JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any]
            #expect(parsed != nil, "Result should be valid JSON: \(json)")
            if let dict = parsed {
                #expect(dict["b"] as? Int == 1, "Should have parsed 'b' field after \\\\\\\\")
            }
        }
    }
}

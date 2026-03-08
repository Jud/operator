import Foundation
import Testing

@testable import OperatorCore

/// Namespace matching the file name to satisfy the file_name lint rule.
private enum RoutingPromptTests {}

// MARK: - System Instruction

@Suite("RoutingPrompt - System Instruction")
internal struct RoutingPromptSystemInstructionTests {
    @Test("system instruction contains key routing directives")
    func systemInstructionContent() {
        let instruction = RoutingPrompt.systemInstruction

        #expect(!instruction.isEmpty)
        #expect(instruction.contains("message router"))
        #expect(instruction.contains("session"))
        #expect(instruction.contains("confident"))
    }

    @Test("system instruction mentions JSON output format")
    func systemInstructionMentionsJSON() {
        let instruction = RoutingPrompt.systemInstruction

        #expect(instruction.contains("JSON"))
        #expect(instruction.contains("session"))
        #expect(instruction.contains("candidates"))
        #expect(instruction.contains("question"))
    }

    @Test("system instruction includes ambiguity handling rules")
    func systemInstructionAmbiguityRules() {
        let instruction = RoutingPrompt.systemInstruction

        #expect(instruction.contains("ambiguous"))
        #expect(instruction.contains("clarify"))
    }
}

// MARK: - JSON Schema

@Suite("RoutingPrompt - JSON Schema")
internal struct RoutingPromptSchemaTests {
    private func parseSchema() throws -> [String: Any] {
        guard let data = RoutingPrompt.outputSchemaString.data(using: .utf8),
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            throw SchemaParseError.invalidJSON
        }
        return json
    }

    private func parseProperties() throws -> [String: Any] {
        let json = try parseSchema()
        guard let properties = json["properties"] as? [String: Any] else {
            throw SchemaParseError.missingProperties
        }
        return properties
    }

    @Test("output schema string is valid JSON")
    func outputSchemaIsValidJSON() throws {
        let json = try parseSchema()
        #expect(json["type"] as? String == "object")
    }

    @Test("output schema has all required fields defined in properties")
    func outputSchemaHasAllFields() throws {
        let properties = try parseProperties()

        #expect(properties["session"] != nil)
        #expect(properties["confident"] != nil)
        #expect(properties["candidates"] != nil)
        #expect(properties["question"] != nil)
    }

    @Test("session field is typed as string")
    func sessionFieldIsString() throws {
        let properties = try parseProperties()
        guard let session = properties["session"] as? [String: Any] else {
            Issue.record("session property is not a dictionary")
            return
        }
        #expect(session["type"] as? String == "string")
    }

    @Test("confident field is typed as boolean")
    func confidentFieldIsBoolean() throws {
        let properties = try parseProperties()
        guard let confident = properties["confident"] as? [String: Any] else {
            Issue.record("confident property is not a dictionary")
            return
        }
        #expect(confident["type"] as? String == "boolean")
    }

    @Test("candidates field is typed as array of strings")
    func candidatesFieldIsArrayOfStrings() throws {
        let properties = try parseProperties()
        guard let candidates = properties["candidates"] as? [String: Any] else {
            Issue.record("candidates property is not a dictionary")
            return
        }
        #expect(candidates["type"] as? String == "array")
        guard let items = candidates["items"] as? [String: Any] else {
            Issue.record("candidates items is not a dictionary")
            return
        }
        #expect(items["type"] as? String == "string")
    }

    @Test("question field is typed as string")
    func questionFieldIsString() throws {
        let properties = try parseProperties()
        guard let question = properties["question"] as? [String: Any] else {
            Issue.record("question property is not a dictionary")
            return
        }
        #expect(question["type"] as? String == "string")
    }

    @Test("required fields include session and confident")
    func requiredFieldsPresent() throws {
        let json = try parseSchema()
        guard let required = json["required"] as? [String] else {
            Issue.record("required is not a string array")
            return
        }
        #expect(required.contains("session"))
        #expect(required.contains("confident"))
    }

    @Test("additionalProperties is false")
    func additionalPropertiesFalse() throws {
        let json = try parseSchema()
        #expect(json["additionalProperties"] as? Bool == false)
    }
}

// MARK: - Prompt Construction

@Suite("RoutingPrompt - Prompt Construction")
internal struct RoutingPromptConstructionTests {
    @Test("wrapForLocalModel appends /no_think flag")
    func wrapAppendsNoThink() {
        let prompt = "Route this message to the right session"
        let wrapped = RoutingPrompt.wrapForLocalModel(prompt)

        #expect(wrapped.hasSuffix("/no_think"))
    }

    @Test("wrapForLocalModel preserves original prompt content")
    func wrapPreservesContent() {
        let prompt = "Given sessions: frontend, backend. Message: fix the CSS"
        let wrapped = RoutingPrompt.wrapForLocalModel(prompt)

        #expect(wrapped.contains(prompt))
    }

    @Test("wrapForLocalModel adds newline before /no_think")
    func wrapHasNewlineBeforeFlag() {
        let prompt = "test prompt"
        let wrapped = RoutingPrompt.wrapForLocalModel(prompt)

        #expect(wrapped.contains("\n/no_think"))
    }
}

// MARK: - Performance

@Suite("RoutingPrompt - Performance")
internal struct RoutingPromptPerformanceTests {
    @Test("prompt construction completes in under 1ms")
    func promptConstructionPerformance() {
        let sessionContext = """
            Sessions:
            1. "frontend" (cwd: /Users/dev/myapp/frontend, context: React dashboard)
            2. "backend" (cwd: /Users/dev/myapp/api, context: Express API server)
            3. "infra" (cwd: /Users/dev/myapp/terraform, context: AWS infrastructure)

            Recent routing history:
            - "fix the login form" -> frontend (15s ago)
            - "add rate limiting" -> backend (45s ago)

            User message: "update the dashboard CSS"
            """

        let start = ContinuousClock.now
        for _ in 0..<1_000 {
            _ = RoutingPrompt.wrapForLocalModel(sessionContext)
        }
        let elapsed = ContinuousClock.now - start

        #expect(elapsed < .milliseconds(100))
    }
}

// MARK: - Private Helpers

private enum SchemaParseError: Error {
    case invalidJSON
    case missingProperties
}

// swiftlint:disable:this file_name
import Foundation
import Testing

@testable import OperatorCore

// MARK: - System Instruction

@Suite("RoutingPrompt - System Instruction")
internal struct RoutingPromptSystemInstructionTests {
    @Test("system instruction contains routing directives and JSON output format")
    func systemInstructionContent() {
        let instruction = RoutingPrompt.systemInstruction

        #expect(!instruction.isEmpty)
        #expect(instruction.contains("Route the user message"))
        #expect(instruction.contains("JSON"))
        #expect(instruction.contains("session"))
        #expect(instruction.contains("confident"))
    }

    @Test("system instruction includes ambiguity handling rules")
    func systemInstructionAmbiguityRules() {
        let instruction = RoutingPrompt.systemInstruction

        #expect(instruction.contains("empty"))
        #expect(instruction.contains("true|false"))
    }
}

// MARK: - JSON Schema

@Suite("RoutingPrompt - JSON Schema")
internal struct RoutingPromptSchemaTests {
    private enum SchemaParseError: Error {
        case invalidJSON
        case missingProperties
    }

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

    @Test("output schema has required fields defined in properties")
    func outputSchemaHasAllFields() throws {
        let properties = try parseProperties()

        #expect(properties["session"] != nil)
        #expect(properties["confident"] != nil)
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
    @Test("buildPrompt includes routing context header")
    func buildPromptIncludesContext() {
        let prompt = RoutingPrompt.buildPrompt(
            text: "fix the CSS grid layout",
            sessions: [],
            routingState: RoutingState()
        )

        #expect(prompt.contains("Routing context:"))
        #expect(prompt.contains("Current user message:"))
        #expect(prompt.contains("fix the CSS grid layout"))
    }

    @Test("splitPrompt separates reusable context from current user message")
    func splitPromptSeparatesContextAndMessage() {
        let prompt = RoutingPrompt.buildPrompt(
            text: "fix the CSS grid layout",
            sessions: [],
            routingState: RoutingState()
        )

        let parts = RoutingPrompt.splitPrompt(prompt)

        #expect(parts != nil)
        #expect(parts?.context.contains("Routing context:") == true)
        #expect(parts?.currentMessage == "fix the CSS grid layout")
    }
}

// MARK: - Performance

@Suite("RoutingPrompt - Performance")
internal struct RoutingPromptPerformanceTests {
    @Test("prompt construction completes in under 1ms")
    func promptConstructionPerformance() {
        let start = ContinuousClock.now
        for _ in 0..<1_000 {
            _ = RoutingPrompt.buildPrompt(
                text: "update the dashboard CSS",
                sessions: [],
                routingState: RoutingState()
            )
        }
        let elapsed = ContinuousClock.now - start

        #expect(elapsed < .milliseconds(100))
    }
}

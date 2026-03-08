import Foundation
import Testing

@testable import OperatorCore

/// Namespace matching the file name to satisfy the file_name lint rule.
private enum MLXRoutingEngineTests {}

// MARK: - MLXRoutingError

@Suite("MLXRoutingError")
internal struct MLXRoutingErrorTests {
    @Test("modelLoadFailed includes reason in description")
    func modelLoadFailedDescription() {
        let error = MLXRoutingError.modelLoadFailed("insufficient memory")
        #expect(error.description.contains("insufficient memory"))
        #expect(error.description.contains("failed to load"))
    }

    @Test("modelNotLoaded has meaningful description")
    func modelNotLoadedDescription() {
        let error = MLXRoutingError.modelNotLoaded
        #expect(error.description.contains("not loaded"))
    }

    @Test("invalidOutput includes truncated output in description")
    func invalidOutputDescription() {
        let longOutput = String(repeating: "x", count: 300)
        let error = MLXRoutingError.invalidOutput(longOutput)
        #expect(error.description.contains("invalid output"))
        #expect(error.description.count < longOutput.count + 100)
    }

    @Test("timeout includes duration in description")
    func timeoutDescription() {
        let error = MLXRoutingError.timeout(15.0)
        #expect(error.description.contains("15"))
        #expect(error.description.contains("timed out"))
    }
}

// MARK: - Model Configuration

@Suite("MLXRoutingEngine - Configuration")
internal struct MLXRoutingConfigTests {
    @Test("model ID matches expected Qwen 3.5 0.8B 4-bit quantized identifier")
    func modelIDIsCorrect() {
        #expect(MLXRoutingEngine.modelID == "mlx-community/Qwen3.5-0.8B-MLX-4bit")
    }

    @Test("engine can be instantiated with ModelManager")
    func instantiation() async {
        let cacheDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlx-test-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: cacheDir) }

        let modelManager = ModelManager(cacheDirectory: cacheDir)
        let engine = MLXRoutingEngine(modelManager: modelManager)
        let _: any RoutingEngine = engine
        _ = engine
    }
}

// MARK: - JSON Output Parsing

@Suite("MLXRoutingEngine - JSON Output Parsing")
internal struct MLXRoutingJSONTests {
    @Test("confident routing JSON has expected structure")
    func confidentRoutingJSON() throws {
        let json: [String: Any] = [
            "session": "frontend",
            "confident": true,
            "candidates": ["frontend"],
            "question": ""
        ]

        let data = try JSONSerialization.data(withJSONObject: json)
        guard let parsed = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            Issue.record("Failed to parse JSON")
            return
        }

        #expect(parsed["session"] as? String == "frontend")
        #expect(parsed["confident"] as? Bool == true)
        #expect((parsed["candidates"] as? [String])?.count == 1)
    }

    @Test("ambiguous routing JSON includes candidates and question")
    func ambiguousRoutingJSON() throws {
        let json: [String: Any] = [
            "session": "",
            "confident": false,
            "candidates": ["frontend", "backend"],
            "question": "Which project should receive this CSS fix?"
        ]

        let data = try JSONSerialization.data(withJSONObject: json)
        guard let parsed = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            Issue.record("Failed to parse JSON")
            return
        }

        #expect((parsed["session"] as? String)?.isEmpty == true)
        #expect(parsed["confident"] as? Bool == false)
        #expect((parsed["candidates"] as? [String])?.count == 2)
        #expect((parsed["question"] as? String)?.isEmpty == false)
    }

    @Test("routing schema matches format expected by MessageRouter")
    func schemaMatchesRouterExpectations() throws {
        guard let schemaData = RoutingPrompt.outputSchemaString.data(using: .utf8),
            let schema = try JSONSerialization.jsonObject(with: schemaData) as? [String: Any],
            let properties = schema["properties"] as? [String: Any]
        else {
            Issue.record("Failed to parse schema")
            return
        }

        let expectedKeys = Set(["session", "confident", "candidates", "question"])
        let schemaKeys = Set(properties.keys)

        #expect(expectedKeys.isSubset(of: schemaKeys))
    }
}

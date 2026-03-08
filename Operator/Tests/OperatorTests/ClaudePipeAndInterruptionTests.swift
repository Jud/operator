import Foundation
import Testing

@testable import OperatorCore

/// Namespace matching the file name to satisfy the file_name lint rule.
private enum ClaudePipeAndInterruptionTests {}

// MARK: - ClaudePipe.extractJSON Tests

@Suite("ClaudePipe.extractJSON")
internal struct ClaudePipeExtractJSONTests {
    @Test("extracts simple flat JSON object")
    func simpleFlatJSON() {
        let input = #"{"action": "skip"}"#
        let result = ClaudePipe.extractJSON(from: input)
        #expect(result == #"{"action": "skip"}"#)
    }

    @Test("extracts JSON with nested braces")
    func nestedBraces() {
        let input = #"{"outer": {"inner": 1}}"#
        let result = ClaudePipe.extractJSON(from: input)
        #expect(result == #"{"outer": {"inner": 1}}"#)
    }

    @Test("handles escaped quotes inside strings")
    func escapedQuotes() {
        let input = #"{"msg": "he said \"hi\""}"#
        let result = ClaudePipe.extractJSON(from: input)
        #expect(result == #"{"msg": "he said \"hi\""}"#)
    }

    @Test("extracts JSON with preamble text before it")
    func preambleText() {
        let input = #"Here is the response: {"session": "sudo"}"#
        let result = ClaudePipe.extractJSON(from: input)
        #expect(result == #"{"session": "sudo"}"#)
    }

    @Test("extracts JSON ignoring trailing text")
    func trailingText() {
        let input = #"{"a": 1} some trailing text"#
        let result = ClaudePipe.extractJSON(from: input)
        #expect(result == #"{"a": 1}"#)
    }

    @Test("returns nil when no JSON is found")
    func noJSON() {
        let result = ClaudePipe.extractJSON(from: "no json here")
        #expect(result == nil)
    }

    @Test("returns nil for empty string")
    func emptyString() {
        let result = ClaudePipe.extractJSON(from: "")
        #expect(result == nil)
    }

    @Test("returns only the first JSON object when multiple are present")
    func multipleJSONObjects() {
        let input = #"{"first": 1} {"second": 2}"#
        let result = ClaudePipe.extractJSON(from: input)
        #expect(result == #"{"first": 1}"#)
    }

    @Test("handles strings containing curly braces")
    func stringWithBraces() {
        let input = #"{"text": "use {curly} braces"}"#
        let result = ClaudePipe.extractJSON(from: input)
        #expect(result == #"{"text": "use {curly} braces"}"#)
    }

    @Test("flat JSON works via fallback regex path as well")
    func fallbackRegexFlat() {
        // A simple flat JSON should be extractable; this confirms the method
        // returns correct results regardless of which internal path fires.
        let input = #"{"key": "value"}"#
        let result = ClaudePipe.extractJSON(from: input)
        #expect(result == #"{"key": "value"}"#)
    }
}

// MARK: - Mock RoutingEngine

private final class MockRoutingEngine: RoutingEngine, @unchecked Sendable {
    var response: [String: Any] = [:]
    var shouldThrow: (any Error)?

    func run(prompt: String, timeout: TimeInterval) async throws -> [String: Any] {
        if let error = shouldThrow { throw error }
        return response
    }
}

// MARK: - InterruptionHandler Tests

@Suite("InterruptionHandler")
internal struct InterruptionHandlerTests {
    /// A reusable interruption context for tests that need engine fallback.
    private static let sampleContext = InterruptionContext(
        agentName: "sudo",
        fullText: "I finished refactoring the module.",
        heardText: "I finished",
        unheardText: "refactoring the module."
    )

    // MARK: - Fast-path skip

    @Test("skip keyword triggers .skip", arguments: ["skip", "move on", "next"])
    func skipKeywords(keyword: String) async {
        let handler = InterruptionHandler(engine: MockRoutingEngine())
        let action = await handler.handle(
            userUtterance: keyword,
            context: Self.sampleContext,
            registeredSessionNames: []
        )
        guard case .skip = action else {
            Issue.record("Expected .skip for '\(keyword)', got \(action)")
            return
        }
    }

    @Test("skip keywords are case-insensitive", arguments: ["SKIP", "Move On", "NEXT"])
    func skipCaseInsensitive(keyword: String) async {
        let handler = InterruptionHandler(engine: MockRoutingEngine())
        let action = await handler.handle(
            userUtterance: keyword,
            context: Self.sampleContext,
            registeredSessionNames: []
        )
        guard case .skip = action else {
            Issue.record("Expected .skip for '\(keyword)', got \(action)")
            return
        }
    }

    // MARK: - Fast-path replay

    @Test(
        "replay keyword triggers .replay",
        arguments: ["replay", "say that again", "what?", "what"]
    )
    func replayKeywords(keyword: String) async {
        let handler = InterruptionHandler(engine: MockRoutingEngine())
        let action = await handler.handle(
            userUtterance: keyword,
            context: Self.sampleContext,
            registeredSessionNames: []
        )
        guard case .replay = action else {
            Issue.record("Expected .replay for '\(keyword)', got \(action)")
            return
        }
    }

    // MARK: - Agent target

    @Test("explicit agent target routes to that agent")
    func agentTarget() async {
        let handler = InterruptionHandler(engine: MockRoutingEngine())
        let action = await handler.handle(
            userUtterance: "tell sudo do something",
            context: Self.sampleContext,
            registeredSessionNames: ["sudo"]
        )
        guard case .route(let target, let message) = action else {
            Issue.record("Expected .route, got \(action)")
            return
        }
        #expect(target == "sudo")
        #expect(message == "do something")
    }

    // MARK: - Engine fallback

    @Test("engine fallback returning skip action yields .skip")
    func engineFallbackSkip() async {
        let mock = MockRoutingEngine()
        mock.response = ["action": "skip", "reason": "test"]
        let handler = InterruptionHandler(engine: mock)
        let action = await handler.handle(
            userUtterance: "hmm I dunno",
            context: Self.sampleContext,
            registeredSessionNames: []
        )
        guard case .skip = action else {
            Issue.record("Expected .skip from engine fallback, got \(action)")
            return
        }
    }

    @Test("engine fallback returning replay action yields .replay")
    func engineFallbackReplay() async {
        let mock = MockRoutingEngine()
        mock.response = ["action": "replay", "reason": "test"]
        let handler = InterruptionHandler(engine: mock)
        let action = await handler.handle(
            userUtterance: "hmm I dunno",
            context: Self.sampleContext,
            registeredSessionNames: []
        )
        guard case .replay = action else {
            Issue.record("Expected .replay from engine fallback, got \(action)")
            return
        }
    }

    @Test("engine fallback returning ask action yields .ask with question")
    func engineFallbackAsk() async {
        let mock = MockRoutingEngine()
        mock.response = ["action": "ask", "question": "Did you mean...?"]
        let handler = InterruptionHandler(engine: mock)
        let action = await handler.handle(
            userUtterance: "hmm I dunno",
            context: Self.sampleContext,
            registeredSessionNames: []
        )
        guard case .ask(let question) = action else {
            Issue.record("Expected .ask from engine fallback, got \(action)")
            return
        }
        #expect(question == "Did you mean...?")
    }

    @Test("engine fallback with unknown action yields .ask with fallback question")
    func engineFallbackUnknownAction() async {
        let mock = MockRoutingEngine()
        mock.response = ["action": "unknown"]
        let handler = InterruptionHandler(engine: mock)
        let action = await handler.handle(
            userUtterance: "hmm I dunno",
            context: Self.sampleContext,
            registeredSessionNames: []
        )
        guard case .ask(let question) = action else {
            Issue.record("Expected .ask from engine fallback, got \(action)")
            return
        }
        #expect(question == "Did you want to hear the rest of sudo's message?")
    }

    @Test("engine failure yields .ask with fallback question")
    func engineFailure() async {
        let mock = MockRoutingEngine()
        mock.shouldThrow = ClaudePipeError.timeout(seconds: 10)
        let handler = InterruptionHandler(engine: mock)
        let action = await handler.handle(
            userUtterance: "hmm I dunno",
            context: Self.sampleContext,
            registeredSessionNames: []
        )
        guard case .ask(let question) = action else {
            Issue.record("Expected .ask from engine failure, got \(action)")
            return
        }
        #expect(question == "Did you want to hear the rest of sudo's message?")
    }

    // MARK: - Whitespace trimming

    @Test("whitespace around keyword is trimmed before matching")
    func whitespaceTrimming() async {
        let handler = InterruptionHandler(engine: MockRoutingEngine())
        let action = await handler.handle(
            userUtterance: "  skip  ",
            context: Self.sampleContext,
            registeredSessionNames: []
        )
        guard case .skip = action else {
            Issue.record("Expected .skip for trimmed input, got \(action)")
            return
        }
    }
}

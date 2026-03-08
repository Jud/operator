import AVFoundation
import Foundation
import Testing

@testable import OperatorCore

/// Namespace matching the file name to satisfy the file_name lint rule.
private enum ClarificationTests {}

// MARK: - matchMetaIntent Tests

@Suite("MessageRouter - matchMetaIntent")
internal struct MatchMetaIntentTests {
    private func makeRouter() -> MessageRouter {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)
        return MessageRouter(registry: registry)
    }

    // MARK: Cancel patterns

    @Test("cancel triggers .cancelled")
    func cancelWord() {
        let result = makeRouter().matchMetaIntent(lowered: "cancel", candidates: ["sudo"])
        guard case .cancelled = result else {
            Issue.record("Expected .cancelled, got \(String(describing: result))")
            return
        }
    }

    @Test("never mind triggers .cancelled")
    func neverMind() {
        let result = makeRouter().matchMetaIntent(lowered: "never mind", candidates: ["sudo"])
        guard case .cancelled = result else {
            Issue.record("Expected .cancelled, got \(String(describing: result))")
            return
        }
    }

    @Test("nevermind triggers .cancelled")
    func nevermind() {
        let result = makeRouter().matchMetaIntent(lowered: "nevermind", candidates: ["sudo"])
        guard case .cancelled = result else {
            Issue.record("Expected .cancelled, got \(String(describing: result))")
            return
        }
    }

    @Test("forget it triggers .cancelled")
    func forgetIt() {
        let result = makeRouter().matchMetaIntent(lowered: "forget it", candidates: ["sudo"])
        guard case .cancelled = result else {
            Issue.record("Expected .cancelled, got \(String(describing: result))")
            return
        }
    }

    @Test("forget about it triggers .cancelled")
    func forgetAboutIt() {
        let result = makeRouter().matchMetaIntent(lowered: "forget about it", candidates: ["sudo"])
        guard case .cancelled = result else {
            Issue.record("Expected .cancelled, got \(String(describing: result))")
            return
        }
    }

    @Test("nah triggers .cancelled")
    func nah() {
        let result = makeRouter().matchMetaIntent(lowered: "nah", candidates: ["sudo"])
        guard case .cancelled = result else {
            Issue.record("Expected .cancelled, got \(String(describing: result))")
            return
        }
    }

    @Test("nope triggers .cancelled")
    func nope() {
        let result = makeRouter().matchMetaIntent(lowered: "nope", candidates: ["sudo"])
        guard case .cancelled = result else {
            Issue.record("Expected .cancelled, got \(String(describing: result))")
            return
        }
    }

    // MARK: Multi-send patterns

    @Test("both triggers .sendToAll")
    func both() {
        let result = makeRouter().matchMetaIntent(lowered: "both", candidates: ["sudo", "frontend"])
        guard case .sendToAll = result else {
            Issue.record("Expected .sendToAll, got \(String(describing: result))")
            return
        }
    }

    @Test("all of them triggers .sendToAll")
    func allOfThem() {
        let result = makeRouter().matchMetaIntent(lowered: "all of them", candidates: ["sudo", "frontend"])
        guard case .sendToAll = result else {
            Issue.record("Expected .sendToAll, got \(String(describing: result))")
            return
        }
    }

    @Test("all of 'em triggers .sendToAll")
    func allOfEm() {
        let result = makeRouter().matchMetaIntent(lowered: "all of 'em", candidates: ["sudo", "frontend"])
        guard case .sendToAll = result else {
            Issue.record("Expected .sendToAll, got \(String(describing: result))")
            return
        }
    }

    @Test("all agents triggers .sendToAll")
    func allAgents() {
        let result = makeRouter().matchMetaIntent(lowered: "all agents", candidates: ["sudo", "frontend"])
        guard case .sendToAll = result else {
            Issue.record("Expected .sendToAll, got \(String(describing: result))")
            return
        }
    }

    @Test("everyone triggers .sendToAll")
    func everyone() {
        let result = makeRouter().matchMetaIntent(lowered: "everyone", candidates: ["sudo", "frontend"])
        guard case .sendToAll = result else {
            Issue.record("Expected .sendToAll, got \(String(describing: result))")
            return
        }
    }

    @Test("exact 'all' triggers .sendToAll")
    func allExact() {
        let result = makeRouter().matchMetaIntent(lowered: "all", candidates: ["sudo", "frontend"])
        guard case .sendToAll = result else {
            Issue.record("Expected .sendToAll, got \(String(describing: result))")
            return
        }
    }

    // MARK: Repeat patterns

    @Test("repeat triggers .repeatQuestion")
    func repeatWord() {
        let result = makeRouter().matchMetaIntent(lowered: "repeat", candidates: ["sudo"])
        guard case .repeatQuestion = result else {
            Issue.record("Expected .repeatQuestion, got \(String(describing: result))")
            return
        }
    }

    @Test("say that again triggers .repeatQuestion")
    func sayThatAgain() {
        let result = makeRouter().matchMetaIntent(lowered: "say that again", candidates: ["sudo"])
        guard case .repeatQuestion = result else {
            Issue.record("Expected .repeatQuestion, got \(String(describing: result))")
            return
        }
    }

    @Test("what? triggers .repeatQuestion")
    func whatQuestion() {
        let result = makeRouter().matchMetaIntent(lowered: "what?", candidates: ["sudo"])
        guard case .repeatQuestion = result else {
            Issue.record("Expected .repeatQuestion, got \(String(describing: result))")
            return
        }
    }

    @Test("what triggers .repeatQuestion")
    func whatPlain() {
        let result = makeRouter().matchMetaIntent(lowered: "what", candidates: ["sudo"])
        guard case .repeatQuestion = result else {
            Issue.record("Expected .repeatQuestion, got \(String(describing: result))")
            return
        }
    }

    @Test("come again triggers .repeatQuestion")
    func comeAgain() {
        let result = makeRouter().matchMetaIntent(lowered: "come again", candidates: ["sudo"])
        guard case .repeatQuestion = result else {
            Issue.record("Expected .repeatQuestion, got \(String(describing: result))")
            return
        }
    }

    @Test("huh triggers .repeatQuestion")
    func huh() {
        let result = makeRouter().matchMetaIntent(lowered: "huh", candidates: ["sudo"])
        guard case .repeatQuestion = result else {
            Issue.record("Expected .repeatQuestion, got \(String(describing: result))")
            return
        }
    }

    // MARK: Non-matching

    @Test("unrecognized input returns nil")
    func unrecognized() {
        let result = makeRouter().matchMetaIntent(lowered: "something random", candidates: ["sudo"])
        #expect(result == nil)
    }

    @Test("send it to sudo returns nil")
    func sendToAgent() {
        let result = makeRouter().matchMetaIntent(lowered: "send it to sudo", candidates: ["sudo"])
        #expect(result == nil)
    }
}

// MARK: - matchOrdinal Tests

@Suite("MessageRouter - matchOrdinal")
internal struct MatchOrdinalTests {
    private func makeRouter() -> MessageRouter {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)
        return MessageRouter(registry: registry)
    }

    @Test("first with two candidates returns first")
    func firstOfTwo() {
        let result = makeRouter().matchOrdinal(lowered: "first", candidates: ["sudo", "frontend"])
        #expect(result == "sudo")
    }

    @Test("the first one with two candidates returns first")
    func theFirstOne() {
        let result = makeRouter().matchOrdinal(lowered: "the first one", candidates: ["sudo", "frontend"])
        #expect(result == "sudo")
    }

    @Test("second with two candidates returns second")
    func secondOfTwo() {
        let result = makeRouter().matchOrdinal(lowered: "second", candidates: ["sudo", "frontend"])
        #expect(result == "frontend")
    }

    @Test("the second one with two candidates returns second")
    func theSecondOne() {
        let result = makeRouter().matchOrdinal(lowered: "the second one", candidates: ["sudo", "frontend"])
        #expect(result == "frontend")
    }

    @Test("third with three candidates returns third")
    func thirdOfThree() {
        let result = makeRouter().matchOrdinal(lowered: "third", candidates: ["sudo", "frontend", "backend"])
        #expect(result == "backend")
    }

    @Test("the third one with three candidates returns third")
    func theThirdOne() {
        let result = makeRouter().matchOrdinal(lowered: "the third one", candidates: ["a", "b", "c"])
        #expect(result == "c")
    }

    @Test("second with single candidate returns nil (out of bounds)")
    func secondOutOfBounds() {
        let result = makeRouter().matchOrdinal(lowered: "second", candidates: ["sudo"])
        #expect(result == nil)
    }

    @Test("third with two candidates returns nil (out of bounds)")
    func thirdOutOfBounds() {
        let result = makeRouter().matchOrdinal(lowered: "third", candidates: ["sudo", "frontend"])
        #expect(result == nil)
    }

    @Test("number 1 with one candidate returns first")
    func numberOne() {
        let result = makeRouter().matchOrdinal(lowered: "number 1", candidates: ["alpha"])
        #expect(result == "alpha")
    }

    @Test("number 2 with two candidates returns second")
    func numberTwo() {
        let result = makeRouter().matchOrdinal(lowered: "number 2", candidates: ["alpha", "beta"])
        #expect(result == "beta")
    }

    @Test("first with empty candidates returns nil")
    func firstEmpty() {
        let result = makeRouter().matchOrdinal(lowered: "first", candidates: [])
        #expect(result == nil)
    }
}

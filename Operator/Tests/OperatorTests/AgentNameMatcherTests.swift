import Foundation
import Testing

@testable import OperatorCore

// MARK: - AgentNameMatcher Tests

@Suite("AgentNameMatcher")
internal struct AgentNameMatcherTests {
    @Test("tell <agent> extracts session and message")
    func tellPattern() {
        let result = AgentNameMatcher.match(in: "tell sudo fix the build", sessionNames: ["sudo"])
        #expect(result?.session == "sudo")
        #expect(result?.message == "fix the build")
    }

    @Test("hey <agent> extracts session and message")
    func heyPattern() {
        let result = AgentNameMatcher.match(in: "hey frontend make it blue", sessionNames: ["frontend"])
        #expect(result?.session == "frontend")
        #expect(result?.message == "make it blue")
    }

    @Test("@agent extracts session and message")
    func atPattern() {
        let result = AgentNameMatcher.match(in: "@sudo check the logs", sessionNames: ["sudo"])
        #expect(result?.session == "sudo")
        #expect(result?.message == "check the logs")
    }

    @Test("case insensitive matching preserves canonical casing")
    func caseInsensitive() {
        let result = AgentNameMatcher.match(in: "tell SUDO fix it", sessionNames: ["Sudo"])
        #expect(result?.session == "Sudo")
        #expect(result?.message == "fix it")
    }

    @Test("unregistered agent returns nil")
    func unregisteredAgent() {
        let result = AgentNameMatcher.match(in: "tell bob fix it", sessionNames: ["sudo"])
        #expect(result == nil)
    }

    @Test("comma separator after agent name is stripped")
    func commaSeparator() {
        let result = AgentNameMatcher.match(in: "tell sudo, fix the build", sessionNames: ["sudo"])
        #expect(result?.session == "sudo")
        #expect(result?.message == "fix the build")
    }

    @Test("colon separator after agent name is stripped")
    func colonSeparator() {
        let result = AgentNameMatcher.match(in: "hey sudo: check logs", sessionNames: ["sudo"])
        #expect(result?.session == "sudo")
        #expect(result?.message == "check logs")
    }

    @Test("no message after name returns full utterance")
    func noMessageAfterName() {
        let result = AgentNameMatcher.match(in: "tell sudo", sessionNames: ["sudo"])
        #expect(result?.session == "sudo")
        #expect(result?.message == "tell sudo")
    }

    @Test("empty session list returns nil")
    func emptySessionList() {
        let result = AgentNameMatcher.match(in: "tell sudo fix it", sessionNames: [])
        #expect(result == nil)
    }

    @Test("no pattern match returns nil")
    func noPatternMatch() {
        let result = AgentNameMatcher.match(in: "just fix the build", sessionNames: ["sudo"])
        #expect(result == nil)
    }

    @Test("multiple registered names matches correct one")
    func multipleRegisteredNames() {
        let result = AgentNameMatcher.match(
            in: "tell frontend update styles",
            sessionNames: ["sudo", "frontend"]
        )
        #expect(result?.session == "frontend")
        #expect(result?.message == "update styles")
    }
}

// MARK: - MessageRouter Operator Command Tests

@Suite("MessageRouter - Operator Commands (comprehensive)")
internal struct OperatorCommandMatchingTests {
    private func routeText(_ text: String) async -> RoutingResult {
        let voiceManager = VoiceManager()
        let registry = SessionRegistry(voiceManager: voiceManager)
        let router = MessageRouter(registry: registry)
        let state = RoutingState()
        return await router.route(text: text, routingState: state)
    }

    // MARK: - Status commands

    @Test("operator status matches status command")
    func operatorStatus() async {
        let result = await routeText("operator status")
        guard case .operatorCommand(.status) = result else {
            Issue.record("Expected .operatorCommand(.status), got \(result)")
            return
        }
    }

    @Test("operator what matches status command")
    func operatorWhat() async {
        let result = await routeText("operator what")
        guard case .operatorCommand(.status) = result else {
            Issue.record("Expected .operatorCommand(.status), got \(result)")
            return
        }
    }

    @Test("what's everyone working on matches status command")
    func whatsEveryoneWorkingOn() async {
        let result = await routeText("what's everyone working on")
        guard case .operatorCommand(.status) = result else {
            Issue.record("Expected .operatorCommand(.status), got \(result)")
            return
        }
    }

    @Test("whats everyone working on matches status command")
    func whatsEveryoneWorkingOnNoApostrophe() async {
        let result = await routeText("whats everyone working on")
        guard case .operatorCommand(.status) = result else {
            Issue.record("Expected .operatorCommand(.status), got \(result)")
            return
        }
    }

    @Test("what is everyone working on matches status command")
    func whatIsEveryoneWorkingOn() async {
        let result = await routeText("what is everyone working on")
        guard case .operatorCommand(.status) = result else {
            Issue.record("Expected .operatorCommand(.status), got \(result)")
            return
        }
    }

    // MARK: - List agents commands

    @Test("list agents matches listAgents command")
    func listAgents() async {
        let result = await routeText("list agents")
        guard case .operatorCommand(.listAgents) = result else {
            Issue.record("Expected .operatorCommand(.listAgents), got \(result)")
            return
        }
    }

    @Test("list sessions matches listAgents command")
    func listSessions() async {
        let result = await routeText("list sessions")
        guard case .operatorCommand(.listAgents) = result else {
            Issue.record("Expected .operatorCommand(.listAgents), got \(result)")
            return
        }
    }

    @Test("who's running matches listAgents command")
    func whosRunning() async {
        let result = await routeText("who's running")
        guard case .operatorCommand(.listAgents) = result else {
            Issue.record("Expected .operatorCommand(.listAgents), got \(result)")
            return
        }
    }

    @Test("whos running matches listAgents command")
    func whosRunningNoApostrophe() async {
        let result = await routeText("whos running")
        guard case .operatorCommand(.listAgents) = result else {
            Issue.record("Expected .operatorCommand(.listAgents), got \(result)")
            return
        }
    }

    @Test("who is running matches listAgents command")
    func whoIsRunning() async {
        let result = await routeText("who is running")
        guard case .operatorCommand(.listAgents) = result else {
            Issue.record("Expected .operatorCommand(.listAgents), got \(result)")
            return
        }
    }

    // MARK: - Replay commands

    @Test("operator replay matches replay command")
    func operatorReplay() async {
        let result = await routeText("operator replay")
        guard case .operatorCommand(.replay) = result else {
            Issue.record("Expected .operatorCommand(.replay), got \(result)")
            return
        }
    }

    // MARK: - What did I miss

    @Test("what did i miss matches whatDidIMiss command")
    func whatDidIMiss() async {
        let result = await routeText("what did i miss")
        guard case .operatorCommand(.whatDidIMiss) = result else {
            Issue.record("Expected .operatorCommand(.whatDidIMiss), got \(result)")
            return
        }
    }

    // MARK: - Help

    @Test("operator help matches help command")
    func operatorHelp() async {
        let result = await routeText("operator help")
        guard case .operatorCommand(.help) = result else {
            Issue.record("Expected .operatorCommand(.help), got \(result)")
            return
        }
    }

    // MARK: - Regex: what did <agent> say

    @Test("what did sudo say matches replayAgent with name sudo")
    func whatDidSudoSay() async {
        let result = await routeText("what did sudo say")
        guard case .operatorCommand(.replayAgent(let name)) = result else {
            Issue.record("Expected .operatorCommand(.replayAgent), got \(result)")
            return
        }
        #expect(name == "sudo")
    }

    @Test("what did frontend say matches replayAgent with name frontend")
    func whatDidFrontendSay() async {
        let result = await routeText("what did frontend say")
        guard case .operatorCommand(.replayAgent(let name)) = result else {
            Issue.record("Expected .operatorCommand(.replayAgent), got \(result)")
            return
        }
        #expect(name == "frontend")
    }

    @Test("what did I say does not match replayAgent and falls through to noSessions")
    func whatDidISay() async {
        let result = await routeText("what did I say")
        guard case .noSessions = result else {
            Issue.record("Expected .noSessions, got \(result)")
            return
        }
    }
}

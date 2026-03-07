import Foundation
import OperatorCore

/// A single test scenario result.
internal struct ScenarioResult: Sendable {
    let name: String
    let passed: Bool
    let detail: String
}

// MARK: - Test Scenarios

extension E2EHarness {
    /// Run all 10 test scenarios sequentially and return results.
    func runAllScenarios() async -> [ScenarioResult] {
        var results: [ScenarioResult] = []

        results.append(await testSessionDiscovery())
        results.append(await testTargetedDelivery())
        results.append(await testInactiveTabDelivery())
        results.append(await testSingleSessionRouting())
        results.append(await testKeywordRouting())
        results.append(await testOperatorCommands())
        results.append(await testSessionAffinity())
        results.append(await testSessionRemoval())
        results.append(await testNoSessionsError())
        results.append(await testMCPSpeak())

        return results
    }

    // MARK: 1. SessionDiscovery

    /// Verify ITermBridge.discoverSessions() finds the test panes by TTY.
    func testSessionDiscovery() async -> ScenarioResult {
        let name = "SessionDiscovery"
        do {
            guard let bridge = itermBridge else {
                return ScenarioResult(name: name, passed: false, detail: "itermBridge is nil")
            }

            let sessions = try await bridge.discoverSessions()
            guard let alphaTTY = testTTYs["alpha"], let betaTTY = testTTYs["beta"] else {
                return ScenarioResult(name: name, passed: false, detail: "TTYs not set")
            }

            let foundAlpha = sessions.contains { $0.tty == alphaTTY }
            let foundBeta = sessions.contains { $0.tty == betaTTY }

            guard foundAlpha && foundBeta else {
                return ScenarioResult(
                    name: name,
                    passed: false,
                    detail: "alpha=\(foundAlpha) beta=\(foundBeta) in \(sessions.count) sessions"
                )
            }
            return ScenarioResult(
                name: name,
                passed: true,
                detail: "Found both test panes among \(sessions.count) sessions"
            )
        } catch {
            return ScenarioResult(name: name, passed: false, detail: "discoverSessions() threw: \(error)")
        }
    }

    // MARK: 2. TargetedDelivery

    /// Register 2 sessions, set mock transcription to "tell alpha fix the build",
    /// simulate trigger start/stop, verify pane alpha received "fix the build".
    func testTargetedDelivery() async -> ScenarioResult {
        let name = "TargetedDelivery"
        guard let alphaTTY = testTTYs["alpha"] else {
            return ScenarioResult(name: name, passed: false, detail: "alpha TTY not set")
        }

        await simulateVoiceCommand("tell alpha fix the build")

        let contents = await readPaneContents(tty: alphaTTY)

        guard contents.contains("fix the build") else {
            return ScenarioResult(
                name: name,
                passed: false,
                detail: "Alpha pane did not contain 'fix the build'. Contents: \(contents.suffix(200))"
            )
        }
        return ScenarioResult(name: name, passed: true, detail: "Alpha pane received 'fix the build'")
    }

    // MARK: 3. InactiveTabDelivery

    /// Deliver to tab 2 (beta, inactive) and verify contents.
    func testInactiveTabDelivery() async -> ScenarioResult {
        let name = "InactiveTabDelivery"
        guard let betaTTY = testTTYs["beta"] else {
            return ScenarioResult(name: name, passed: false, detail: "beta TTY not set")
        }

        await simulateVoiceCommand("tell beta check the tests")

        let contents = await readPaneContents(tty: betaTTY)

        guard contents.contains("check the tests") else {
            return ScenarioResult(
                name: name,
                passed: false,
                detail: "Beta pane did not contain message. Contents: \(contents.suffix(200))"
            )
        }
        return ScenarioResult(name: name, passed: true, detail: "Beta pane (inactive tab) received message")
    }

    // MARK: 4. SingleSessionRouting

    /// Deregister one session, send message without target, verify it routes to sole remaining.
    func testSingleSessionRouting() async -> ScenarioResult {
        let name = "SingleSessionRouting"
        guard let betaTTY = testTTYs["beta"],
            let alphaTTY = testTTYs["alpha"],
            let reg = registry
        else {
            return ScenarioResult(name: name, passed: false, detail: "Missing components")
        }

        // Deregister alpha, leaving only beta
        await reg.deregister(tty: alphaTTY)

        await simulateVoiceCommand("run the linter please")

        let contents = await readPaneContents(tty: betaTTY)

        // Re-register alpha for subsequent tests
        await reg.register(name: "alpha", tty: alphaTTY, cwd: "/tmp/e2e-alpha", context: nil)

        guard contents.contains("run the linter please") else {
            return ScenarioResult(
                name: name,
                passed: false,
                detail: "Beta did not receive message. Contents: \(contents.suffix(200))"
            )
        }
        return ScenarioResult(name: name, passed: true, detail: "Single-session bypass routed to beta")
    }

    // MARK: 5. KeywordRouting

    /// Test "hey beta check this" routes to beta.
    func testKeywordRouting() async -> ScenarioResult {
        let name = "KeywordRouting"
        guard let betaTTY = testTTYs["beta"] else {
            return ScenarioResult(name: name, passed: false, detail: "beta TTY not set")
        }

        await simulateVoiceCommand("hey beta look at this error")

        let contents = await readPaneContents(tty: betaTTY)

        guard contents.contains("look at this error") else {
            return ScenarioResult(
                name: name,
                passed: false,
                detail: "Beta pane did not contain message. Contents: \(contents.suffix(200))"
            )
        }
        return ScenarioResult(name: name, passed: true, detail: "Keyword 'hey beta' routed correctly")
    }

    // MARK: 6. OperatorCommands

    /// Test "operator status" triggers spoken response (check MockSpeechManager.spokenMessages).
    func testOperatorCommands() async -> ScenarioResult {
        let name = "OperatorCommands"

        let countBefore = mockSpeechManager.spokenMessages.count

        await simulateVoiceCommand("operator status")

        let newMessages = mockSpeechManager.spokenMessages.dropFirst(countBefore)
        let hasStatusResponse = newMessages.contains { $0.prefix == "Operator" && $0.text.contains("agent") }

        guard hasStatusResponse else {
            let msgs = newMessages.map { "[\($0.prefix)]: \($0.text)" }.joined(separator: "; ")
            return ScenarioResult(
                name: name,
                passed: false,
                detail: "No status response in spoken messages. Got: \(msgs)"
            )
        }
        return ScenarioResult(name: name, passed: true, detail: "Operator spoke status response")
    }

    // MARK: 7. SessionAffinity

    /// Send two messages rapidly, verify second uses affinity (same target, no re-routing).
    func testSessionAffinity() async -> ScenarioResult {
        let name = "SessionAffinity"
        guard let alphaTTY = testTTYs["alpha"] else {
            return ScenarioResult(name: name, passed: false, detail: "alpha TTY not set")
        }

        // First message explicitly targets alpha
        await simulateVoiceCommand("tell alpha first message")

        // Second message without target name -- should go to alpha via affinity
        await simulateVoiceCommand("and also do this second thing")

        let contents = await readPaneContents(tty: alphaTTY)

        guard contents.contains("second thing") else {
            return ScenarioResult(
                name: name,
                passed: false,
                detail: "Alpha did not contain second message. Contents: \(contents.suffix(200))"
            )
        }
        return ScenarioResult(name: name, passed: true, detail: "Affinity routed second message to alpha")
    }

    // MARK: 8. SessionRemoval

    /// Deregister a session via registry, verify it no longer receives messages.
    func testSessionRemoval() async -> ScenarioResult {
        let name = "SessionRemoval"
        guard let alphaTTY = testTTYs["alpha"], let reg = registry else {
            return ScenarioResult(name: name, passed: false, detail: "Missing components")
        }

        // Deregister alpha
        await reg.deregister(tty: alphaTTY)

        // Try to send to alpha -- should fail (session not found)
        await simulateVoiceCommand("tell alpha this should fail")

        let contentsAfter = await readPaneContents(tty: alphaTTY)

        // Re-register alpha for subsequent tests
        await reg.register(name: "alpha", tty: alphaTTY, cwd: "/tmp/e2e-alpha", context: nil)

        // Alpha should NOT have received the new message
        guard !contentsAfter.contains("this should fail") else {
            return ScenarioResult(name: name, passed: false, detail: "Deregistered session still received message")
        }
        return ScenarioResult(name: name, passed: true, detail: "Deregistered session did not receive message")
    }

    // MARK: 9. NoSessionsError

    /// Deregister all, send message, verify "No agents" spoken.
    func testNoSessionsError() async -> ScenarioResult {
        let name = "NoSessionsError"
        guard let reg = registry else {
            return ScenarioResult(name: name, passed: false, detail: "registry is nil")
        }

        // Deregister all sessions
        for (_, tty) in testTTYs {
            await reg.deregister(tty: tty)
        }

        let countBefore = mockSpeechManager.spokenMessages.count

        await simulateVoiceCommand("is anyone there")

        let newMessages = mockSpeechManager.spokenMessages.dropFirst(countBefore)
        let hasNoAgents = newMessages.contains {
            $0.prefix == "Operator" && $0.text.lowercased().contains("no agent")
        }

        // Re-register for cleanup
        for (sessionName, tty) in testTTYs {
            await reg.register(name: sessionName, tty: tty, cwd: "/tmp/e2e-\(sessionName)", context: nil)
        }

        guard hasNoAgents else {
            let msgs = newMessages.map { "[\($0.prefix)]: \($0.text)" }.joined(separator: "; ")
            return ScenarioResult(name: name, passed: false, detail: "Expected 'No agents' message. Got: \(msgs)")
        }
        return ScenarioResult(name: name, passed: true, detail: "Operator spoke 'No agents' message")
    }

    // MARK: 10. MCPSpeak

    /// POST to /speak endpoint, verify MockSpeechManager receives the message.
    func testMCPSpeak() async -> ScenarioResult {
        let name = "MCPSpeak"

        let countBefore = mockSpeechManager.spokenMessages.count

        do {
            try await httpSpeak(message: "Hello from the MCP server", session: "alpha")
        } catch {
            return ScenarioResult(name: name, passed: false, detail: "HTTP /speak failed: \(error)")
        }

        // Wait for audio queue to process
        try? await Task.sleep(nanoseconds: 500_000_000)

        let newMessages = mockSpeechManager.spokenMessages.dropFirst(countBefore)
        let received = newMessages.contains { $0.text.contains("Hello from the MCP server") }

        guard received else {
            let msgs = newMessages.map { "[\($0.prefix)]: \($0.text)" }.joined(separator: "; ")
            return ScenarioResult(
                name: name,
                passed: false,
                detail: "Message not found in spoken messages. Got: \(msgs)"
            )
        }
        return ScenarioResult(name: name, passed: true, detail: "MockSpeechManager received /speak message")
    }
}

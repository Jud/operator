import Foundation
import OperatorCore

// MARK: - Test Helpers (HTTP, Auth, JXA)

extension E2EHarness {
    /// Simulate a complete push-to-talk cycle.
    func simulateVoiceCommand(_ text: String) async {
        mockTranscriber.nextTranscription = text
        mockTrigger.simulateTriggerStart()
        // Small delay for state machine to process triggerStart
        try? await Task.sleep(nanoseconds: 100_000_000)
        mockTrigger.simulateTriggerStop()
        // Wait for transcription + routing + delivery pipeline
        try? await Task.sleep(nanoseconds: 1_500_000_000)
    }

    /// Read the visible contents of a test pane via JXA session.contents().
    func readPaneContents(tty: String) async -> String {
        let script = """
            (function() {
                const app = Application("iTerm2");
                for (const win of app.windows()) {
                    for (const tab of win.tabs()) {
                        for (const sess of tab.sessions()) {
                            if (sess.tty() === '\(tty)') {
                                return sess.contents();
                            }
                        }
                    }
                }
                return "";
            })();
            """

        do {
            return try await runOsascript(script)
        } catch {
            print("[harness] Failed to read pane contents for \(tty): \(error)")
            return ""
        }
    }

    /// Send an HTTP POST /speak request.
    func httpSpeak(message: String, session: String? = nil) async throws {
        var body: [String: String] = ["message": message]
        if let session { body["session"] = session }
        _ = try await httpPost(path: "/speak", body: body)
    }

    /// Send an HTTP POST request to the test server.
    @discardableResult
    func httpPost(path: String, body: [String: String]) async throws -> Data {
        guard let url = URL(string: "http://127.0.0.1:\(testPort)\(path)") else {
            throw E2EError.setupFailed("Invalid URL for path: \(path)")
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("Bearer \(authToken)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse,
            (200..<300).contains(httpResponse.statusCode)
        else {
            let statusCode = (response as? HTTPURLResponse)?.statusCode ?? 0
            let responseBody = String(data: data, encoding: .utf8) ?? ""
            throw E2EError.httpError(path: path, status: statusCode, body: responseBody)
        }
        return data
    }

    /// Read the existing auth token from ~/.operator/token or create one.
    func readOrCreateAuthToken() throws -> String {
        let dir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".operator")

        let tokenFile = dir.appendingPathComponent("token")

        if let data = FileManager.default.contents(atPath: tokenFile.path),
            let token = String(data: data, encoding: .utf8)
        {
            let trimmed = token.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                return trimmed
            }
        }

        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let token = UUID().uuidString

        try token.write(to: tokenFile, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: tokenFile.path)
        return token
    }

    /// Execute a JXA script and return the output.
    func runOsascript(_ script: String) async throws -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
        process.arguments = ["-l", "JavaScript", "-e", script]

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        try process.run()

        let stdoutData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
        let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()

        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            let stderr = String(data: stderrData, encoding: .utf8) ?? ""
            throw E2EError.jxaFailed(status: process.terminationStatus, stderr: stderr)
        }

        return (String(data: stdoutData, encoding: .utf8) ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

// MARK: - Errors

internal enum E2EError: Error, CustomStringConvertible {
    case setupFailed(String)
    case jxaFailed(status: Int32, stderr: String)
    case httpError(path: String, status: Int, body: String)
    case testFailed(scenario: String, reason: String)

    var description: String {
        switch self {
        case .setupFailed(let msg):
            return "Setup failed: \(msg)"

        case .jxaFailed(let status, let stderr):
            return "JXA failed (status \(status)): \(stderr)"

        case .httpError(let path, let status, let body):
            return "HTTP \(path) failed (\(status)): \(body)"

        case .testFailed(let scenario, let reason):
            return "Test '\(scenario)' failed: \(reason)"
        }
    }
}

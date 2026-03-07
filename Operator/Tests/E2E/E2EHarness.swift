import AVFoundation
import Foundation
import OperatorCore

/// E2E test harness that creates real iTerm panes via JXA and validates the full
/// message delivery pipeline using mock voice I/O with real routing and delivery.
///
/// The harness creates a dedicated iTerm window with 2 tabs, captures their TTY
/// paths, boots real daemon components wired with mock transcriber/speechManager,
/// registers test sessions via HTTP, and runs test scenarios that verify text
/// delivery to the real iTerm panes.
@MainActor
internal final class E2EHarness {
    // MARK: - iTerm State

    /// JXA window identifier for the test iTerm window. Used for teardown.
    private var testWindowIndex: Int?

    /// TTY paths for the test panes, keyed by session name ("alpha", "beta").
    private(set) var testTTYs: [String: String] = [:]

    // MARK: - Components

    let mockTranscriber = MockSpeechTranscriber()
    let mockSpeechManager = MockSpeechManager()
    let mockTrigger = MockTriggerSource()

    private(set) var itermBridge: ITermBridge?
    private(set) var voiceManager: VoiceManager?
    private(set) var feedback: AudioFeedback?
    private(set) var audioQueue: AudioQueue?
    private(set) var registry: SessionRegistry?
    private(set) var router: MessageRouter?
    private(set) var stateMachine: StateMachine?
    private(set) var httpServer: OperatorHTTPServer?

    /// Auth token read from ~/.operator/token for HTTP requests.
    private(set) var authToken: String = ""

    /// Port used by the test HTTP server (17420 to avoid conflicting with real daemon).
    let testPort = 17_420

    // MARK: - Setup

    /// Create the test iTerm window with 2 tabs, capture TTYs, boot all components.
    func setUp() async throws {
        print("[harness] Setting up E2E test environment...")

        // Read or create auth token
        authToken = try readOrCreateAuthToken()

        // Create iTerm window with 2 tabs
        try await createTestWindow()

        // Boot daemon components with mock voice I/O
        await bootComponents()

        // Start HTTP server on test port
        try await startHTTPServer()

        // Register test sessions
        try await registerTestSessions()

        print("[harness] Setup complete. TTYs: \(testTTYs)")
    }

    // MARK: - Teardown

    /// Close the test iTerm window and clean up.
    func tearDown() {
        print("[harness] Tearing down...")
        closeTestWindow()
        print("[harness] Teardown complete.")
    }

    // MARK: - iTerm Window Management

    /// Create a new iTerm window with 2 tabs via JXA and capture their TTY paths.
    private func createTestWindow() async throws {
        let script = """
            const app = Application("iTerm2");
            const win = app.createWindowWithDefaultProfile();
            win.createTabWithDefaultProfile();
            delay(1);
            const tabs = win.tabs();
            const result = [];
            for (let i = 0; i < tabs.length; i++) {
                const sess = tabs[i].sessions()[0];
                result.push({ tab: i, tty: sess.tty() });
            }
            JSON.stringify(result);
            """

        let output = try await runOsascript(script)
        guard let data = output.data(using: .utf8),
            let tabs = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]]
        else {
            throw E2EError.setupFailed("Failed to parse iTerm tab info: \(output)")
        }

        guard tabs.count >= 2 else {
            throw E2EError.setupFailed("Expected 2 tabs but got \(tabs.count)")
        }

        testTTYs["alpha"] = tabs[0]["tty"] as? String
        testTTYs["beta"] = tabs[1]["tty"] as? String

        guard let alphaTTY = testTTYs["alpha"], let betaTTY = testTTYs["beta"] else {
            throw E2EError.setupFailed("Failed to capture TTY paths from test tabs")
        }

        // Track the window for teardown -- it's the most recently created window
        // which is at index 0 (iTerm orders windows newest-first)
        testWindowIndex = 0

        print("[harness] Created test window with tabs: alpha=\(alphaTTY), beta=\(betaTTY)")
    }

    /// Close the test iTerm window via JXA.
    private func closeTestWindow() {
        guard testWindowIndex != nil else {
            return
        }

        // Close the window by matching the TTYs we captured
        let alphaTTY = testTTYs["alpha"] ?? ""
        let script = """
            const app = Application("iTerm2");
            for (const win of app.windows()) {
                for (const tab of win.tabs()) {
                    for (const sess of tab.sessions()) {
                        if (sess.tty() === '\(alphaTTY)') {
                            win.close();
                            return "closed";
                        }
                    }
                }
            }
            return "not_found";
            """

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
        process.arguments = ["-l", "JavaScript", "-e", script]
        process.standardOutput = Pipe()
        process.standardError = Pipe()
        try? process.run()
        process.waitUntilExit()

        testWindowIndex = nil
    }

    // MARK: - Component Boot

    /// Instantiate and wire all daemon components with mock voice I/O.
    private func bootComponents() async {
        let bridge = ITermBridge()
        itermBridge = bridge

        let vm = VoiceManager()
        voiceManager = vm

        let fb = AudioFeedback()
        feedback = fb

        let aq = AudioQueue(speechManager: mockSpeechManager, feedback: fb)
        audioQueue = aq
        await aq.setUp()

        let reg = SessionRegistry(itermBridge: bridge, voiceManager: vm)
        registry = reg

        let rt = MessageRouter(registry: reg)
        router = rt

        let sm = StateMachine(
            transcriber: mockTranscriber,
            audioQueue: aq,
            router: rt,
            feedback: fb,
            itermBridge: bridge,
            registry: reg,
            voiceManager: vm,
            waveformPanel: nil,
            speechManager: mockSpeechManager
        )
        stateMachine = sm

        // Wire trigger callbacks to state machine
        mockTrigger.onStart = { [weak sm] in
            sm?.triggerStart()
        }
        mockTrigger.onStop = { [weak sm] in
            sm?.triggerStop()
        }

        print("[harness] Components booted with mock voice I/O")
    }

    // MARK: - HTTP Server

    /// Start the HTTP server on the test port in the background.
    private func startHTTPServer() async throws {
        guard let reg = registry, let aq = audioQueue else {
            return
        }

        let server = OperatorHTTPServer(
            sessionRegistry: reg,
            audioQueue: aq,
            port: testPort
        ) { [weak self] in
            await MainActor.run {
                self?.stateMachine?.currentState.rawValue ?? "unknown"
            }
        }
        httpServer = server

        Task {
            do {
                try await server.start()
            } catch {
                print("[harness] WARNING: HTTP server failed to start: \(error)")
            }
        }

        // Give the server a moment to bind
        try await Task.sleep(nanoseconds: 500_000_000)
        print("[harness] HTTP server started on port \(testPort)")
    }

    // MARK: - Session Registration

    /// Register the test sessions via HTTP POST /register.
    private func registerTestSessions() async throws {
        for (name, tty) in testTTYs {
            try await httpRegister(name: name, tty: tty, cwd: "/tmp/e2e-\(name)")
        }
        print("[harness] Registered \(testTTYs.count) test sessions")
    }

    // MARK: - Test Helpers

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
            """

        do {
            return try await runOsascript(script)
        } catch {
            print("[harness] Failed to read pane contents for \(tty): \(error)")
            return ""
        }
    }

    /// Send an HTTP POST /register request.
    func httpRegister(name: String, tty: String, cwd: String) async throws {
        let body: [String: String] = ["name": name, "tty": tty, "cwd": cwd]
        _ = try await httpPost(path: "/register", body: body)
    }

    /// Send an HTTP POST /speak request.
    func httpSpeak(message: String, session: String? = nil) async throws {
        var body: [String: String] = ["message": message]
        if let session { body["session"] = session }
        _ = try await httpPost(path: "/speak", body: body)
    }

    /// Send an HTTP POST request to the test server.
    @discardableResult
    private func httpPost(path: String, body: [String: String]) async throws -> Data {
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

    // MARK: - Auth Token

    /// Read the existing auth token from ~/.operator/token or create one.
    private func readOrCreateAuthToken() throws -> String {
        let dir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".operator")

        let tokenFile = dir.appendingPathComponent("token")

        if let data = FileManager.default.contents(atPath: tokenFile.path),
            let token = String(data: data, encoding: .utf8) {
            let trimmed = token.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                return trimmed
            }
        }

        // Create token directory and file
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let token = UUID().uuidString

        try token.write(to: tokenFile, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: tokenFile.path)
        return token
    }

    // MARK: - JXA Helper

    /// Execute a JXA script and return the output.
    private func runOsascript(_ script: String) async throws -> String {
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
        case let .setupFailed(msg):
            return "Setup failed: \(msg)"

        case let .jxaFailed(status, stderr):
            return "JXA failed (status \(status)): \(stderr)"

        case let .httpError(path, status, body):
            return "HTTP \(path) failed (\(status)): \(body)"

        case let .testFailed(scenario, reason):
            return "Test '\(scenario)' failed: \(reason)"
        }
    }
}

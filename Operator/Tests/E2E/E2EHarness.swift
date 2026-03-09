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

    /// JXA window identifier for the test iTerm window.
    ///
    /// Used for teardown.
    private var testWindowIndex: Int?

    /// TTY paths for the test panes, keyed by session name ("alpha", "beta").
    private(set) var testTTYs: [String: String] = [:]

    // MARK: - Components

    let mockTranscriber = MockSpeechTranscriber()
    let mockSpeechManager = MockSpeechManager()
    let mockTrigger = MockTriggerSource()

    private(set) var itermBridge: (any TerminalBridge)?
    private(set) var voiceManager: VoiceManager?
    private(set) var feedback: (any AudioFeedbackProviding)?
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
            (function() {
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
            })();
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
        await aq.startListening()

        let reg = SessionRegistry(voiceManager: vm)
        registry = reg

        let rt = MessageRouter(registry: reg)
        router = rt

        let sm = buildStateMachine(
            transcriber: mockTranscriber,
            aq: aq,
            router: rt,
            fb: fb,
            bridge: bridge,
            reg: reg,
            vm: vm
        )
        stateMachine = sm

        // Wire trigger callbacks to state machine
        mockTrigger.onStart = { [weak sm] in
            sm?.triggerStart()
        }
        mockTrigger.onStop = { [weak sm] in
            sm?.triggerStop()
        }
        mockTrigger.onCancel = { [weak sm] in
            sm?.triggerCancel()
        }
        mockTrigger.onDoubleTap = { [weak sm] in
            sm?.triggerDoubleTap()
        }

        print("[harness] Components booted with mock voice I/O")
    }

    /// Build the state machine with bimodal decision engine and dictation delivery.
    private func buildStateMachine(  // swiftlint:disable:this function_parameter_count
        transcriber: MockSpeechTranscriber,
        aq: AudioQueue,
        router: MessageRouter,
        fb: AudioFeedback,
        bridge: ITermBridge,
        reg: SessionRegistry,
        vm: VoiceManager
    ) -> StateMachine {
        let bimodalEngine = BimodalDecisionEngine(
            accessibilityQuery: AccessibilityQueryService(),
            router: router,
            registry: reg
        )
        return StateMachine(
            transcriber: transcriber,
            audioQueue: aq,
            router: router,
            feedback: fb,
            terminalBridge: bridge,
            registry: reg,
            voiceManager: vm,
            waveformPanel: nil,
            speechManager: mockSpeechManager,
            bimodalEngine: bimodalEngine,
            dictationDelivery: DictationDelivery()
        )
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

    /// Register the test sessions directly via the registry actor.
    private func registerTestSessions() async throws {
        guard let reg = registry else {
            throw E2EError.setupFailed("Registry not initialized")
        }
        for (name, tty) in testTTYs {
            await reg.register(name: name, tty: tty, cwd: "/tmp/e2e-\(name)", context: nil)
        }
        print("[harness] Registered \(testTTYs.count) test sessions")
    }
}

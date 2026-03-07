import AVFoundation
import AppKit
import OperatorCore
import ServiceManagement
import Speech
import SwiftUI

/// The main entry point for the Operator macOS application.
@main
public struct OperatorApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self)
    var appDelegate

    /// The app scenes: menu bar extra and settings window.
    public var body: some Scene {
        MenuBarExtra("Operator", systemImage: MenuBarModel.shared.menuBarIcon) {
            MenuBarContentView(model: MenuBarModel.shared)
        }
        .menuBarExtraStyle(.menu)

        Settings {
            SettingsView()
        }
    }

    /// Creates a new OperatorApp instance.
    public init() {}
}

/// Menu bar dropdown content showing state, sessions, and actions.
public struct MenuBarContentView: View {
    /// The model driving menu bar state.
    var model: MenuBarModel

    /// The view body rendering menu bar dropdown items.
    public var body: some View {
        Text("Operator: \(model.currentState)")

        Divider()

        sessionSection

        Divider()

        SettingsLink {
            Text("Settings...")
        }

        Button("Quit Operator") {
            NSApplication.shared.terminate(nil)
        }
        .keyboardShortcut("q")
    }

    @ViewBuilder private var sessionSection: some View {
        if model.sessions.isEmpty {
            Text("No sessions connected")
                .foregroundStyle(.secondary)
        } else {
            Section("Sessions (\(model.sessions.count))") {
                ForEach(model.sessions, id: \.name) { session in
                    Text("\(session.name) - \(session.cwd)")
                }
            }
        }
    }

    /// Creates a new menu bar content view.
    public init(model: MenuBarModel) {
        self.model = model
    }
}

/// Application delegate that bootstraps all Operator daemon components.
@MainActor
public class AppDelegate: NSObject, NSApplicationDelegate {
    private static let logger = Log.logger(for: "AppDelegate")

    private var stateMachine: StateMachine?
    private var waveformPanel: WaveformPanel?
    private var httpServer: OperatorHTTPServer?
    private var trigger: FNKeyTrigger?
    private var voiceManager: VoiceManager?
    private var speechManager: SpeechManager?
    private var audioQueue: AudioQueue?
    private var registry: SessionRegistry?
    private var discoveryService: SessionDiscoveryService?

    /// Called when the application finishes launching to bootstrap all components.
    nonisolated public func applicationDidFinishLaunching(_ notification: Notification) {
        Task { @MainActor in
            await self.bootstrap()
        }
    }

    private func bootstrap() async {
        Self.logger.info("Operator launching")

        setupAuthToken()

        let vm = VoiceManager()
        voiceManager = vm

        let fb = AudioFeedback()
        let sm = SpeechManager()
        speechManager = sm

        await checkPermissions(speechManager: sm, voiceManager: vm)

        let aq = AudioQueue(speechManager: sm, feedback: fb)
        audioQueue = aq
        await aq.startListening()

        let transcriber = SpeechTranscriber()
        let itermBridge: any TerminalBridge = ITermBridge()

        let reg = SessionRegistry(voiceManager: vm)
        registry = reg

        let router = MessageRouter(registry: reg)

        bootstrapHTTPServer(reg: reg, aq: aq, sm: sm, vm: vm)

        let wp = WaveformPanel()
        waveformPanel = wp

        let mbm = MenuBarModel.shared
        mbm.attach(registry: reg)

        stateMachine = StateMachine(
            transcriber: transcriber,
            audioQueue: aq,
            router: router,
            feedback: fb,
            terminalBridge: itermBridge,
            registry: reg,
            voiceManager: vm,
            waveformPanel: wp,
            speechManager: sm,
            menuBarModel: mbm
        )

        bootstrapTrigger()
        bootstrapDiscovery(terminalBridge: itermBridge, reg: reg, aq: aq, vm: vm)

        registerLoginItem()

        sm.speak("Operator is ready.", voice: vm.operatorVoice, prefix: "Operator")
        Self.logger.info("Operator startup complete")
    }

    private func bootstrapHTTPServer(
        reg: SessionRegistry,
        aq: AudioQueue,
        sm: SpeechManager,
        vm: VoiceManager
    ) {
        let server = OperatorHTTPServer(
            sessionRegistry: reg,
            audioQueue: aq,
            port: 7_420
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
                Self.logger.error("HTTP server failed to start: \(error)")
                sm.speak(
                    "Port 7420 is already in use. Another Operator instance may be running.",
                    voice: vm.operatorVoice,
                    prefix: "Operator"
                )
            }
        }
    }

    private func bootstrapTrigger() {
        let fnTrigger = FNKeyTrigger()
        trigger = fnTrigger
        fnTrigger.onStart = { [weak self] in
            self?.stateMachine?.triggerStart()
        }
        fnTrigger.onStop = { [weak self] in
            self?.stateMachine?.triggerStop()
        }
        fnTrigger.onCancel = { [weak self] in
            self?.stateMachine?.triggerCancel()
        }
        fnTrigger.setupEventTap()
    }

    private func bootstrapDiscovery(
        terminalBridge: any TerminalBridge,
        reg: SessionRegistry,
        aq: AudioQueue,
        vm: VoiceManager
    ) {
        let service = SessionDiscoveryService(
            terminalBridge: terminalBridge,
            registry: reg
        ) { name in
            Task {
                await aq.enqueue(
                    AudioQueue.QueuedMessage(
                        sessionName: "Operator",
                        text: "\(name) disconnected.",
                        priority: .urgent,
                        voice: vm.operatorVoice
                    )
                )
            }
        }
        discoveryService = service
        service.startPolling(interval: 10)
    }

    private func setupAuthToken() {
        let dir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".operator")

        let tokenFile = dir.appendingPathComponent("token")

        guard !FileManager.default.fileExists(atPath: tokenFile.path) else {
            Self.logger.info("Auth token already exists at \(tokenFile.path)")
            return
        }

        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

            let token = UUID().uuidString

            try token.write(to: tokenFile, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o600],
                ofItemAtPath: tokenFile.path
            )
            Self.logger.info("Generated auth token at \(tokenFile.path)")
        } catch {
            Self.logger.error("Failed to set up auth token: \(error)")
        }
    }

    private func checkPermissions(speechManager: SpeechManager, voiceManager: VoiceManager) async {
        if !AXIsProcessTrusted() {
            Self.logger.warning("Accessibility permission not granted")
            speechManager.speak(
                "Operator needs Accessibility permission. Opening System Settings.",
                voice: voiceManager.operatorVoice,
                prefix: "Operator"
            )
            if let url = URL(
                string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
            ) {
                NSWorkspace.shared.open(url)
            }
        }

        let micGranted = await AVCaptureDevice.requestAccess(for: .audio)
        if !micGranted {
            Self.logger.warning("Microphone permission not granted")
            speechManager.speak(
                "I can't access the microphone. Check System Settings.",
                voice: voiceManager.operatorVoice,
                prefix: "Operator"
            )
        }

        let speechStatus = await AppleSpeechEngine.requestAuthorization()
        if speechStatus != .authorized {
            Self.logger.warning(
                "Speech recognition permission not granted: \(String(describing: speechStatus))"
            )
        }
    }

    private func registerLoginItem() {
        do {
            try SMAppService.mainApp.register()
            Self.logger.info("Registered as login item")
        } catch {
            Self.logger.warning("Failed to register as login item: \(error)")
        }
    }
}

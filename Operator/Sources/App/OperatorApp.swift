// swiftlint:disable file_length
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

    @Environment(\.openSettings)
    private var openSettings

    /// The view body rendering menu bar dropdown items.
    public var body: some View {
        Text("Operator: \(model.currentState)")

        Divider()

        sessionSection

        Divider()

        Button("Settings...") {
            NSApp.activate(ignoringOtherApps: true)
            openSettings()
        }

        Button("Relaunch") {
            Self.relaunch()
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

    // swiftlint:disable:next type_contents_order
    private static func relaunch() {
        let url = Bundle.main.bundleURL
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/open")
        task.arguments = ["-n", url.path]
        try? task.run()
        NSApp.terminate(nil)
    }

    /// Creates a new menu bar content view.
    public init(model: MenuBarModel) {
        self.model = model
    }
}

// swiftlint:disable type_contents_order type_body_length
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
    private var adaptiveSpeechManager: AdaptiveSpeechManager?
    private var audioQueue: AudioQueue?
    private var registry: SessionRegistry?
    private var discoveryService: SessionDiscoveryService?
    private var modelManager: ModelManager?
    private var adaptiveTranscriptionEngine: AdaptiveTranscriptionEngine?
    private var adaptiveRoutingEngine: AdaptiveRoutingEngine?

    /// Called when the application finishes launching to bootstrap all components.
    nonisolated public func applicationDidFinishLaunching(_ notification: Notification) {
        Task { @MainActor in
            await self.bootstrap()
        }
    }

    // swiftlint:disable:next function_body_length
    private func bootstrap() async {
        Self.logger.info("Operator launching")

        registerDefaults()
        setupAuthToken()
        installCLISymlinks()
        registerPlugin()

        let vm = VoiceManager()
        voiceManager = vm

        let fb = AudioFeedback()
        let mm = ModelManager()
        modelManager = mm
        await mm.scanCachedModels()

        let sm = SpeechManager()
        speechManager = sm

        await checkPermissions(speechManager: sm, voiceManager: vm)

        let engines = bootstrapAdaptiveEngines(mm: mm, sm: sm)
        let aq = AudioQueue(speechManager: engines.adaptiveTTS, feedback: fb)
        audioQueue = aq
        await aq.startListening()

        let itermBridge: any TerminalBridge = ITermBridge()
        let reg = SessionRegistry(voiceManager: vm)
        registry = reg
        let router = MessageRouter(registry: reg, engine: engines.adaptiveRouting)

        bootstrapHTTPServer(reg: reg, aq: aq, sm: sm, vm: vm)
        bootstrapStateMachine(
            engines: engines,
            aq: aq,
            router: router,
            fb: fb,
            itermBridge: itermBridge,
            reg: reg,
            vm: vm
        )
        bootstrapTrigger()
        bootstrapDiscovery(terminalBridge: itermBridge, reg: reg, aq: aq, vm: vm)
        wireFallbackNotifications(
            adaptiveSTT: engines.adaptiveSTT,
            adaptiveTTS: engines.adaptiveTTS,
            adaptiveRouting: engines.adaptiveRouting,
            audioQueue: aq,
            voiceManager: vm
        )
        wireSettingsModel(
            adaptiveSTT: engines.adaptiveSTT,
            adaptiveTTS: engines.adaptiveTTS,
            adaptiveRouting: engines.adaptiveRouting,
            modelManager: mm
        )

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

    /// Grouped return type for the three adaptive engine wrappers.
    private struct AdaptiveEngines {
        let adaptiveSTT: AdaptiveTranscriptionEngine
        let adaptiveTTS: AdaptiveSpeechManager
        let adaptiveRouting: AdaptiveRoutingEngine
    }

    /// Create and configure all adaptive engine wrappers based on user preferences.
    private func bootstrapAdaptiveEngines(
        mm: ModelManager,
        sm: SpeechManager
    ) -> AdaptiveEngines {
        let sttPref = UserDefaults.standard.string(forKey: "sttEngine") ?? "parakeet"
        let ttsPref = UserDefaults.standard.string(forKey: "ttsEngine") ?? "qwen3"
        let routingPref = UserDefaults.standard.string(forKey: "routingEngine") ?? "local"

        let parakeet = ParakeetEngine(modelManager: mm)
        let apple = AppleSpeechEngine()
        let adaptiveSTT = AdaptiveTranscriptionEngine(
            localEngine: parakeet,
            fallbackEngine: apple,
            modelManager: mm,
            preferLocal: sttPref == "parakeet"
        )
        adaptiveTranscriptionEngine = adaptiveSTT

        let qwenTTS = Qwen3TTSSpeechManager(modelManager: mm)
        let adaptiveTTS = AdaptiveSpeechManager(
            localEngine: qwenTTS,
            fallbackEngine: sm,
            modelManager: mm,
            preferLocal: ttsPref == "qwen3"
        )
        adaptiveSpeechManager = adaptiveTTS

        let mlxRouting = MLXRoutingEngine(modelManager: mm)
        let adaptiveRouting = AdaptiveRoutingEngine(
            localEngine: mlxRouting,
            preferLocal: routingPref == "local"
        )
        adaptiveRoutingEngine = adaptiveRouting

        startSelectedModelWarmups(
            adaptiveSTT: adaptiveSTT,
            adaptiveTTS: qwenTTS,
            adaptiveRouting: adaptiveRouting,
            modelManager: mm,
            ttsPref: ttsPref
        )

        return AdaptiveEngines(
            adaptiveSTT: adaptiveSTT,
            adaptiveTTS: adaptiveTTS,
            adaptiveRouting: adaptiveRouting
        )
    }

    private func startSelectedModelWarmups(
        adaptiveSTT: AdaptiveTranscriptionEngine,
        adaptiveTTS: Qwen3TTSSpeechManager,
        adaptiveRouting: AdaptiveRoutingEngine,
        modelManager: ModelManager,
        ttsPref: String
    ) {
        Task {
            await adaptiveSTT.prewarmSelectedLocalModel()
        }

        if ttsPref == "qwen3" {
            Task {
                guard await modelManager.isCached(.tts) else {
                    return
                }
                await adaptiveTTS.prewarm()
            }
        }

        Task {
            await adaptiveRouting.prewarmSelectedLocalModel()
        }
    }

    /// Create the state machine, waveform panel, and menu bar model.
    private func bootstrapStateMachine(  // swiftlint:disable:this function_parameter_count
        engines: AdaptiveEngines,
        aq: AudioQueue,
        router: MessageRouter,
        fb: AudioFeedback,
        itermBridge: any TerminalBridge,
        reg: SessionRegistry,
        vm: VoiceManager
    ) {
        let transcriber = SpeechTranscriber(engine: engines.adaptiveSTT)
        let wp = WaveformPanel()
        waveformPanel = wp

        let mbm = MenuBarModel.shared
        mbm.attach(registry: reg)

        let accessibilityQuery = AccessibilityQueryService()
        let dictDelivery = DictationDelivery()
        let bimodalEngine = BimodalDecisionEngine(
            accessibilityQuery: accessibilityQuery,
            router: router,
            registry: reg
        )

        stateMachine = StateMachine(
            transcriber: transcriber,
            audioQueue: aq,
            router: router,
            feedback: fb,
            terminalBridge: itermBridge,
            registry: reg,
            voiceManager: vm,
            waveformPanel: wp,
            speechManager: engines.adaptiveTTS,
            bimodalEngine: bimodalEngine,
            dictationDelivery: dictDelivery,
            menuBarModel: mbm
        )
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
        fnTrigger.onDoubleTap = { [weak self] in
            self?.stateMachine?.triggerDoubleTap()
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

    private func registerDefaults() {
        UserDefaults.standard.register(defaults: [
            "toggleModeEnabled": true,
            "feedbackSoundsEnabled": true,
            "triggerKeyCode": 63,
            "inputDeviceUID": "",
            "sttEngine": "parakeet",
            "ttsEngine": "qwen3",
            "routingEngine": "local"
        ])
    }

    /// Wire fallback notification callbacks on each adaptive engine.
    ///
    /// When a local engine fails and falls back to its Apple counterpart,
    /// the notification is spoken via the AudioQueue using the Operator voice.
    /// The notification uses whichever TTS engine is currently active (which
    /// may be Apple TTS if the local TTS itself is what failed).
    private func wireFallbackNotifications(  // swiftlint:disable:this function_body_length
        adaptiveSTT: AdaptiveTranscriptionEngine,
        adaptiveTTS: AdaptiveSpeechManager,
        adaptiveRouting: AdaptiveRoutingEngine,
        audioQueue: AudioQueue,
        voiceManager: VoiceManager
    ) {
        let operatorVoice = voiceManager.operatorVoice

        adaptiveSTT.onFallback = { [weak audioQueue] message in
            guard let aq = audioQueue else {
                return
            }
            Task {
                await aq.enqueue(
                    AudioQueue.QueuedMessage(
                        sessionName: "Operator",
                        text: message,
                        priority: .urgent,
                        voice: operatorVoice
                    )
                )
            }
        }

        adaptiveTTS.onFallback = { [weak audioQueue] message in
            guard let aq = audioQueue else {
                return
            }
            Task {
                await aq.enqueue(
                    AudioQueue.QueuedMessage(
                        sessionName: "Operator",
                        text: message,
                        priority: .urgent,
                        voice: operatorVoice
                    )
                )
            }
        }

        adaptiveRouting.onFallback = { [weak audioQueue] message in
            guard let aq = audioQueue else {
                return
            }
            Task {
                await aq.enqueue(
                    AudioQueue.QueuedMessage(
                        sessionName: "Operator",
                        text: message,
                        priority: .urgent,
                        voice: operatorVoice
                    )
                )
            }
        }
    }

    /// Connect the EngineSettingsModel singleton to the adaptive engines
    /// and ModelManager so that Settings UI changes propagate live.
    private func wireSettingsModel(
        adaptiveSTT: AdaptiveTranscriptionEngine,
        adaptiveTTS: AdaptiveSpeechManager,
        adaptiveRouting: AdaptiveRoutingEngine,
        modelManager: ModelManager
    ) {
        let settings = EngineSettingsModel.shared
        settings.attach(modelManager: modelManager)
        settings.onSTTEngineChanged = { [weak adaptiveSTT] useLocal in
            adaptiveSTT?.setUseLocal(useLocal)
        }
        settings.onTTSEngineChanged = { [weak adaptiveTTS] useLocal in
            adaptiveTTS?.setUseLocal(useLocal)
        }
        settings.onRoutingEngineChanged = { [weak adaptiveRouting] useLocal in
            adaptiveRouting?.setUseLocal(useLocal)
        }
    }

    private func installCLISymlinks() {
        let fm = FileManager.default
        let binDir = fm.homeDirectoryForCurrentUser
            .appendingPathComponent(".operator/bin")

        do {
            try fm.createDirectory(at: binDir, withIntermediateDirectories: true)
        } catch {
            Self.logger.error("Failed to create ~/.operator/bin: \(error)")
            return
        }

        let appMacOS = Bundle.main.bundleURL
            .appendingPathComponent("Contents/MacOS")

        for tool in ["operator-mcp"] {
            let source = appMacOS.appendingPathComponent(tool)
            let link = binDir.appendingPathComponent(tool)

            // Remove existing symlink/file so we can update it
            try? fm.removeItem(at: link)

            do {
                try fm.createSymbolicLink(at: link, withDestinationURL: source)
                Self.logger.info("Symlinked \(link.path) -> \(source.path)")
            } catch {
                Self.logger.error("Failed to symlink \(tool): \(error)")
            }
        }
    }

    private func registerPlugin() {
        guard let claudePath = Self.findClaude() else {
            Self.logger.warning("Claude CLI not found — skipping plugin registration")
            return
        }

        Task.detached {
            await Self.runPluginRegistration(claudePath: claudePath)
        }
    }

    private static func findClaude() -> String? {
        for candidate in [
            "/usr/local/bin/claude",
            "/opt/homebrew/bin/claude",
            "\(FileManager.default.homeDirectoryForCurrentUser.path)/.claude/bin/claude"
        ] where FileManager.default.isExecutableFile(atPath: candidate) {
            return candidate
        }
        return nil
    }

    private static func runPluginRegistration(claudePath: String) async {
        let addResult = Self.runProcess(
            claudePath,
            ["plugin", "marketplace", "add", "Jud/operator-plugin"]
        )
        if addResult.status != 0 {
            logger.warning("Failed to add marketplace: \(addResult.stderr)")
        } else {
            logger.info("Plugin marketplace registered")
        }

        let installResult = Self.runProcess(
            claudePath,
            ["plugin", "install", "operator@operator"]
        )
        if installResult.status != 0 {
            logger.warning("Failed to install plugin: \(installResult.stderr)")
        } else {
            logger.info("Plugin installed")
        }
    }

    private static func runProcess(
        _ path: String,
        _ arguments: [String]
    ) -> (status: Int32, stdout: String, stderr: String) {  // swiftlint:disable:this large_tuple
        let process = Process()
        process.executableURL = URL(fileURLWithPath: path)
        process.arguments = arguments

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        do {
            try process.run()
        } catch {
            return (-1, "", error.localizedDescription)
        }

        let stdoutData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
        let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()

        return (
            process.terminationStatus,
            String(data: stdoutData, encoding: .utf8) ?? "",
            String(data: stderrData, encoding: .utf8) ?? ""
        )
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
// swiftlint:enable type_contents_order type_body_length

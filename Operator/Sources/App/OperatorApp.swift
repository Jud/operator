import AVFoundation
import AppKit
import KokoroCoreML
import OperatorCore
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

        Button("Setup Guide...") {
            if let appDelegate = NSApp.delegate as? AppDelegate {
                appDelegate.showSetupGuide()
            }
        }

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
                ForEach(model.sessions) { session in
                    Text("\(session.name) (\(session.voice))")
                }
            }
        }
    }

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

/// Application delegate that bootstraps all Operator daemon components.
@MainActor
public class AppDelegate: NSObject, NSApplicationDelegate {
    static let logger = Log.logger(for: "AppDelegate")
    private var stateMachine: StateMachine?
    private var waveformPanel: WaveformPanel?
    private var httpServer: OperatorHTTPServer?
    private var trigger: FNKeyTrigger?
    private var voiceManager: VoiceManager?
    private var speechManager: (any SpeechManaging)?
    private var audioQueue: AudioQueue?
    private var registry: SessionRegistry?
    private var discoveryService: SessionDiscoveryService?
    private var onboardingWindow: OnboardingWindow?
    private var onboardingWindowDelegate: OnboardingWindowDelegate?
    private var routingFeedbackPanel: RoutingFeedbackPanel?
    private var traceObserver: NSObjectProtocol?

    /// Called when the application finishes launching to bootstrap all components.
    nonisolated public func applicationDidFinishLaunching(_ notification: Notification) {
        Task { @MainActor in
            await self.bootstrap()
        }
    }

    private func bootstrap() async {
        Self.logger.info("Operator launching")
        bootstrapPrerequisites()

        let (ttsManager, vm) = bootstrapTTS()
        voiceManager = vm
        let fb = AudioFeedback()
        speechManager = ttsManager

        if OnboardingViewModel.shouldShowOnboarding() {
            await showOnboardingAndWait()
        } else {
            await checkPermissions(speechManager: ttsManager, voiceManager: vm)
        }

        let aq = AudioQueue(speechManager: ttsManager, feedback: fb)
        audioQueue = aq
        await aq.startListening()

        let terminalBridge: any TerminalBridge = MultiTerminalBridge(
            itermBridge: ITermBridge(),
            ghosttyBridge: GhosttyBridge()
        )
        let reg = SessionRegistry(voiceManager: vm)
        registry = reg
        let traceStore = RoutingTraceStore()
        let router = MessageRouter(registry: reg, traceStore: traceStore)

        bootstrapFeedbackPanel(traceStore: traceStore)
        bootstrapHTTPServer(reg: reg, aq: aq, tts: ttsManager, vm: vm)
        let sttEngine = AppleSpeechEngine()
        let transcriber = bootstrapStateMachine(
            stt: sttEngine,
            tts: ttsManager,
            aq: aq,
            router: router,
            fb: fb,
            terminalBridge: terminalBridge,
            reg: reg,
            vm: vm
        )
        warmUpWhisperKit(transcriber: transcriber)
        bootstrapDiscovery(terminalBridge: terminalBridge, reg: reg, aq: aq, vm: vm)
        wireVoicePreview()
        bootstrapTrigger()
        await aq.enqueue(
            AudioQueue.QueuedMessage(
                sessionName: "Operator",
                text: "ready.",
                priority: .normal,
                voice: vm.operatorVoice
            )
        )
        Self.logger.info("Operator startup complete")
    }

    private func bootstrapTTS() -> (any SpeechManaging, VoiceManager) {
        if KokoroEngine.isDownloaded {
            do {
                let engine = try KokoroEngine(
                    modelDirectory: KokoroEngine.defaultModelDirectory
                )
                let mgr = KokoroSpeechManager(engine: engine)
                Self.logger.info("Using Kokoro CoreML (\(engine.availableVoices.count) voices)")
                return (mgr, VoiceManager(kokoroVoices: engine.availableVoices))
            } catch {
                Self.logger.warning("Kokoro CoreML init failed: \(error). Falling back to Apple TTS.")
            }
        } else {
            Self.logger.info("Kokoro models not downloaded. Starting background download.")
            startKokoroDownload()
        }
        return (SpeechManager(), VoiceManager())
    }

    /// Download Kokoro models in the background and hot-swap when ready.
    private func startKokoroDownload() {
        KokoroDownloadModel.shared.downloadIfNeeded { [weak self] engine in
            guard let self
            else { return }
            let mgr = KokoroSpeechManager(engine: engine)
            let vm = VoiceManager(kokoroVoices: engine.availableVoices)
            self.speechManager = mgr
            self.voiceManager = vm
            self.stateMachine?.replaceSpeechManager(mgr)
            Task {
                await self.audioQueue?.replaceSpeechManager(mgr)
            }
            Self.logger.notice("Hot-swapped to Kokoro CoreML TTS")
        }
    }

    private func bootstrapPrerequisites() {
        registerDefaults()
        setupAuthToken()
        installCLISymlinks()
        registerPlugin()
    }

    private func bootstrapHTTPServer(
        reg: SessionRegistry,
        aq: AudioQueue,
        tts: any SpeechManaging,
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
                await MainActor.run {
                    tts.speak(
                        "Port 7420 is already in use. Another Operator instance may be running.",
                        voice: vm.operatorVoice,
                        prefix: "Operator"
                    )
                }
            }
        }
    }

    /// Create the state machine, waveform panel, and menu bar model.
    @discardableResult
    private func bootstrapStateMachine(
        stt: any TranscriptionEngine,
        tts: any SpeechManaging,
        aq: AudioQueue,
        router: MessageRouter,
        fb: AudioFeedback,
        terminalBridge: any TerminalBridge,
        reg: SessionRegistry,
        vm: VoiceManager
    ) -> OperatorCore.SpeechTranscriber {
        let levelMonitor = AudioLevelMonitor()
        let transcriber = SpeechTranscriber(engine: stt, levelMonitor: levelMonitor)
        let wp = WaveformPanel(levelMonitor: levelMonitor)
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
            terminalBridge: terminalBridge,
            registry: reg,
            voiceManager: vm,
            waveformPanel: wp,
            speechManager: tts,
            bimodalEngine: bimodalEngine,
            dictationDelivery: dictDelivery,
            menuBarModel: mbm
        )
        return transcriber
    }

    /// Load WhisperKit in the background and swap in when ready.
    ///
    /// Downloads the model first if not present.
    private func warmUpWhisperKit(transcriber: OperatorCore.SpeechTranscriber) {
        let preference =
            STTEnginePreference(
                rawValue: UserDefaults.standard.string(forKey: "sttEngine") ?? "auto"
            ) ?? .auto
        let model =
            UserDefaults.standard.string(forKey: "whisperKitModel")
            ?? WhisperKitModelManager.defaultModel

        guard preference != .apple
        else { return }

        Task {
            do {
                // Download if needed.
                var modelPath = WhisperKitModelManager.modelPath(model)
                if modelPath == nil {
                    let folder = try await WhisperKitModelManager.download(variant: model)
                    modelPath = folder.path
                }

                guard let path = modelPath
                else { return }

                let engine = try await WhisperKitEngine.create(modelFolder: path)
                transcriber.replaceEngine(engine)
                Self.logger.notice("STT engine upgraded to WhisperKit (\(model, privacy: .public))")
            } catch {
                Self.logger.warning("WhisperKit warmup failed: \(error). Staying on Apple Speech.")
            }
        }
    }

    private func bootstrapFeedbackPanel(traceStore: RoutingTraceStore) {
        let panel = RoutingFeedbackPanel(traceStore: traceStore)
        routingFeedbackPanel = panel
        traceObserver = NotificationCenter.default.addObserver(
            forName: .routingTraceAppended,
            object: nil,
            queue: .main
        ) { [weak panel] notification in
            guard let trace = notification.userInfo?["trace"] as? RoutingTrace,
                let names = notification.userInfo?["sessionNames"] as? [String]
            else { return }
            panel?.show(trace: trace, sessionNames: names)
        }
    }

    private func wireVoicePreview() {
        let settings = EngineSettingsModel.shared
        settings.onPreviewVoice = { [weak self] in
            guard let vm = self?.voiceManager, let aq = self?.audioQueue else {
                return
            }
            let voice = vm.operatorVoice
            Task {
                await aq.enqueue(
                    AudioQueue.QueuedMessage(
                        sessionName: "Operator",
                        text: "This is how I sound now.",
                        priority: .urgent,
                        voice: voice
                    )
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
        ) { sessionName in
            Task {
                await aq.enqueue(
                    AudioQueue.QueuedMessage(
                        sessionName: sessionName,
                        text: "disconnected.",
                        priority: .urgent,
                        voice: vm.operatorVoice
                    )
                )
            }
        }
        discoveryService = service
        service.startPolling(interval: 10)
    }

    private func checkPermissions(
        speechManager: some SpeechManaging,
        voiceManager: VoiceManager
    ) async {
        let voice = voiceManager.operatorVoice
        if !AXIsProcessTrusted() {
            Self.logger.warning("Accessibility permission not granted")
            speechManager.speak(
                "Operator needs Accessibility permission. Opening System Settings.",
                voice: voice,
                prefix: "Operator"
            )
            SystemSettings.openAccessibility()
        }
        if await !AVCaptureDevice.requestAccess(for: .audio) {
            Self.logger.warning("Microphone permission not granted")
            speechManager.speak(
                "I can't access the microphone. Check System Settings.",
                voice: voice,
                prefix: "Operator"
            )
        }
        let speechStatus = await AppleSpeechEngine.requestAuthorization()
        if speechStatus != .authorized {
            Self.logger.warning(
                "Speech recognition not granted: \(String(describing: speechStatus))"
            )
        }
    }

    private func registerDefaults() {
        UserDefaults.standard.register(defaults: [
            "toggleModeEnabled": true,
            "feedbackSoundsEnabled": true,
            "triggerKeyCode": 63,
            "inputDeviceUID": "",
            "hasCompletedOnboarding": false,
            "sttEngine": STTEnginePreference.auto.rawValue,
            "whisperKitModel": WhisperKitModelManager.defaultModel
        ])
    }

    private func showOnboardingAndWait() async {
        let window = ensureOnboardingWindow()
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            OnboardingViewModel.shared.prepare { continuation.resume() }
            window.show()
        }
        window.close()
        onboardingWindow = nil
        onboardingWindowDelegate = nil
    }

    /// Opens the setup guide window for permission configuration.
    public func showSetupGuide() {
        let vm = OnboardingViewModel.shared
        vm.refreshPermissionStates()
        vm.currentStep = .welcome
        ensureOnboardingWindow().show()
    }

    @discardableResult
    private func ensureOnboardingWindow() -> OnboardingWindow {
        if let existing = onboardingWindow {
            return existing
        }
        let window = OnboardingWindow()
        let delegate = OnboardingWindowDelegate()
        window.delegate = delegate
        onboardingWindow = window
        onboardingWindowDelegate = delegate
        return window
    }
}

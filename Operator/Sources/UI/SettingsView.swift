import SwiftUI

/// Settings view for the Operator application.
///
/// Provides user-configurable preferences for trigger behavior,
/// audio feedback, engine selection, model status, and application
/// information. Uses @AppStorage for automatic UserDefaults persistence.
public struct SettingsView: View {
    // MARK: - Stored Preferences

    /// The key code for the trigger key (default: 63, the Fn/Globe key).
    @AppStorage("triggerKeyCode")
    private var triggerKeyCode: Int = 63

    /// Whether quick-tap toggle mode is enabled for recording.
    @AppStorage("toggleModeEnabled")
    private var toggleModeEnabled: Bool = true

    /// Whether audio feedback sounds are enabled.
    @AppStorage("feedbackSoundsEnabled")
    private var feedbackSoundsEnabled: Bool = true

    /// UID of the selected input (microphone) device.
    ///
    /// Empty string means system default.
    @AppStorage("inputDeviceUID")
    private var inputDeviceUID: String = ""

    /// Speech-to-text engine preference: "parakeet" for local, "apple" for system.
    @AppStorage("sttEngine")
    private var sttEngine: String = "parakeet"

    /// Text-to-speech engine preference: "qwen3" for local, "apple" for system.
    @AppStorage("ttsEngine")
    private var ttsEngine: String = "qwen3"

    /// Routing engine preference: "local" for MLX model, "claude" for Claude CLI.
    @AppStorage("routingEngine")
    private var routingEngine: String = "local"

    /// Speech playback speed multiplier (0.5–2.0, default 1.0).
    @AppStorage("speechRate")
    private var speechRate: Double = 1.0

    /// Voice style instruction for Qwen3-TTS.
    @AppStorage("voiceInstruct")
    private var voiceInstruct: String = "Speak naturally."

    // MARK: - Local State

    @State private var inputDevices: [AudioDevice] = []

    /// The view body rendering the settings form.
    public var body: some View {
        settingsForm
    }

    /// Creates a new settings view.
    public init() {}
}

// MARK: - Sections

extension SettingsView {
    private var settingsForm: some View {
        Form {
            audioDeviceSection
            engineSelectionSection
            voiceSection
            modelStatusSection
            triggerKeySection
            triggerBehaviorSection
            audioSection
            aboutSection
        }
        .formStyle(.grouped)
        .frame(width: 450, height: 680)
        .onAppear { refreshDevices() }
        .onChange(of: sttEngine) { handleSTTEngineChange($1) }
        .onChange(of: ttsEngine) { handleTTSEngineChange($1) }
        .onChange(of: routingEngine) { handleRoutingEngineChange($1) }
        .onChange(of: speechRate) { handleSpeechRateChange($1) }
        .onChange(of: voiceInstruct) { handleVoiceInstructChange($1) }
    }

    private var audioDeviceSection: some View {
        Section("Audio Devices") {
            Picker("Microphone", selection: $inputDeviceUID) {
                Text("System Default").tag("")
                ForEach(inputDevices) { device in
                    Text(device.name).tag(device.uid)
                }
            }

            Button("Refresh Devices") {
                refreshDevices()
            }
            .font(.caption)
        }
    }

    private var engineSelectionSection: some View {
        Section("Speech Engines") {
            sttEnginePicker
            ttsEnginePicker
            routingEnginePicker
        }
    }

    private var sttEnginePicker: some View {
        VStack(alignment: .leading, spacing: 4) {
            Picker("Speech-to-Text", selection: $sttEngine) {
                Text("Parakeet (Local)").tag("parakeet")
                Text("Apple").tag("apple")
            }
            if sttEngine == "parakeet" {
                engineStatusRow(for: .stt)
            }
        }
    }

    private var ttsEnginePicker: some View {
        VStack(alignment: .leading, spacing: 4) {
            Picker("Text-to-Speech", selection: $ttsEngine) {
                Text("Qwen3-TTS (Local)").tag("qwen3")
                Text("Apple").tag("apple")
            }
            if ttsEngine == "qwen3" {
                engineStatusRow(for: .tts)
            }
        }
    }

    private var routingEnginePicker: some View {
        VStack(alignment: .leading, spacing: 4) {
            Picker("Routing", selection: $routingEngine) {
                Text("Local (MLX)").tag("local")
                Text("Claude CLI").tag("claude")
            }
            if routingEngine == "local" {
                engineStatusRow(for: .routing)
            }
        }
    }

    private var voiceSection: some View {
        Section("Voice") {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Speech Rate")
                    Spacer()
                    Text(speechRateLabel)
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                }
                Slider(value: $speechRate, in: 0.5...2.0, step: 0.05)
            }
            Picker("Voice Style", selection: $voiceInstruct) {
                Text("Natural").tag("Speak naturally.")
                Text("Brisk").tag("Speak at a brisk, clear pace.")
                Text("Calm").tag("Speak slowly and calmly.")
                Text("Energetic").tag("Speak with energy and enthusiasm.")
            }
            Text(
                "Speech rate adjusts playback speed. Voice style guides Qwen3-TTS pacing and tone."
            )
            .font(.caption)
            .foregroundStyle(.tertiary)
        }
    }

    private var modelStatusSection: some View {
        Section("Local Models") {
            modelStatusRow(for: .stt)
            modelStatusRow(for: .tts)
            modelStatusRow(for: .routing)
            storageSummaryRow
        }
    }

    private var storageSummaryRow: some View {
        LabeledContent("Storage Used") {
            Text(formatBytes(EngineSettingsModel.shared.totalCacheSizeBytes))
                .foregroundStyle(.secondary)
        }
    }

    private var triggerKeySection: some View {
        Section("Trigger Key") {
            LabeledContent("Current Key") {
                Text(triggerKeyLabel)
                    .foregroundStyle(.secondary)
            }
            Text(
                "The trigger key activates voice recording."
                    + " Fn/Globe (key code 63) is the default."
            )
            .font(.caption)
            .foregroundStyle(.tertiary)
        }
    }

    private var triggerBehaviorSection: some View {
        Section("Trigger Behavior") {
            Toggle("Quick tap toggles recording", isOn: $toggleModeEnabled)
            Text(
                "When enabled, a quick press starts recording and a second press stops. "
                    + "Hold for push-to-talk."
            )
            .font(.caption)
            .foregroundStyle(.tertiary)
        }
    }

    private var audioSection: some View {
        Section("Audio") {
            Toggle("Play feedback sounds", isOn: $feedbackSoundsEnabled)
        }
    }

    private var aboutSection: some View {
        Section("About") {
            LabeledContent("Version") {
                Text(appVersion)
                    .foregroundStyle(.secondary)
            }
            if let repoURL = URL(string: "https://github.com/anthropics/operator") {
                Link("GitHub Repository", destination: repoURL)
            }
        }
    }

    // MARK: - Computed Properties

    private var triggerKeyLabel: String {
        if triggerKeyCode == 63 {
            return "FN/Globe (key code 63)"
        }
        return "Key code \(triggerKeyCode)"
    }

    private var speechRateLabel: String {
        String(format: "%.0f%%", speechRate * 100)
    }

    private var appVersion: String {
        let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "0.1.0"
        let build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "1"
        return "\(version) (\(build))"
    }

    // MARK: - View Builder Methods

    @ViewBuilder
    private func engineStatusRow(for type: ModelManager.ModelType) -> some View {
        let state = EngineSettingsModel.shared.modelStates[type] ?? .notDownloaded
        HStack(spacing: 4) {
            Circle()
                .fill(statusColor(for: state))
                .frame(width: 8, height: 8)
            Text(engineHintText(for: state))
        }
        .font(.caption)
        .foregroundStyle(.secondary)
    }

    @ViewBuilder
    private func modelStatusRow(for type: ModelManager.ModelType) -> some View {
        let state = EngineSettingsModel.shared.modelStates[type] ?? .notDownloaded
        LabeledContent(type.displayName) {
            HStack(spacing: 6) {
                Circle()
                    .fill(statusColor(for: state))
                    .frame(width: 8, height: 8)
                Text(statusLabel(for: state))
                    .foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - Helper Methods

    private func statusColor(for state: ModelManager.ModelState) -> Color {
        switch state {
        case .notDownloaded:
            return .gray

        case .downloading:
            return .orange

        case .ready, .unloaded:
            return .blue

        case .loading:
            return .yellow

        case .loaded:
            return .green

        case .error:
            return .red
        }
    }

    private func engineHintText(for state: ModelManager.ModelState) -> String {
        switch state {
        case .notDownloaded:
            return "Model will download on first use"

        case .downloading(let progress):
            return "Downloading \(Int(progress * 100))%..."

        case .ready, .unloaded:
            return "Model downloaded, loads on demand"

        case .loading:
            return "Loading model..."

        case .loaded:
            return "Model active"

        case .error(let message):
            return "Error: \(message.prefix(60))"
        }
    }

    private func statusLabel(for state: ModelManager.ModelState) -> String {
        switch state {
        case .notDownloaded:
            return "Not Downloaded"

        case .downloading(let progress):
            return "Downloading \(Int(progress * 100))%"

        case .ready:
            return "Downloaded"

        case .loading:
            return "Loading..."

        case .loaded:
            return "Active"

        case .unloaded:
            return "Downloaded"

        case .error(let message):
            return "Error: \(message.prefix(60))"
        }
    }

    private func formatBytes(_ bytes: UInt64) -> String {
        if bytes == 0 {
            return "None"
        }
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter.string(fromByteCount: Int64(bytes))
    }

    // MARK: - Actions

    private func refreshDevices() {
        inputDevices = AudioDeviceManager.inputDevices()
    }

    private func handleSTTEngineChange(_ newValue: String) {
        EngineSettingsModel.shared.onSTTEngineChanged?(newValue == "parakeet")
    }

    private func handleTTSEngineChange(_ newValue: String) {
        EngineSettingsModel.shared.onTTSEngineChanged?(newValue == "qwen3")
    }

    private func handleRoutingEngineChange(_ newValue: String) {
        EngineSettingsModel.shared.onRoutingEngineChanged?(newValue == "local")
    }

    private func handleSpeechRateChange(_ newValue: Double) {
        EngineSettingsModel.shared.onSpeechRateChanged?(Float(newValue))
    }

    private func handleVoiceInstructChange(_ newValue: String) {
        EngineSettingsModel.shared.onVoiceInstructChanged?(newValue)
    }
}

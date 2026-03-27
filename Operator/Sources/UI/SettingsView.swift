import SwiftUI

/// Settings view for the Operator application.
///
/// Provides user-configurable preferences for trigger behavior,
/// audio feedback, and application information.
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
    @AppStorage("inputDeviceUID")
    private var inputDeviceUID: String = ""

    /// Selected WhisperKit model variant.
    @AppStorage("whisperKitModel")
    private var whisperKitModel: String = WhisperKitModelManager.defaultModel

    // MARK: - Local State

    @State private var inputDevices: [AudioDevice] = []
    @State private var isDownloadingModel = false
    @State private var downloadProgress: Double = 0
    @State private var downloadError: String?
    @State private var modelStatus: [String: Bool] = [:]
    @ObservedObject private var kokoroModel = KokoroDownloadModel.shared

    /// The view body rendering the settings form.
    public var body: some View {
        settingsForm
    }

    /// Creates a new settings view.
    public init() {}
}

// MARK: - Sections

extension SettingsView {
    private static let appVersion: String = {
        let info = Bundle.main.infoDictionary
        let version = info?["CFBundleShortVersionString"] as? String ?? "0.1.0"
        let build = info?["CFBundleVersion"] as? String ?? "1"
        return "\(version) (\(build))"
    }()

    private var settingsForm: some View {
        Form {
            audioDeviceSection
            speechRecognitionSection
            kokoroSection
            voiceSection
            triggerKeySection
            triggerBehaviorSection
            audioSection
            aboutSection
        }
        .formStyle(.grouped)
        .frame(width: 450, height: 620)
        .onAppear {
            refreshDevices()
            refreshModelStatus()
        }
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

    private var voiceSection: some View {
        Section("Voice") {
            Button("Preview Voice") {
                EngineSettingsModel.shared.onPreviewVoice?()
            }
        }
    }

    private var speechRecognitionSection: some View {
        Section("Speech Recognition") {
            whisperKitStatusRow

            if let downloadError {
                Text(downloadError)
                    .foregroundStyle(.red)
                    .font(.caption)
            }
        }
    }

    @ViewBuilder private var whisperKitStatusRow: some View {
        if modelStatus[whisperKitModel] == true {
            Label(
                "WhisperKit Small — on-device transcription",
                systemImage: "checkmark.circle.fill"
            )
            .foregroundStyle(.green)
            .font(.caption)
        } else if isDownloadingModel {
            HStack(spacing: 8) {
                ProgressView(value: downloadProgress)
                    .progressViewStyle(.circular)
                    .controlSize(.small)

                Text("Downloading model... \(Int(downloadProgress * 100))%")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }
        } else {
            HStack {
                Text("WhisperKit model not downloaded")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Button("Download") {
                    downloadModel()
                }
            }
        }
    }

    private var kokoroSection: some View {
        Section("Text-to-Speech (Kokoro CoreML)") {
            kokoroStatusView

            if let error = kokoroModel.error {
                Text(error)
                    .foregroundStyle(.red)
                    .font(.caption)
            }

            Text(
                "Kokoro CoreML provides natural-sounding neural TTS. "
                    + "Downloads automatically on first launch (~640 MB). "
                    + "Switches over seamlessly when ready."
            )
            .font(.caption)
            .foregroundStyle(.tertiary)
        }
    }

    @ViewBuilder private var kokoroStatusView: some View {
        if kokoroModel.isDownloaded {
            Label("Models downloaded", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)

            Button("Preview Voice") {
                EngineSettingsModel.shared.onPreviewVoice?()
            }
        } else if kokoroModel.isDownloading {
            HStack {
                Text("Downloading...")
                ProgressView(value: kokoroModel.progress)
                    .frame(width: 120)
                Text("\(Int(kokoroModel.progress * 100))%")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        } else {
            Button("Download Kokoro Models") {
                kokoroModel.downloadIfNeeded()
            }
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
                Text(Self.appVersion)
                    .foregroundStyle(.secondary)
            }
            if let url = URL(string: "https://github.com/anthropics/operator") {
                Link("GitHub Repository", destination: url)
            }
        }
    }

    private var triggerKeyLabel: String {
        if triggerKeyCode == 63 {
            return "FN/Globe (key code 63)"
        }
        return "Key code \(triggerKeyCode)"
    }

    private func refreshDevices() {
        inputDevices = AudioDeviceManager.inputDevices()
    }

    private func refreshModelStatus() {
        modelStatus = Dictionary(
            uniqueKeysWithValues: WhisperKitModelManager.availableModels.map {
                ($0.variant, WhisperKitModelManager.modelAvailable($0.variant))
            }
        )
    }

    private func downloadModel() {
        isDownloadingModel = true
        downloadProgress = 0
        downloadError = nil
        let model = whisperKitModel
        Task {
            do {
                try await WhisperKitModelManager.download(variant: model) { progress in
                    Task { @MainActor in
                        downloadProgress = progress.fractionCompleted
                    }
                }
                isDownloadingModel = false
                refreshModelStatus()
            } catch {
                downloadError = "Download failed: \(error.localizedDescription)"
                isDownloadingModel = false
            }
        }
    }
}

import AVFoundation
import Foundation

/// View model managing the first-launch onboarding flow, including step
/// navigation, permission tracking, and bootstrap completion signaling.
@MainActor
@Observable
public final class OnboardingViewModel {
    // MARK: - Subtypes

    /// The sequential steps of the onboarding flow.
    public enum Step: Int, CaseIterable, Sendable {
        case welcome = 0
        case permissions = 1
        case accessibility = 2
        case modelDownload = 3
        case howItWorks = 4
        case done = 5
    }

    // MARK: - Constants

    /// UserDefaults key for tracking onboarding completion.
    private static let completedKey = "hasCompletedOnboarding"

    // MARK: - Type Properties

    /// Shared singleton instance used by AppDelegate and MenuBarContentView.
    public static let shared = OnboardingViewModel()

    private static let logger = Log.logger(for: "OnboardingViewModel")

    // MARK: - Instance Properties

    /// The currently displayed onboarding step.
    public var currentStep: Step = .welcome

    /// Whether microphone access has been granted.
    public var microphoneGranted: Bool = false

    /// Whether Accessibility (AXIsProcessTrusted) has been granted.
    public var accessibilityGranted: Bool = false

    /// Whether microphone permission has been requested during this session.
    public var microphoneRequested: Bool = false

    /// Whether the WhisperKit model is currently being downloaded.
    public var modelDownloading: Bool = false

    /// Progress of the WhisperKit model download (0.0–1.0).
    public var modelDownloadProgress: Double = 0

    /// Whether the WhisperKit model has been downloaded successfully.
    public var modelDownloaded: Bool = false

    /// Error message if the model download failed.
    public var modelDownloadError: String?

    private var completionHandler: (() -> Void)?
    private var axPollingTimer: Timer?
    private var autoAdvanceTask: Task<Void, Never>?
    private var downloadTask: Task<Void, Never>?

    // MARK: - Initialization

    /// Creates a new onboarding view model.
    public init() {}

    // MARK: - Type Methods

    /// Check whether onboarding should be shown.
    public static func shouldShowOnboarding() -> Bool {
        let hasCompleted = UserDefaults.standard.bool(forKey: Self.completedKey)
        if hasCompleted {
            logger.info("Onboarding already completed (flag is true)")
            return false
        }

        let micAuthorized = AVCaptureDevice.authorizationStatus(for: .audio) == .authorized
        let axTrusted = AXIsProcessTrusted()

        if micAuthorized && axTrusted {
            UserDefaults.standard.set(true, forKey: Self.completedKey)
            logger.info("All permissions already granted, skipping onboarding")
            return false
        }

        logger.info(
            "Onboarding needed: mic=\(micAuthorized), ax=\(axTrusted)"
        )
        return true
    }

    // MARK: - Public Methods

    /// Store the bootstrap completion handler.
    public func prepare(completion: @escaping () -> Void) {
        completionHandler = completion
        Self.logger.info("Onboarding prepared with completion handler")
    }

    /// Advance to the next step.
    public func advance() {
        guard let nextRaw = Step(rawValue: currentStep.rawValue + 1) else {
            return
        }
        let previous = currentStep
        currentStep = nextRaw
        Self.logger.info("Advanced from \(previous.rawValue) to \(nextRaw.rawValue)")
    }

    /// Return to the previous step.
    public func goBack() {
        guard let previousRaw = Step(rawValue: currentStep.rawValue - 1) else {
            return
        }
        currentStep = previousRaw
        Self.logger.info("Went back to step \(previousRaw.rawValue)")
    }

    /// Request Microphone permission.
    public func requestPermissions() {
        Task {
            await self.requestMicrophonePermission()
        }
    }

    /// Open the Accessibility pane in System Settings.
    public func openAccessibilitySettings() {
        SystemSettings.openAccessibility()
        Self.logger.info("Opened Accessibility System Settings")
    }

    /// Start polling `AXIsProcessTrusted()` every 1 second.
    public func startAccessibilityPolling() {
        stopAccessibilityPolling()

        if AXIsProcessTrusted() {
            accessibilityGranted = true
            Self.logger.info("Accessibility already granted (pre-poll check)")
            return
        }

        axPollingTimer = Timer.scheduledTimer(
            withTimeInterval: 1.0,
            repeats: true
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.checkAccessibilityStatus()
            }
        }
        Self.logger.info("Started Accessibility polling")
    }

    /// Stop the accessibility permission polling timer.
    public func stopAccessibilityPolling() {
        axPollingTimer?.invalidate()
        axPollingTimer = nil
        autoAdvanceTask?.cancel()
        autoAdvanceTask = nil
    }

    /// Start downloading the WhisperKit model.
    public func startModelDownload() {
        guard !modelDownloading, !modelDownloaded
        else { return }

        let model = WhisperKitModelManager.defaultModel

        // Already downloaded (e.g. user re-visited step).
        if WhisperKitModelManager.modelAvailable(model) {
            modelDownloaded = true
            modelDownloadProgress = 1.0
            Self.logger.info("WhisperKit model already available, skipping download")
            return
        }

        modelDownloading = true
        modelDownloadError = nil
        modelDownloadProgress = 0

        weak var weakSelf = self
        let callback: @Sendable (Progress) -> Void = { progress in
            Task { @MainActor in
                weakSelf?.modelDownloadProgress = progress.fractionCompleted
            }
        }

        downloadTask = Task {
            do {
                Self.logger.info("Downloading WhisperKit model: \(model)")
                try await WhisperKitModelManager.download(variant: model, progressCallback: callback)
                self.modelDownloading = false
                self.modelDownloaded = true
                Self.logger.info("WhisperKit model download complete")
            } catch {
                guard !Task.isCancelled
                else { return }
                self.modelDownloading = false
                self.modelDownloadError = error.localizedDescription
                Self.logger.warning("WhisperKit model download failed: \(error)")
            }
        }
    }

    /// Complete the onboarding flow.
    public func completeOnboarding() {
        stopAccessibilityPolling()
        UserDefaults.standard.set(true, forKey: Self.completedKey)
        Self.logger.info("Onboarding completed, flag set in UserDefaults")

        let handler = completionHandler
        completionHandler = nil
        handler?()
    }

    /// Refresh permission states by reading current OS status.
    public func refreshPermissionStates() {
        let mic = AVCaptureDevice.authorizationStatus(for: .audio) == .authorized
        let ax = AXIsProcessTrusted()
        microphoneGranted = mic
        accessibilityGranted = ax
        Self.logger.info(
            "Refreshed permission states: mic=\(mic), ax=\(ax)"
        )
    }

    // MARK: - Private Methods

    private func requestMicrophonePermission() async {
        let currentStatus = AVCaptureDevice.authorizationStatus(for: .audio)
        if currentStatus == .authorized {
            microphoneGranted = true
            microphoneRequested = true
            Self.logger.info("Microphone already authorized, skipping request")
            return
        }

        microphoneRequested = true
        let granted = await AVCaptureDevice.requestAccess(for: .audio)
        microphoneGranted = granted
        Self.logger.info("Microphone permission \(granted ? "granted" : "denied")")
    }

    private func checkAccessibilityStatus() {
        guard !accessibilityGranted else {
            stopAccessibilityPolling()
            return
        }

        guard AXIsProcessTrusted() else {
            return
        }

        accessibilityGranted = true
        // Stop the timer but keep autoAdvanceTask slot free for the delayed advance.
        axPollingTimer?.invalidate()
        axPollingTimer = nil
        Self.logger.info("Accessibility permission detected as granted")

        autoAdvanceTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(1))
            guard !Task.isCancelled,
                let self,
                self.currentStep == .accessibility
            else { return }
            self.advance()
            Self.logger.info("Auto-advanced from accessibility step")
        }
    }
}

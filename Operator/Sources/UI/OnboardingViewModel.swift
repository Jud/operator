import AVFoundation
import Foundation
import Speech

/// View model managing the first-launch onboarding flow, including step
/// navigation, permission tracking, and bootstrap completion signaling.
///
/// Follows the same `@MainActor @Observable` singleton pattern as
/// ``MenuBarModel`` and ``EngineSettingsModel``. Wired during application
/// bootstrap via ``prepare(completion:)`` and signaled back via
/// ``completeOnboarding()``.
@MainActor
@Observable
public final class OnboardingViewModel {
    // MARK: - Subtypes

    /// The five sequential steps of the onboarding flow.
    public enum Step: Int, CaseIterable, Sendable {
        case welcome = 0
        case permissions = 1
        case accessibility = 2
        case howItWorks = 3
        case done = 4
    }

    // MARK: - Type Properties

    /// Shared singleton instance used by AppDelegate and MenuBarContentView.
    public static let shared = OnboardingViewModel()

    private static let logger = Log.logger(for: "OnboardingViewModel")

    // MARK: - Instance Properties

    /// The currently displayed onboarding step.
    public var currentStep: Step = .welcome

    /// Whether microphone access has been granted.
    public var microphoneGranted: Bool = false

    /// Whether speech recognition access has been granted.
    public var speechRecognitionGranted: Bool = false

    /// Whether Accessibility (AXIsProcessTrusted) has been granted.
    public var accessibilityGranted: Bool = false

    /// Whether microphone permission has been requested during this session.
    public var microphoneRequested: Bool = false

    /// Whether speech recognition permission has been requested during this session.
    public var speechRecognitionRequested: Bool = false

    private var completionHandler: (() -> Void)?
    private var axPollingTimer: Timer?
    private var autoAdvanceWorkItem: DispatchWorkItem?

    // MARK: - Initialization

    /// Creates a new onboarding view model.
    public init() {}

    // MARK: - Type Methods

    /// Check whether onboarding should be shown.
    ///
    /// Returns `true` when `hasCompletedOnboarding` is `false` AND at least
    /// one of the three permissions is not granted. If all permissions are
    /// already granted, sets the flag to `true` and returns `false`.
    public static func shouldShowOnboarding() -> Bool {
        let hasCompleted = UserDefaults.standard.bool(forKey: "hasCompletedOnboarding")
        if hasCompleted {
            logger.info("Onboarding already completed (flag is true)")
            return false
        }

        let micAuthorized = AVCaptureDevice.authorizationStatus(for: .audio) == .authorized
        let speechAuthorized = SFSpeechRecognizer.authorizationStatus() == .authorized
        let axTrusted = AXIsProcessTrusted()

        if micAuthorized && speechAuthorized && axTrusted {
            UserDefaults.standard.set(true, forKey: "hasCompletedOnboarding")
            logger.info("All permissions already granted, skipping onboarding")
            return false
        }

        logger.info(
            "Onboarding needed: mic=\(micAuthorized), speech=\(speechAuthorized), ax=\(axTrusted)"
        )
        return true
    }

    // MARK: - Public Methods

    /// Store the bootstrap completion handler.
    ///
    /// Called by AppDelegate with a `withCheckedContinuation` resume closure
    /// so the bootstrap can await onboarding completion.
    ///
    /// - Parameter completion: Closure invoked when onboarding finishes.
    public func prepare(completion: @escaping () -> Void) {
        completionHandler = completion
        Self.logger.info("Onboarding prepared with completion handler")
    }

    /// Advance to the next step.
    ///
    /// Does nothing if already on `.done`.
    public func advance() {
        guard let nextRaw = Step(rawValue: currentStep.rawValue + 1) else {
            return
        }
        let previous = currentStep
        currentStep = nextRaw
        Self.logger.info("Advanced from \(previous.rawValue) to \(nextRaw.rawValue)")
    }

    /// Return to the previous step.
    ///
    /// Does nothing if already on `.welcome`.
    public func goBack() {
        guard let previousRaw = Step(rawValue: currentStep.rawValue - 1) else {
            return
        }
        currentStep = previousRaw
        Self.logger.info("Went back to step \(previousRaw.rawValue)")
    }

    /// Request Microphone and Speech Recognition permissions sequentially.
    ///
    /// Microphone is requested first. Only after it resolves (granted or denied)
    /// is Speech Recognition requested. Already-granted permissions are skipped
    /// (no duplicate OS dialogs).
    public func requestPermissions() {
        Task { @MainActor in
            await self.requestMicrophonePermission()
            await self.requestSpeechRecognitionPermission()
        }
    }

    /// Open the Accessibility pane in System Settings.
    public func openAccessibilitySettings() {
        SystemSettings.openAccessibility()
        Self.logger.info("Opened Accessibility System Settings")
    }

    /// Start polling `AXIsProcessTrusted()` every 1 second.
    ///
    /// When the permission is detected as granted, sets `accessibilityGranted`
    /// to `true`, stops polling, and auto-advances to the next step after a
    /// 1-second visual confirmation delay.
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
        autoAdvanceWorkItem?.cancel()
        autoAdvanceWorkItem = nil
    }

    /// Complete the onboarding flow.
    ///
    /// Sets the `hasCompletedOnboarding` flag in UserDefaults, invokes the
    /// stored completion handler (resuming bootstrap), and cleans up.
    public func completeOnboarding() {
        stopAccessibilityPolling()
        UserDefaults.standard.set(true, forKey: "hasCompletedOnboarding")
        Self.logger.info("Onboarding completed, flag set in UserDefaults")

        let handler = completionHandler
        completionHandler = nil
        handler?()
    }

    /// Refresh permission states by reading current OS status without triggering request dialogs.
    ///
    /// Used when re-opening onboarding via the "Setup Guide..." menu item
    /// to show checkmarks for already-granted permissions.
    public func refreshPermissionStates() {
        let mic = AVCaptureDevice.authorizationStatus(for: .audio) == .authorized
        let speech = SFSpeechRecognizer.authorizationStatus() == .authorized
        let ax = AXIsProcessTrusted()
        microphoneGranted = mic
        speechRecognitionGranted = speech
        accessibilityGranted = ax
        Self.logger.info(
            "Refreshed permission states: mic=\(mic), speech=\(speech), ax=\(ax)"
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

    private func requestSpeechRecognitionPermission() async {
        let currentStatus = SFSpeechRecognizer.authorizationStatus()
        if currentStatus == .authorized {
            speechRecognitionGranted = true
            speechRecognitionRequested = true
            Self.logger.info("Speech recognition already authorized, skipping request")
            return
        }

        speechRecognitionRequested = true
        let status = await AppleSpeechEngine.requestAuthorization()
        speechRecognitionGranted = status == .authorized
        Self.logger.info(
            "Speech recognition permission: \(String(describing: status))"
        )
    }

    private func checkAccessibilityStatus() {
        guard !accessibilityGranted else {
            stopAccessibilityPolling()
            return
        }

        if AXIsProcessTrusted() {
            accessibilityGranted = true
            axPollingTimer?.invalidate()
            axPollingTimer = nil
            Self.logger.info("Accessibility permission detected as granted")

            let workItem = DispatchWorkItem { [weak self] in
                Task { @MainActor [weak self] in
                    guard let self, self.currentStep == .accessibility else {
                        return
                    }
                    self.advance()
                    Self.logger.info("Auto-advanced from accessibility step")
                }
            }
            autoAdvanceWorkItem = workItem
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0, execute: workItem)
        }
    }
}

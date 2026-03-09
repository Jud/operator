import SwiftUI

/// Five-step onboarding interface that walks new users through macOS permissions
/// and teaches core Operator interaction mechanics.
///
/// Driven by ``OnboardingViewModel/currentStep`` to display step-specific content.
/// Each step uses SF Symbols for visual identity and consistent navigation
/// controls at the bottom of the view.
public struct OnboardingView: View {
    var model: OnboardingViewModel

    public var body: some View {
        VStack(spacing: 0) {
            stepContent
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .padding(.horizontal, 40)
                .padding(.top, 32)

            Divider()

            navigationBar
                .padding(.horizontal, 24)
                .padding(.vertical, 16)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    public init(model: OnboardingViewModel) {
        self.model = model
    }
}

// MARK: - Step Router

extension OnboardingView {
    @ViewBuilder private var stepContent: some View {
        switch model.currentStep {
        case .welcome:
            welcomeStep

        case .permissions:
            permissionsStep

        case .accessibility:
            accessibilityStep

        case .howItWorks:
            howItWorksStep

        case .done:
            doneStep
        }
    }
}

// MARK: - Step 1: Welcome

extension OnboardingView {
    private var welcomeStep: some View {
        VStack(spacing: 20) {
            Image(systemName: "waveform.circle")
                .font(.system(size: 56))
                .foregroundStyle(.tint)

            Text("Welcome to Operator")
                .font(.title)
                .fontWeight(.semibold)

            VStack(spacing: 12) {
                Text(
                    "Operator is a voice layer for Claude Code. "
                        + "It lets you speak to your coding sessions and hear responses "
                        + "read aloud."
                )
                .multilineTextAlignment(.center)

                Text(
                    "Press and hold the FN key to talk, then release to send your "
                        + "message to the right session."
                )
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
            }
            .font(.body)
        }
    }
}

// MARK: - Step 2: Permissions

extension OnboardingView {
    private var permissionsStep: some View {
        VStack(spacing: 20) {
            Image(systemName: "mic.circle")
                .font(.system(size: 56))
                .foregroundStyle(.tint)

            Text("Voice Permissions")
                .font(.title)
                .fontWeight(.semibold)

            Text("Operator needs microphone and speech recognition access to hear your voice commands.")
                .font(.body)
                .multilineTextAlignment(.center)

            VStack(spacing: 12) {
                permissionRow(
                    label: "Microphone",
                    granted: model.microphoneGranted,
                    requested: model.microphoneRequested
                )

                permissionRow(
                    label: "Speech Recognition",
                    granted: model.speechRecognitionGranted,
                    requested: model.speechRecognitionRequested
                )
            }
            .padding(.top, 4)

            if !model.microphoneRequested {
                Button("Request Permissions") {
                    model.requestPermissions()
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .onAppear {
            if !model.microphoneRequested {
                model.requestPermissions()
            }
        }
    }

    @ViewBuilder
    private func permissionRow(label: String, granted: Bool, requested: Bool) -> some View {
        HStack(spacing: 10) {
            permissionStatusIcon(granted: granted, requested: requested)
                .frame(width: 20)

            Text(label)
                .font(.body)

            Spacer()
        }
        .padding(.horizontal, 40)
    }

    @ViewBuilder
    private func permissionStatusIcon(granted: Bool, requested: Bool) -> some View {
        if granted {
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(.green)
        } else if requested {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
        } else {
            Image(systemName: "circle.dashed")
                .foregroundStyle(.secondary)
        }
    }
}

// MARK: - Step 3: Accessibility

extension OnboardingView {
    private var accessibilityStep: some View {
        VStack(spacing: 20) {
            Image(systemName: "hand.raised.circle")
                .font(.system(size: 56))
                .foregroundStyle(.tint)

            Text("Accessibility")
                .font(.title)
                .fontWeight(.semibold)

            Text(
                "Operator needs Accessibility permission to detect the FN key for "
                    + "push-to-talk. This permission must be granted manually in System Settings."
            )
            .font(.body)
            .multilineTextAlignment(.center)

            accessibilityStatusRow

            if !model.accessibilityGranted {
                Button("Open System Settings") {
                    model.openAccessibilitySettings()
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .onAppear { model.startAccessibilityPolling() }
        .onDisappear { model.stopAccessibilityPolling() }
    }

    @ViewBuilder private var accessibilityStatusRow: some View {
        HStack(spacing: 10) {
            permissionStatusIcon(granted: model.accessibilityGranted, requested: true)
            Text(model.accessibilityGranted ? "Accessibility granted" : "Not yet granted")
                .foregroundStyle(model.accessibilityGranted ? .green : .secondary)
        }
        .font(.body)
    }
}

// MARK: - Step 4: How It Works

extension OnboardingView {
    private var howItWorksStep: some View {
        VStack(spacing: 20) {
            Image(systemName: "keyboard")
                .font(.system(size: 56))
                .foregroundStyle(.tint)

            Text("How It Works")
                .font(.title)
                .fontWeight(.semibold)

            controlsList

            howItWorksNotes
        }
    }

    private var controlsList: some View {
        VStack(alignment: .leading, spacing: 14) {
            controlRow(
                icon: "hand.tap",
                title: "Hold FN",
                description: "Press and hold to record, release to send"
            )
            controlRow(
                icon: "arrow.triangle.2.circlepath",
                title: "Tap FN",
                description: "Quick tap to toggle recording on or off"
            )
            controlRow(
                icon: "escape",
                title: "Escape",
                description: "Cancel the current recording"
            )
        }
        .padding(.horizontal, 20)
    }

    private var howItWorksNotes: some View {
        VStack(spacing: 6) {
            Text("Operator lives in your menu bar at the top of the screen.")
                .font(.callout)
                .foregroundStyle(.secondary)

            Text("Access settings from the menu bar dropdown.")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
        .multilineTextAlignment(.center)
    }

    @ViewBuilder
    private func controlRow(icon: String, title: String, description: String) -> some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.title3)
                .frame(width: 28, alignment: .center)
                .foregroundStyle(.tint)

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.body)
                    .fontWeight(.medium)
                Text(description)
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        }
    }
}

// MARK: - Step 5: Done

extension OnboardingView {
    private var doneStep: some View {
        VStack(spacing: 20) {
            Image(systemName: "checkmark.circle")
                .font(.system(size: 56))
                .foregroundStyle(.green)

            Text("You're All Set")
                .font(.title)
                .fontWeight(.semibold)

            Text("Operator is ready to go. Press the FN key any time to start talking.")
                .font(.body)
                .multilineTextAlignment(.center)

            VStack(spacing: 8) {
                permissionSummaryRow(label: "Microphone", granted: model.microphoneGranted)
                permissionSummaryRow(label: "Speech Recognition", granted: model.speechRecognitionGranted)
                permissionSummaryRow(label: "Accessibility", granted: model.accessibilityGranted)
            }
            .padding(.top, 4)
        }
    }

    @ViewBuilder
    private func permissionSummaryRow(label: String, granted: Bool) -> some View {
        HStack(spacing: 8) {
            permissionStatusIcon(granted: granted, requested: true)
                .frame(width: 18)

            Text(label)
                .font(.callout)

            Spacer()
        }
        .padding(.horizontal, 60)
    }
}

// MARK: - Navigation Bar

extension OnboardingView {
    private var navigationBar: some View {
        HStack {
            if model.currentStep != .welcome {
                Button("Back") {
                    model.goBack()
                }
            }

            Spacer()

            if model.currentStep == .done {
                Button("Get Started") {
                    model.completeOnboarding()
                }
                .buttonStyle(.borderedProminent)
            } else {
                Button("Continue") {
                    model.advance()
                }
                .buttonStyle(.borderedProminent)
            }
        }
    }
}

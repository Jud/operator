import AppKit
import SwiftUI

/// Standard macOS window hosting the onboarding SwiftUI view.
///
/// Follows the ``WaveformPanel`` pattern of using `NSWindow` + `NSHostingView`
/// for programmatic window control from the AppDelegate. Unlike WaveformPanel,
/// this uses standard window chrome (title bar, close button).
///
/// The close button (red X) triggers ``OnboardingViewModel/completeOnboarding()``
/// to ensure the bootstrap continuation is resumed even if the user dismisses
/// the window without reaching the final step.
@MainActor
public final class OnboardingWindow: NSWindow {
    private static let logger = Log.logger(for: "OnboardingWindow")

    /// Creates the onboarding window centered on screen at 500x400.
    public init() {
        super.init(
            contentRect: NSRect(x: 0, y: 0, width: 500, height: 400),
            styleMask: [.titled, .closable],
            backing: .buffered,
            defer: false
        )
        self.title = "Operator Setup"
        self.isReleasedWhenClosed = false
        self.contentView = NSHostingView(
            rootView: OnboardingView(model: OnboardingViewModel.shared)
        )
        self.center()

        Self.logger.debug("OnboardingWindow initialized")
    }

    /// Activate the app and bring the onboarding window to front.
    public func show() {
        NSApp.activate(ignoringOtherApps: true)
        self.makeKeyAndOrderFront(nil)
        self.center()

        Self.logger.info("OnboardingWindow shown")
    }
}

/// Delegate that handles the window close button by completing onboarding,
/// ensuring the bootstrap continuation is resumed.
@MainActor
public final class OnboardingWindowDelegate: NSObject, NSWindowDelegate {
    private static let logger = Log.logger(for: "OnboardingWindowDelegate")

    /// Completes onboarding when the user closes the window via the close button.
    nonisolated public func windowWillClose(_ notification: Notification) {
        Task { @MainActor in
            OnboardingViewModel.shared.completeOnboarding()
            Self.logger.info("Window closed via close button, onboarding completed")
        }
    }
}

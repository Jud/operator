import AppKit

/// Helpers for opening specific System Settings panes.
public enum SystemSettings {
    private static let accessibilityURL = URL(
        string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
    )!

    /// Open the Accessibility pane in System Settings.
    public static func openAccessibility() {
        NSWorkspace.shared.open(accessibilityURL)
    }
}

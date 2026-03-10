import AppKit

/// Helpers for opening specific System Settings panes.
public enum SystemSettings {
    /// Open the Accessibility pane in System Settings.
    public static func openAccessibility() {
        guard
            let url = URL(
                string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
            )
        else { return }
        NSWorkspace.shared.open(url)
    }
}

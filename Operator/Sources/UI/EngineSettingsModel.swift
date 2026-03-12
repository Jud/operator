import Foundation

/// Observable model for Settings UI callbacks.
///
/// Minimal singleton providing action callbacks wired during bootstrap.
@MainActor
@Observable
public final class EngineSettingsModel {
    /// Shared singleton instance used by both SettingsView and AppDelegate.
    public static let shared = EngineSettingsModel()

    /// Called to play a short voice preview.
    public var onPreviewVoice: (() -> Void)?

    /// Creates a new engine settings model.
    public init() {}
}

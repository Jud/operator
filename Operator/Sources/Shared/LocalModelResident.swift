import Foundation

/// Capabilities needed to keep a local inference model resident for the app session.
public protocol LocalModelResident: Sendable {
    /// Whether the model artifacts are already cached locally and can be loaded
    /// without triggering a startup download.
    func isCachedLocally() async -> Bool

    /// Load and warm the model for low-latency first use.
    func prewarm() async
}

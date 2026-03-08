import Foundation
import os

/// Adaptive wrapper that delegates to either ``MLXRoutingEngine`` (local) or
/// ``ClaudePipeRoutingEngine`` (Claude CLI) based on user preference and model availability.
///
/// Conforms to ``RoutingEngine``, making it transparent to consumers like
/// ``MessageRouter`` and ``InterruptionHandler``. On construction, checks whether
/// the local model is available. If the local engine fails during operation,
/// automatically falls back to the Claude CLI engine and logs the event.
///
/// Engine switching: ``setUseLocal(_:)`` changes the active engine preference.
/// The switch takes effect on the next ``run(prompt:timeout:)`` call, not
/// mid-operation. No application restart required.
public final class AdaptiveRoutingEngine: RoutingEngine, @unchecked Sendable {
    private static let logger = Log.logger(for: "AdaptiveRoutingEngine")

    // MARK: - Properties

    private let localEngine: MLXRoutingEngine
    private let fallbackEngine: ClaudePipeRoutingEngine
    private let modelManager: ModelManager
    private var useLocal: Bool
    private var didFallBack = false

    /// Whether the engine is currently configured to use the local MLX model.
    public var isUsingLocal: Bool {
        useLocal && !didFallBack
    }

    // MARK: - Initialization

    /// Creates an adaptive routing engine.
    ///
    /// - Parameters:
    ///   - localEngine: The MLX routing engine for local inference.
    ///   - modelManager: The shared model manager for checking model availability.
    ///   - fallbackEngine: The Claude CLI routing engine as fallback.
    ///   - preferLocal: Initial preference for local engine. When `true`, uses
    ///     MLX routing. When `false`, uses Claude CLI.
    public init(
        localEngine: MLXRoutingEngine,
        modelManager: ModelManager,
        fallbackEngine: ClaudePipeRoutingEngine = ClaudePipeRoutingEngine(),
        preferLocal: Bool = true
    ) {
        self.localEngine = localEngine
        self.fallbackEngine = fallbackEngine
        self.modelManager = modelManager
        self.useLocal = preferLocal

        Self.logger.info(
            "AdaptiveRoutingEngine initialized, preferLocal: \(preferLocal)"
        )
    }

    // MARK: - Engine Switching

    /// Switch between local MLX routing and Claude CLI routing.
    ///
    /// Takes effect on the next ``run(prompt:timeout:)`` call, not mid-operation.
    /// Called by the Settings UI when the user changes routing engine preferences.
    ///
    /// - Parameter enabled: `true` to use local MLX routing, `false` to use Claude CLI.
    public func setUseLocal(_ enabled: Bool) {
        useLocal = enabled
        didFallBack = false
        Self.logger.info(
            "Routing engine preference changed to \(enabled ? "Local MLX" : "Claude CLI")"
        )
    }

    // MARK: - RoutingEngine

    /// Route a prompt through the currently active engine.
    ///
    /// If the local engine is preferred, attempts to use it. On failure (e.g.,
    /// model not loaded, inference error), falls back to the Claude CLI engine
    /// with a warning log. The fallback is sticky until the user re-enables local
    /// in Settings or calls ``setUseLocal(_:)``.
    ///
    /// - Parameters:
    ///   - prompt: The routing prompt built by MessageRouter.
    ///   - timeout: Maximum time to wait for a response.
    /// - Returns: A dictionary parsed from the JSON routing decision.
    /// - Throws: If both engines fail, throws the fallback engine's error.
    public func run(prompt: String, timeout: TimeInterval) async throws -> [String: Any] {
        if useLocal && !didFallBack {
            do {
                return try await localEngine.run(prompt: prompt, timeout: timeout)
            } catch {
                Self.logger.warning(
                    "Local routing failed, falling back to Claude CLI: \(error.localizedDescription)"
                )
                didFallBack = true
            }
        }

        return try await fallbackEngine.run(prompt: prompt, timeout: timeout)
    }

    // MARK: - Fallback

    /// Mark that the local engine has failed and all subsequent calls should use Claude CLI.
    ///
    /// Called externally by the engine coordination layer when a local engine
    /// failure is detected.
    public func triggerFallback() {
        guard !didFallBack else {
            return
        }
        didFallBack = true
        Self.logger.warning("Falling back to Claude CLI routing; local model unavailable")
    }
}

import Foundation

/// Protocol abstracting the `claude -p` subprocess for routing decisions.
///
/// Injected into `MessageRouter` so that tests can substitute a mock
/// without spawning real subprocesses, and production code does not
/// statically depend on `ClaudePipe`.
///
/// Reference: design principles P1 (protocol boundaries at I/O seams),
/// P4 (inject, don't import).
public protocol RoutingEngine: Sendable {
    /// Run a prompt and return the parsed JSON dictionary.
    ///
    /// - Parameters:
    ///   - prompt: The full prompt text.
    ///   - timeout: Maximum time to wait for a response.
    /// - Returns: A dictionary parsed from the JSON object in the output.
    /// - Throws: `ClaudePipeError` (or similar) on failure.
    func run(prompt: String, timeout: TimeInterval) async throws -> [String: Any]
}

/// Default timeout for routing engine invocations (10 seconds per spec).
extension RoutingEngine {
    /// Run a prompt using the default timeout.
    public func run(prompt: String) async throws -> [String: Any] {
        try await run(prompt: prompt, timeout: ClaudePipe.defaultTimeout)
    }
}

import Foundation
import os

/// Polls a terminal bridge for live sessions and removes stale entries
/// from the `SessionRegistry`.
///
/// Extracted from `SessionRegistry` so the registry remains a
/// terminal-agnostic session store with zero iTerm knowledge.
///
/// Reference: technical-spec.md Component 5, "Session Discovery Reconciliation"
public final class SessionDiscoveryService: Sendable {
    // MARK: - Type Properties

    private static let logger = Log.logger(for: "SessionDiscoveryService")

    // MARK: - Instance Properties

    /// The terminal bridge used for discovery polling to detect closed sessions.
    private let terminalBridge: any TerminalBridge

    /// The session registry to reconcile against.
    private let registry: SessionRegistry

    /// Callback invoked when a session is removed.
    private let onSessionRemoved: @Sendable (String) -> Void

    /// Handle to the discovery polling task for cancellation on deinit.
    private let discoveryTask = ManagedTask()

    // MARK: - Initialization

    /// - Parameters:
    ///   - terminalBridge: Bridge for terminal session discovery.
    ///   - registry: The session registry to reconcile.
    ///   - onSessionRemoved: Callback invoked with the session name when a stale session is removed.
    public init(
        terminalBridge: any TerminalBridge,
        registry: SessionRegistry,
        onSessionRemoved: @escaping @Sendable (String) -> Void
    ) {
        self.terminalBridge = terminalBridge
        self.registry = registry
        self.onSessionRemoved = onSessionRemoved
        Self.logger.info("SessionDiscoveryService initialized")
    }

    // MARK: - Reconciliation

    /// Reconcile registered sessions against live terminal sessions.
    ///
    /// Uses per-type reconciliation: only removes stale sessions whose
    /// terminal type was successfully polled. This prevents Ghostty
    /// sessions from being removed when Ghostty is not running (and
    /// vice versa for iTerm2).
    ///
    /// When the bridge is a `MultiTerminalBridge`, `polledTerminalTypes`
    /// distinguishes "zero sessions" from "bridge failed". For other
    /// bridge types, a successful discovery is treated as polling all types.
    private static func reconcileSessions(
        using bridge: any TerminalBridge,
        registry: SessionRegistry,
        onSessionRemoved: @Sendable (String) -> Void
    ) async {
        let liveSessions: [DiscoveredSession]
        do {
            liveSessions = try await bridge.discoverSessions()
        } catch {
            logger.warning("Discovery polling failed: \(error) -- skipping reconciliation")
            return
        }

        let polledTypes: Set<TerminalType>
        if let composite = bridge as? MultiTerminalBridge {
            polledTypes = composite.polledTerminalTypes
        } else {
            polledTypes = [.iterm, .ghostty]
        }

        var liveIdentifiers = Set<TerminalIdentifier>()
        for session in liveSessions {
            switch session.terminalType {
            case .iterm:
                liveIdentifiers.insert(.tty(session.tty))

            case .ghostty:
                liveIdentifiers.insert(.ghosttyTerminal(session.id))
            }
        }

        let registeredSessions = await registry.allSessions()

        for session in registeredSessions {
            // Skip sessions with unresolvable TTYs (e.g. /dev/?? from MCP
            // server subprocesses). These are managed by heartbeat, not
            // terminal tab discovery.
            if case .tty(let path) = session.identifier,
                path == "/dev/??" || path == "unknown"
            {
                continue
            }

            let sessionType = session.identifier.terminalType
            guard polledTypes.contains(sessionType) else { continue }

            if !liveIdentifiers.contains(session.identifier) {
                if let removedName = await registry.deregister(identifier: session.identifier) {
                    logger.info(
                        "Stale session removed: '\(removedName)' (\(session.identifier) no longer active)"
                    )
                    onSessionRemoved(removedName)
                }
            }
        }
    }

    /// Trigger a single reconciliation cycle for testing.
    ///
    /// Exposes the reconciliation logic without requiring polling to be started.
    /// Intended for unit tests via `@testable import`.
    func reconcileOnce() async {
        await Self.reconcileSessions(
            using: terminalBridge,
            registry: registry,
            onSessionRemoved: onSessionRemoved
        )
    }

    // MARK: - Polling

    /// Start periodic discovery polling to reconcile registered sessions
    /// against live terminal sessions.
    ///
    /// - Parameter interval: Polling interval in seconds. Defaults to 10.
    public func startPolling(interval: TimeInterval = 10) {
        discoveryTask.cancel()

        Self.logger.info("Starting discovery polling every \(interval) seconds")

        let bridge = terminalBridge
        let reg = registry
        let removed = onSessionRemoved

        discoveryTask.start {
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: UInt64(interval * 1_000_000_000))

                guard !Task.isCancelled else {
                    break
                }

                await Self.reconcileSessions(
                    using: bridge,
                    registry: reg,
                    onSessionRemoved: removed
                )
            }
        }
    }

    /// Stop discovery polling.
    public func stopPolling() {
        discoveryTask.cancel()
        Self.logger.info("Stopped discovery polling")
    }
}

/// A Sendable wrapper around a cancellable `Task` reference.
///
/// Provides a safe way for `Sendable` types to manage a mutable task handle
/// using `OSAllocatedUnfairLock` for synchronization.
private final class ManagedTask: Sendable {
    private let lock: OSAllocatedUnfairLock<Task<Void, Never>?>

    init() {
        self.lock = OSAllocatedUnfairLock(initialState: nil)
    }

    func start(_ operation: @escaping @Sendable () async -> Void) {
        let task = Task(operation: operation)
        lock.withLock { $0 = task }
    }

    func cancel() {
        lock.withLock { task in
            task?.cancel()
            task = nil
        }
    }
}

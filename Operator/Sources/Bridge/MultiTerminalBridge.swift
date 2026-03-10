import Foundation
import os

/// Composite bridge that aggregates session discovery from multiple terminal
/// bridges and routes message delivery to the correct bridge based on the
/// session's terminal identifier type.
///
/// Discovery runs both sub-bridges concurrently. If one bridge fails (e.g.,
/// the terminal is not running), the other's results are still returned.
/// After each discovery call, `polledTerminalTypes` records which terminal
/// types were successfully polled -- callers can use this to distinguish
/// "zero sessions" from "bridge failed" during reconciliation.
///
/// Delivery switches on the `TerminalIdentifier` variant:
/// `.tty` routes to the iTerm bridge, `.ghosttyTerminal` routes to Ghostty.
public final class MultiTerminalBridge: @unchecked Sendable, TerminalBridge {
    private static let logger = Log.logger(for: "MultiTerminalBridge")

    private let itermBridge: any TerminalBridge
    private let ghosttyBridge: any TerminalBridge
    private let polledTypes = OSAllocatedUnfairLock<Set<TerminalType>>(initialState: [])

    /// The set of terminal types that were successfully polled during the
    /// last `discoverSessions()` call.
    ///
    /// Empty before the first call. Callers use this to distinguish
    /// "zero sessions" from "bridge failed" during reconciliation.
    public var polledTerminalTypes: Set<TerminalType> {
        polledTypes.withLock { $0 }
    }

    /// Creates a composite bridge wrapping iTerm2 and Ghostty bridges.
    ///
    /// - Parameters:
    ///   - itermBridge: The bridge for iTerm2 session discovery and delivery.
    ///   - ghosttyBridge: The bridge for Ghostty session discovery and delivery.
    public init(itermBridge: any TerminalBridge, ghosttyBridge: any TerminalBridge) {
        self.itermBridge = itermBridge
        self.ghosttyBridge = ghosttyBridge
    }

    /// Discover sessions from both iTerm2 and Ghostty, aggregating results.
    ///
    /// Runs both sub-bridge discoveries concurrently. If one bridge throws
    /// (e.g., the terminal is not installed or not running), its result is
    /// treated as an empty list and the other bridge's sessions are still
    /// returned. The `polledTerminalTypes` property is updated to reflect
    /// which bridges succeeded.
    ///
    /// - Returns: Combined array of discovered sessions from all bridges.
    public func discoverSessions() async throws -> [DiscoveredSession] {
        async let itermResult = safeDiscover(itermBridge, type: .iterm)
        async let ghosttyResult = safeDiscover(ghosttyBridge, type: .ghostty)

        let (itermOutcome, ghosttyOutcome) = await (itermResult, ghosttyResult)

        var polled = Set<TerminalType>()
        var all: [DiscoveredSession] = []

        if let sessions = itermOutcome {
            polled.insert(.iterm)
            all.append(contentsOf: sessions)
        }
        if let sessions = ghosttyOutcome {
            polled.insert(.ghostty)
            all.append(contentsOf: sessions)
        }

        let polledSnapshot = polled
        polledTypes.withLock { $0 = polledSnapshot }

        Self.logger.info(
            "Discovery complete: \(all.count) session(s) from \(polled.count) terminal type(s)"
        )
        return all
    }

    /// Deliver text to the correct terminal bridge based on the identifier type.
    ///
    /// `.tty` identifiers are routed to the iTerm bridge; `.ghosttyTerminal`
    /// identifiers are routed to the Ghostty bridge.
    ///
    /// - Parameters:
    ///   - identifier: The terminal identifier for the target session.
    ///   - text: The message text to deliver.
    /// - Returns: `true` if the session was found and text was delivered.
    /// - Throws: Bridge-specific errors from the underlying terminal bridge.
    public func writeToSession(identifier: TerminalIdentifier, text: String) async throws -> Bool {
        switch identifier {
        case .tty:
            return try await itermBridge.writeToSession(identifier: identifier, text: text)

        case .ghosttyTerminal:
            return try await ghosttyBridge.writeToSession(identifier: identifier, text: text)
        }
    }

    /// Attempt discovery on a single bridge, returning nil on failure.
    private func safeDiscover(
        _ bridge: any TerminalBridge,
        type: TerminalType
    ) async -> [DiscoveredSession]? {
        do {
            return try await bridge.discoverSessions()
        } catch {
            Self.logger.debug(
                "Discovery skipped for \(type.rawValue): \(error.localizedDescription)"
            )
            return nil
        }
    }
}

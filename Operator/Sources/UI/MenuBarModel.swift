import Foundation

/// A connected session for display in the menu bar.
public struct MenuBarSession: Identifiable, Equatable {
    /// Unique identifier (TTY path).
    public let id: String
    /// Display name of the session.
    public let name: String
    /// Working directory of the session.
    public let cwd: String
    /// Voice preset name assigned to this session.
    public let voice: String
}

/// View model for the menu bar extra, tracking Operator state and connected sessions.
///
/// Updated by StateMachine on state transitions and polls SessionRegistry
/// for connected session information on a periodic timer.
@MainActor
@Observable
public final class MenuBarModel {
    // MARK: - Type Properties

    /// Shared singleton instance used by both the SwiftUI scene and AppDelegate.
    public static let shared = MenuBarModel()

    // MARK: - Instance Properties

    /// The current state label shown in the menu bar.
    public var currentState: String = "Idle"

    /// Connected sessions for display.
    public var sessions: [MenuBarSession] = []

    /// SF Symbol name for the current state.
    public var menuBarIcon: String {
        switch currentState {
        case "LISTENING":
            "mic.circle.fill"

        case "TRANSCRIBING", "ROUTING", "DELIVERING":
            "ellipsis.circle"

        case "ERROR":
            "exclamationmark.circle"

        case "CLARIFYING":
            "questionmark.circle"

        default:
            "waveform.circle"
        }
    }

    /// Weak reference to the session registry for polling.
    private var registry: SessionRegistry?

    /// Timer for polling session state.
    private var pollTimer: Timer?

    // MARK: - Initialization

    /// Creates a new menu bar model.
    public init() {}

    // MARK: - Public Methods

    /// Attach the session registry for periodic polling.
    ///
    /// Starts a 3-second timer that refreshes session data from the registry.
    /// - Parameter registry: The session registry actor to poll.
    public func attach(registry: SessionRegistry) {
        self.registry = registry
        startPolling()
    }

    /// Update the displayed state label from a state transition.
    ///
    /// Called by StateMachine whenever it transitions to a new state.
    /// - Parameter state: The raw value of the new OperatorState.
    public func update(state: String) {
        currentState = state
    }

    /// Stop polling and clean up the timer.
    public func stopPolling() {
        pollTimer?.invalidate()
        pollTimer = nil
    }

    // MARK: - Private

    private func startPolling() {
        pollTimer = Timer.scheduledTimer(withTimeInterval: 3.0, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                await self?.refreshSessions()
            }
        }
    }

    private func refreshSessions() async {
        guard let registry else {
            return
        }

        let allSessions = await registry.allSessions()
        let updated = allSessions.map {
            MenuBarSession(id: $0.name, name: $0.name, cwd: $0.cwd, voice: $0.voice.displayName)
        }

        // Only mutate if data changed — avoids triggering a SwiftUI re-render
        // that could race with a concurrent state transition update.
        if updated != sessions {
            sessions = updated
        }
    }
}

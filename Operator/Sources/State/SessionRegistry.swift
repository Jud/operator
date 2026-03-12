import Foundation

/// Thread-safe registry of active Claude Code sessions.
///
/// Manages session lifecycle: registration, context updates, voice assignment,
/// and state queries for the HTTP API. Sessions are keyed by TerminalIdentifier
/// (TTY path for iTerm2, terminal ID for Ghostty). Terminal-agnostic --
/// discovery polling is handled by `SessionDiscoveryService`.
///
/// Concurrency: Swift actor ensures all state mutations are serialized without
/// manual locks, per technical-spec.md Key Design Decision #10.
///
/// Reference: technical-spec.md Component 5; design.md Section 3.1.10
public actor SessionRegistry {
    // MARK: - Type Properties

    private static let logger = Log.logger(for: "SessionRegistry")

    // MARK: - Instance Properties

    /// Registered sessions keyed by TerminalIdentifier.
    private var sessions: [TerminalIdentifier: SessionState] = [:]

    /// Cache of TTY-to-Ghostty terminal ID mappings for terminal ID resolution.
    private var ttyToGhosttyId: [String: String] = [:]

    /// Voice manager for assigning distinct voices to each agent.
    private let voiceManager: VoiceManager

    /// Get the count of registered sessions.
    public var sessionCount: Int {
        sessions.count
    }

    // MARK: - Initialization

    /// - Parameter voiceManager: Manager for assigning per-agent voices with pitch variation.
    public init(voiceManager: VoiceManager) {
        self.voiceManager = voiceManager
        Self.logger.info("SessionRegistry initialized")
    }

    // MARK: - Registration

    /// Register a new Claude Code session with the given metadata.
    ///
    /// Assigns a distinct voice (from VoiceManager) with a unique pitch multiplier
    /// so the user can auditorily distinguish agents. Rejects the reserved name
    /// "operator" (case-insensitive) per technical-spec.md Component 4.
    ///
    /// - Parameters:
    ///   - name: Human-readable session name (e.g., "sudo", "frontend").
    ///   - identifier: The typed terminal identifier for this session.
    ///   - cwd: Working directory of the session.
    ///   - context: Initial context summary describing what the session is working on.
    /// - Returns: `true` if registration succeeded, `false` if the name is reserved.
    @discardableResult
    public func register(name: String, identifier: TerminalIdentifier, cwd: String, context: String?) -> Bool {
        if name.lowercased() == "operator" {
            Self.logger.warning("Rejected registration: name 'operator' is reserved")
            return false
        }

        if let existing = sessions[identifier] {
            var updated = existing
            updated.name = name
            updated.cwd = cwd
            if let context { updated.context = context }
            updated.lastActivity = Date()
            sessions[identifier] = updated
            Self.logger.info("Re-registered session '\(name)' at \(identifier)")
            return true
        }

        let voice = voiceManager.nextAgentVoice()

        let state = SessionState(
            name: name,
            identifier: identifier,
            cwd: cwd,
            context: context ?? "",
            recentMessages: [],
            status: .idle,
            lastActivity: Date(),
            voice: voice,
            pitchMultiplier: 1.0
        )
        sessions[identifier] = state
        Self.logger.info("Registered session '\(name)' at \(identifier)")
        return true
    }

    /// Convenience: register an iTerm2 session by TTY path.
    @discardableResult
    public func register(name: String, tty: String, cwd: String, context: String?) -> Bool {
        register(name: name, identifier: .tty(tty), cwd: cwd, context: context)
    }

    /// Remove a session by terminal identifier.
    ///
    /// - Parameter identifier: The terminal identifier of the session to remove.
    /// - Returns: The name of the removed session, or nil if not found.
    @discardableResult
    public func deregister(identifier: TerminalIdentifier) -> String? {
        guard let state = sessions.removeValue(forKey: identifier) else {
            Self.logger.debug("Deregister called for unknown identifier: \(identifier)")
            return nil
        }
        clearGhosttyMapping(for: identifier)
        Self.logger.info("Deregistered session '\(state.name)' from \(identifier)")
        return state.name
    }

    /// Convenience: remove an iTerm2 session by TTY path.
    @discardableResult
    public func deregister(tty: String) -> String? {
        deregister(identifier: .tty(tty))
    }

    // MARK: - Context Updates

    /// Update the routing context for a session identified by TTY.
    ///
    /// Resolves the TTY to its current terminal identifier (accounting for
    /// Ghostty sessions that have been re-keyed after terminal ID resolution).
    ///
    /// - Parameters:
    ///   - tty: TTY path identifying the session to update.
    ///   - summary: Updated context summary.
    ///   - recentMessages: Recent conversation messages for routing context.
    public func updateContext(tty: String, summary: String, recentMessages: [SessionMessage]) {
        let id = resolveIdentifier(forTTY: tty)
        guard var state = sessions[id] else {
            Self.logger.warning("updateContext called for unregistered TTY: \(tty)")
            return
        }
        state.context = summary
        state.recentMessages = recentMessages
        state.lastActivity = Date()
        sessions[id] = state
        Self.logger.debug("Updated context for \(id)")
    }

    // MARK: - Voice Lookup

    /// Look up the assigned voice for a session by name.
    ///
    /// - Parameter session: The session name to look up.
    /// - Returns: The VoiceDescriptor assigned to the session.
    public func voiceFor(session: String?) -> VoiceDescriptor {
        session.flatMap(self.session(named:))?.voice ?? voiceManager.defaultAgentVoice
    }

    /// Look up the pitch multiplier for a session by name.
    ///
    /// - Parameter session: The session name to look up.
    /// - Returns: The pitch multiplier, defaulting to 1.0 if not found.
    public func pitchFor(session: String?) -> Float {
        session.flatMap(self.session(named:))?.pitchMultiplier ?? 1.0
    }

    // MARK: - Hook Handlers

    /// Handle a Claude Code session-start hook.
    ///
    /// Registers or re-registers a session using the hook payload's session ID and TTY.
    /// If a session with the same TTY already exists, it is updated with the new session ID.
    /// For Ghostty sessions, returns `needsTerminalId: true` when no TTY-to-terminal-ID
    /// mapping exists yet, signaling the MCP server to perform the title-marker handshake.
    ///
    /// - Parameters:
    ///   - sessionId: The Claude Code session identifier.
    ///   - tty: The TTY device path for the session.
    ///   - cwd: The working directory of the session.
    ///   - terminalType: The terminal emulator type. Defaults to `.iterm`.
    /// - Returns: Tuple indicating success and whether terminal ID resolution is needed.
    @discardableResult
    public func handleSessionStart(
        sessionId: String,
        tty: String,
        cwd: String,
        terminalType: TerminalType = .iterm
    ) -> (ok: Bool, needsTerminalId: Bool) {
        let id = resolveIdentifier(forTTY: tty)

        if var existing = sessions[id] {
            existing.sessionId = sessionId
            existing.lastActivity = Date()
            sessions[id] = existing
            Self.logger.info("Hook session-start: updated session ID for \(id)")
        } else {
            let voice = voiceManager.nextAgentVoice()
            let state = SessionState(
                name: sessionId,
                identifier: id,
                cwd: cwd,
                context: "",
                recentMessages: [],
                status: .idle,
                lastActivity: Date(),
                voice: voice,
                pitchMultiplier: 1.0,
                sessionId: sessionId
            )
            sessions[id] = state
            Self.logger.info("Hook session-start: registered new session \(sessionId) at \(id)")
        }

        let needsTerminalId = terminalType == .ghostty && ttyToGhosttyId[tty] == nil
        return (ok: true, needsTerminalId: needsTerminalId)
    }

    /// Handle a Claude Code stop hook.
    ///
    /// Updates the session's last assistant message as context and marks it active.
    /// Resolves the TTY to the current terminal identifier for re-keyed Ghostty sessions.
    ///
    /// - Parameters:
    ///   - sessionId: The Claude Code session identifier.
    ///   - tty: The TTY device path for the session.
    ///   - lastAssistantMessage: The last message from the assistant, if any.
    public func handleStop(sessionId: String, tty: String, lastAssistantMessage: String?) {
        let id = resolveIdentifier(forTTY: tty)
        guard var state = sessions[id] else {
            Self.logger.warning("Hook stop: no session found for TTY \(tty)")
            return
        }
        state.sessionId = sessionId
        state.lastActivity = Date()
        if let message = lastAssistantMessage, !message.isEmpty {
            state.context = message
        }
        sessions[id] = state
        Self.logger.info("Hook stop: updated session at \(id)")
    }

    /// Handle a Claude Code session-end hook.
    ///
    /// Removes the session identified by TTY from the registry. Resolves the TTY
    /// to the current terminal identifier for re-keyed Ghostty sessions.
    ///
    /// - Parameters:
    ///   - sessionId: The Claude Code session identifier.
    ///   - tty: The TTY device path for the session.
    /// - Returns: The name of the removed session, or nil if not found.
    @discardableResult
    public func handleSessionEnd(sessionId: String, tty: String) -> String? {
        let id = resolveIdentifier(forTTY: tty)
        guard let state = sessions.removeValue(forKey: id) else {
            Self.logger.debug("Hook session-end: no session found for TTY \(tty)")
            return nil
        }
        clearGhosttyMapping(for: id)
        Self.logger.info("Hook session-end: removed session '\(state.name)' from \(id)")
        return state.name
    }

    // MARK: - Terminal ID Resolution

    /// Re-key a Ghostty session from its temporary TTY key to its resolved terminal ID.
    ///
    /// Called when the MCP server completes the title-marker handshake and reports
    /// the Ghostty terminal ID corresponding to a TTY path. The session is removed
    /// from the `.tty(path)` key and re-inserted under `.ghosttyTerminal(id)`.
    ///
    /// - Parameters:
    ///   - tty: The TTY path originally used to register the session.
    ///   - ghosttyId: The resolved Ghostty terminal ID.
    /// - Returns: `true` if the re-keying succeeded, `false` if no session found for the TTY.
    @discardableResult
    public func handleTerminalIdResolution(tty: String, ghosttyId: String) -> Bool {
        let ttyKey = TerminalIdentifier.tty(tty)
        guard let state = sessions.removeValue(forKey: ttyKey) else {
            Self.logger.warning("Terminal ID resolution: no session found for TTY \(tty)")
            return false
        }

        let ghosttyKey = TerminalIdentifier.ghosttyTerminal(ghosttyId)
        sessions[ghosttyKey] = state.withIdentifier(ghosttyKey)
        ttyToGhosttyId[tty] = ghosttyId

        Self.logger.info(
            "Terminal ID resolved: TTY \(tty) -> Ghostty terminal \(ghosttyId)"
        )
        return true
    }

    // MARK: - State Queries

    /// Get a snapshot of all registered sessions for the HTTP /state endpoint.
    public func getState() -> RegistrySnapshot {
        let snapshots = sessions.values.map(SessionSnapshot.init(from:))
        return RegistrySnapshot(sessions: snapshots, sessionCount: snapshots.count)
    }

    /// Get all currently registered session states.
    public func allSessions() -> [SessionState] {
        Array(sessions.values)
    }

    /// Look up a session state by name.
    ///
    /// - Parameter name: The session name to look up.
    /// - Returns: The SessionState if found.
    public func session(named name: String) -> SessionState? {
        sessions.values.first { $0.name == name }
    }

    /// Look up a session state by terminal identifier.
    ///
    /// - Parameter identifier: The terminal identifier to look up.
    /// - Returns: The SessionState if found.
    public func session(byIdentifier identifier: TerminalIdentifier) -> SessionState? {
        sessions[identifier]
    }

    /// Look up a session state by TTY path.
    ///
    /// Checks the TTY-to-Ghostty-ID mapping first, so re-keyed Ghostty sessions
    /// can still be found by their original TTY path.
    ///
    /// - Parameter tty: The TTY path to look up.
    /// - Returns: The SessionState if found.
    public func session(byTTY tty: String) -> SessionState? {
        sessions[resolveIdentifier(forTTY: tty)]
    }

    // MARK: - Internal Helpers

    /// Resolve a TTY path to the appropriate terminal identifier.
    ///
    /// If the TTY has a known Ghostty terminal ID mapping (from a completed
    /// terminal ID resolution handshake), returns `.ghosttyTerminal(id)`.
    /// Otherwise returns `.tty(path)`.
    private func resolveIdentifier(forTTY tty: String) -> TerminalIdentifier {
        if let ghosttyId = ttyToGhosttyId[tty] {
            return .ghosttyTerminal(ghosttyId)
        }
        return .tty(tty)
    }

    /// Remove cached TTY-to-Ghostty mapping when a Ghostty session is removed.
    private func clearGhosttyMapping(for identifier: TerminalIdentifier) {
        if case .ghosttyTerminal(let ghosttyId) = identifier {
            ttyToGhosttyId = ttyToGhosttyId.filter { $0.value != ghosttyId }
        }
    }
}

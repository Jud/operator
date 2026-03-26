import AVFoundation

/// Handles operator voice commands (status, list, replay, help, what did I miss).
///
/// These commands are handled directly by the daemon without routing to any
/// Claude Code session. Extracted from StateMachine to keep command handling
/// logic separate from state transition orchestration.
///
/// Reference: technical-spec.md Component 7
@MainActor
public struct OperatorCommandHandler {
    private let registry: SessionRegistry
    private let audioQueue: AudioQueue
    private let voiceManager: VoiceManager

    /// Creates a new operator command handler.
    ///
    /// - Parameters:
    ///   - registry: Actor managing registered Claude Code sessions.
    ///   - audioQueue: Actor managing sequential speech playback.
    ///   - voiceManager: Voice selection for operator and agent speech.
    public init(registry: SessionRegistry, audioQueue: AudioQueue, voiceManager: VoiceManager) {
        self.registry = registry
        self.audioQueue = audioQueue
        self.voiceManager = voiceManager
    }

    /// Handle an operator voice command and return the text to speak as the Operator.
    ///
    /// Some commands enqueue audio directly (replay). All commands return a text
    /// string for the Operator to speak (may be nil for replay when audio is enqueued).
    ///
    /// - Parameters:
    ///   - command: The operator command to handle.
    ///   - pendingInterruption: The current pending interruption, if any (for "what did I miss").
    /// - Returns: The text to speak as the Operator, or nil if audio was enqueued directly.
    public func handle(
        _ command: OperatorCommand,
        pendingInterruption: PendingInterruption? = nil
    ) async -> String? {
        switch command {
        case .status:
            return await handleStatus()

        case .listAgents:
            return await handleListAgents()

        case .replay:
            return await handleReplay()

        case .replayAgent(let name):
            return await handleReplayAgent(name: name)

        case .whatDidIMiss:
            return await handleWhatDidIMiss(pendingInterruption: pendingInterruption)

        case .help:
            return "Available commands: operator status, list agents, replay, what did I miss, and operator help."
        }
    }

    /// Handle the "operator status" voice command.
    private func handleStatus() async -> String {
        let sessions = await registry.allSessions()
        if sessions.isEmpty {
            return "No agents are connected."
        }
        var parts: [String] = ["\(sessions.count) agent\(sessions.count == 1 ? "" : "s") connected."]
        for session in sessions {
            var line = "\(session.name), working in \(session.cwd)"
            if !session.context.isEmpty {
                line += ". \(session.context)"
            }
            parts.append(line)
        }
        return parts.joined(separator: " ")
    }

    /// Handle the "list agents" voice command.
    private func handleListAgents() async -> String {
        let sessions = await registry.allSessions()
        if sessions.isEmpty {
            return "No agents are connected."
        }
        let names = sessions.map { $0.name }
        let list = names.joined(separator: ", ")
        return "\(sessions.count) agent\(sessions.count == 1 ? "" : "s"): \(list)."
    }

    /// Handle the "operator replay" voice command.
    private func handleReplay() async -> String? {
        if let last = await audioQueue.lastSpokenMessage() {
            async let voice = registry.voiceFor(session: last.session)
            async let pitch = registry.pitchFor(session: last.session)
            async let nick = registry.nicknameFor(session: last.session)
            await audioQueue.enqueue(
                AudioQueue.QueuedMessage(
                    sessionName: last.session,
                    text: last.text,
                    priority: .urgent,
                    voice: await voice,
                    pitchMultiplier: await pitch,
                    spokenName: await nick
                )
            )
            return nil
        }
        return "Nothing to replay."
    }

    /// Handle the "what did [name] say" voice command.
    private func handleReplayAgent(name: String) async -> String? {
        let sessions = await registry.allSessions()
        let canonical = sessions.first { $0.name.lowercased() == name.lowercased() }?.name
        if let agentName = canonical,
            let text = await audioQueue.lastMessage(forAgent: agentName)
        {
            async let voice = registry.voiceFor(session: agentName)
            async let pitch = registry.pitchFor(session: agentName)
            async let nick = registry.nicknameFor(session: agentName)
            await audioQueue.enqueue(
                AudioQueue.QueuedMessage(
                    sessionName: agentName,
                    text: text,
                    priority: .urgent,
                    voice: await voice,
                    pitchMultiplier: await pitch,
                    spokenName: await nick
                )
            )
            return nil
        } else if let agentName = canonical {
            return "I don't have a recent message from \(agentName)."
        } else {
            return "I don't know an agent called \(name)."
        }
    }

    /// Handle the "what did I miss" voice command.
    private func handleWhatDidIMiss(pendingInterruption: PendingInterruption?) async -> String {
        let pendingCount = await audioQueue.pendingCount
        if pendingCount > 0 {
            return "You have \(pendingCount) message\(pendingCount == 1 ? "" : "s") waiting."
        } else if let interruption = pendingInterruption {
            return "You interrupted \(interruption.session). They were saying: \(interruption.heardText)"
        }
        return "You're all caught up."
    }
}

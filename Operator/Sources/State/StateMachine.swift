// swiftlint:disable file_length
import AVFoundation
import Foundation
import OSLog

/// Operational states of the Operator daemon.
///
/// Each state corresponds to a phase in the voice-to-delivery pipeline.
/// State transitions are managed exclusively by the StateMachine on @MainActor.
///
/// Reference: technical-spec.md Component 6; design.md Section 3.1.12
public enum OperatorState: String, Sendable, CustomStringConvertible {
    /// No active interaction. Audio queue plays freely.
    case idle = "IDLE"

    /// Push-to-talk active. Audio capturing via AVAudioEngine.
    case listening = "LISTENING"

    /// FN released. SFSpeechRecognizer processing captured audio.
    case transcribing = "TRANSCRIBING"

    /// Transcription complete. Routing through priority chain.
    case routing = "ROUTING"

    /// Route resolved. Delivering text to iTerm via JXA.
    case delivering = "DELIVERING"

    /// Route ambiguous. Waiting for user clarification via next push-to-talk.
    case clarifying = "CLARIFYING"

    /// An error occurred. Will auto-recover to IDLE after 2 seconds.
    case error = "ERROR"

    public var description: String { rawValue }
}

/// Central state machine that orchestrates all Operator components.
///
/// Owns the lifecycle of a single voice interaction: the user presses push-to-talk,
/// speaks, releases, and Operator transcribes, routes, and delivers the message.
/// Every state transition produces audio feedback (tones and/or speech) so the
/// product works fully without visual attention.
///
/// "Latest utterance wins": a trigger start from ANY state immediately cancels
/// in-flight work (transcription, routing subprocess, delivery) and re-enters
/// LISTENING. This ensures the user is never blocked by a stale interaction.
///
/// Concurrency: @MainActor ensures all state transitions are serialized on
/// the main thread. Background work (transcription, claude -p, JXA delivery)
/// runs in child Tasks whose results are delivered back to @MainActor. The
/// router, transcriber, and bridge are all called directly from @MainActor
/// context -- their async methods hop to the appropriate isolation domain
/// internally (e.g., SessionRegistry actor, osascript subprocess).
///
/// Reference: technical-spec.md Component 6; design.md Section 3.1.12, 3.3
@MainActor
public final class StateMachine {
    private static let logger = Logger(subsystem: "com.operator.app", category: "StateMachine")

    // MARK: - State

    /// Current operational state. All mutations go through transition(to:).
    public private(set) var currentState: OperatorState = .idle

    /// Routing state owned by the state machine. Passed to MessageRouter for
    /// routing decisions (affinity, history) and updated after successful routes.
    public private(set) var routingState = RoutingState()

    /// Count of clarification attempts for the current ambiguous message.
    /// Incremented on each re-ask. Max 2 attempts before giving up.
    private var clarificationAttempts = 0

    /// Handle to the current in-flight async work (transcription, routing, delivery).
    /// Cancelled on trigger start ("latest utterance wins") or timeout.
    private var currentTask: Task<Void, Never>?

    /// Handle to the timeout task for the current state. Each state with a timeout
    /// (TRANSCRIBING, ROUTING, DELIVERING, CLARIFYING) starts a timeout task that
    /// transitions to ERROR if the work does not complete in time.
    private var timeoutTask: Task<Void, Never>?

    // MARK: - Dependencies

    private let transcriber: any SpeechTranscribing
    private let audioQueue: AudioQueue
    private let router: MessageRouter
    private let feedback: AudioFeedback
    private let itermBridge: ITermBridge
    private let registry: SessionRegistry
    private let voiceManager: VoiceManager
    private let waveformPanel: WaveformPanel?
    private let speechManager: any SpeechManaging

    // MARK: - Initialization

    /// - Parameters:
    ///   - transcriber: SFSpeechRecognizer wrapper for voice-to-text.
    ///   - audioQueue: Actor managing sequential speech playback.
    ///   - router: Message routing priority chain.
    ///   - feedback: Non-verbal audio tone player.
    ///   - itermBridge: JXA bridge for iTerm2 text delivery.
    ///   - registry: Actor managing registered Claude Code sessions.
    ///   - voiceManager: Voice selection for operator and agent speech.
    ///   - waveformPanel: Floating UI panel (nil in headless/test mode).
    ///   - speechManager: AVSpeechSynthesizer wrapper for direct speech control.
    public init(
        transcriber: any SpeechTranscribing,
        audioQueue: AudioQueue,
        router: MessageRouter,
        feedback: AudioFeedback,
        itermBridge: ITermBridge,
        registry: SessionRegistry,
        voiceManager: VoiceManager,
        waveformPanel: WaveformPanel?,
        speechManager: any SpeechManaging
    ) {
        self.transcriber = transcriber
        self.audioQueue = audioQueue
        self.router = router
        self.feedback = feedback
        self.itermBridge = itermBridge
        self.registry = registry
        self.voiceManager = voiceManager
        self.waveformPanel = waveformPanel
        self.speechManager = speechManager

        Self.logger.info("StateMachine initialized in IDLE state")
    }

    // MARK: - Trigger Callbacks

    /// Called when push-to-talk key is pressed (FN down).
    ///
    /// "Latest utterance wins": this can be called from ANY state.
    /// All in-flight work is cancelled, current speech is stopped,
    /// any interruption point is noted, and we enter LISTENING.
    ///
    /// Reference: technical-spec.md Component 6, "Trigger start from any state"
    public func triggerStart() {
        Self.logger.info("Trigger START from state: \(self.currentState)")

        cancelInFlightWork()

        // Stop any current speech and note the interruption point.
        // We check SpeechManager directly (synchronous on @MainActor) rather than
        // going through AudioQueue (async actor) to capture the interruption info
        // immediately before it's lost.
        if speechManager.synthesizer.isSpeaking {
            let info = speechManager.interrupt()
            if !info.session.isEmpty {
                routingState.storeInterruption(
                    heardText: info.heardText,
                    unheardText: info.unheardText,
                    session: info.session
                )
                Self.logger.info("Noted interruption of \(info.session) at word boundary")
            }
        }

        // Pause audio queue (mute-on-talk). This is a fire-and-forget call;
        // the actor will process it asynchronously. We already captured
        // interruption info above from SpeechManager directly.
        Task {
            _ = await audioQueue.userStartedSpeaking()
        }

        transition(to: .listening)
        feedback.play(.listening)
        waveformPanel?.show()

        do {
            try transcriber.startListening()
        } catch {
            Self.logger.error("Failed to start audio capture: \(error)")
            speakOperator("I can't access the microphone. Check System Settings.")
            transition(to: .error)
            scheduleErrorRecovery()
        }
    }

    /// Called when push-to-talk key is released (FN up).
    ///
    /// Transitions to TRANSCRIBING and kicks off async transcription with a 30s timeout.
    public func triggerStop() {
        guard currentState == .listening else {
            Self.logger.warning("Trigger STOP ignored: not in LISTENING state (current: \(self.currentState))")
            return
        }

        Self.logger.info("Trigger STOP: entering TRANSCRIBING")

        transition(to: .transcribing)

        feedback.play(.processing)

        // Redundant timeout (belt and suspenders with transcriber's internal 30s timeout).
        startTimeout(seconds: 30)

        currentTask = Task { @MainActor [weak self] in
            guard let self else {
                return
            }

            let text = await self.transcriber.stopListeningWithTimeout(seconds: 30)

            await self.handleTranscriptionResult(text)
        }
    }
}

// MARK: - Transcription Processing

extension StateMachine {
    /// Process the result of transcription on @MainActor.
    private func handleTranscriptionResult(_ text: String?) async {
        guard !Task.isCancelled else {
            return
        }
        guard currentState == .transcribing else {
            Self.logger.debug("Transcription completed but state changed to \(self.currentState); discarding")
            return
        }

        cancelTimeout()

        if let text, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            Self.logger.info("Transcription result: \"\(text.prefix(80))\"")
            await processTranscription(text)
        } else {
            Self.logger.info("Empty transcription; returning to IDLE")
            speakOperator("I didn't catch that. Try again?")
            enterIdle()
        }
    }

    /// Process a successful transcription by routing through the message router.
    ///
    /// Checks for pending interruptions and clarifications before normal routing.
    private func processTranscription(_ text: String) async {
        if let interruption = routingState.pendingInterruption {
            routingState.clearInterruption()
            await handleInterruption(userUtterance: text, context: interruption)
            return
        }

        if let clarification = routingState.pendingClarification {
            await handleClarificationResponse(text: text, clarification: clarification)
            return
        }

        await performRouting(text)
    }

    /// Run the message through the routing priority chain.
    private func performRouting(_ text: String) async {
        transition(to: .routing)
        startTimeout(seconds: 15)

        let result = await router.route(text: text, routingState: routingState)

        guard !Task.isCancelled else {
            return
        }
        guard currentState == .routing else {
            Self.logger.debug("Routing completed but state changed to \(self.currentState); discarding")
            return
        }

        cancelTimeout()

        switch result {
        case let .operatorCommand(command):
            await handleOperatorCommand(command)

        case let .route(session, message):
            await deliverMessage(message, to: session)

        case let .clarify(candidates, question, originalText):
            enterClarifying(candidates: candidates, question: question, originalText: originalText)

        case .noSessions:
            speakOperator("No agents are connected. Start a Claude Code session to get going.")
            enterIdle()

        case .cliNotFound:
            speakOperator("Claude CLI not found. I need it for smart routing.")
            feedback.play(.error)
            enterIdle()
        }
    }
}

// MARK: - Interruption Handling

extension StateMachine {
    /// Process the user's utterance in the context of an interrupted speech playback.
    ///
    /// Uses InterruptionHandler's fast-path keywords first, then falls back to
    /// claude -p for genuinely ambiguous cases.
    private func handleInterruption(userUtterance: String, context: PendingInterruption) async {
        let interruptionContext = InterruptionContext(
            agentName: context.session,
            fullText: context.heardText + context.unheardText,
            heardText: context.heardText,
            unheardText: context.unheardText
        )

        let sessionNames = await registry.allSessions().map { $0.name }
        let action = await InterruptionHandler.handle(
            userUtterance: userUtterance,
            context: interruptionContext,
            registeredSessionNames: sessionNames
        )

        guard !Task.isCancelled else {
            return
        }

        switch action {
        case .skip:
            Self.logger.info("Interruption: skipping \(context.session)'s message")
            await performRouting(userUtterance)

        case .replay:
            Self.logger.info("Interruption: replaying \(context.session)'s message")
            await enqueueReplay(context: context)
            enterIdle()

        case let .route(target, message):
            Self.logger.info("Interruption: routing new message to \(target)")
            await deliverMessage(message, to: target)

        case let .ask(question):
            Self.logger.info("Interruption: asking user: \(question)")
            speakOperator(question)
            enterIdle()
        }
    }

    /// Enqueue a replay of an interrupted message.
    private func enqueueReplay(context: PendingInterruption) async {
        let voice = await registry.voiceFor(session: context.session)
        let pitch = await registry.pitchFor(session: context.session)

        let fullText = context.heardText + context.unheardText

        await audioQueue.enqueue(
            AudioQueue.QueuedMessage(
                sessionName: context.session,
                text: fullText,
                priority: .urgent,
                voice: voice,
                pitchMultiplier: pitch
            )
        )
    }
}

// MARK: - Delivery

extension StateMachine {
    /// Deliver a routed message to a specific Claude Code session via JXA.
    ///
    /// Enters DELIVERING state, writes to iTerm via the three-step sequence
    /// (Escape + text + CR), plays the delivered tone, and speaks a confirmation.
    ///
    /// Reference: technical-spec.md Component 5; research doc "Proven Recipe"
    private func deliverMessage(_ message: String, to session: String) async {
        transition(to: .delivering)
        startTimeout(seconds: 5)

        guard let sessionState = await registry.session(named: session) else {
            Self.logger.error("Session '\(session)' not found in registry for delivery")
            cancelTimeout()
            speakOperator("Couldn't deliver to \(session). The session may have closed.")
            feedback.play(.error)
            enterIdle()
            return
        }

        let tty = sessionState.tty

        do {
            let success = try await itermBridge.writeToSession(tty: tty, text: message)

            guard !Task.isCancelled else {
                return
            }
            guard currentState == .delivering else {
                Self.logger.debug("Delivery completed but state changed; discarding")
                return
            }

            cancelTimeout()

            if success {
                await handleDeliverySuccess(message: message, session: session)
            } else {
                Self.logger.error("writeToSession returned false for \(session)")
                speakOperator("Couldn't deliver to \(session). The session may have closed.")
                feedback.play(.error)
                enterIdle()
            }
        } catch ITermBridgeError.itermNotRunning {
            handleDeliveryError(session: session, isItermNotRunning: true)
        } catch {
            Self.logger.error("Delivery failed for \(session): \(error)")
            handleDeliveryError(session: session, isItermNotRunning: false)
        }
    }

    /// Handle a successful message delivery.
    private func handleDeliverySuccess(message: String, session: String) async {
        Self.logger.info("Delivered message to \(session)")
        feedback.play(.delivered)
        routingState.recordRoute(text: message, session: session)

        let sessionCount = await registry.sessionCount

        if sessionCount <= 1 {
            speakOperator("Sent.")
        } else {
            speakOperator("Sent to \(session).")
        }

        enterIdle()
    }

    /// Handle a delivery error, distinguishing iTerm-not-running from other errors.
    private func handleDeliveryError(session: String, isItermNotRunning: Bool) {
        guard !Task.isCancelled else {
            return
        }
        guard currentState == .delivering else {
            return
        }

        cancelTimeout()

        if isItermNotRunning {
            Self.logger.error("iTerm not running during delivery to \(session)")
            speakOperator("iTerm doesn't seem to be running.")
        } else {
            speakOperator("Couldn't deliver to \(session). The session may have closed.")
        }

        feedback.play(.error)
        enterIdle()
    }

    /// Deliver a message to a session without state transitions (used for multi-send).
    private func deliverToSessionDirect(_ message: String, to session: String) async {
        guard let sessionState = await registry.session(named: session) else {
            Self.logger.warning("Session '\(session)' not found for direct delivery")
            return
        }
        do {
            _ = try await itermBridge.writeToSession(tty: sessionState.tty, text: message)
            routingState.recordRoute(text: message, session: session)
            Self.logger.info("Direct delivery to \(session) succeeded")
        } catch {
            Self.logger.error("Direct delivery to \(session) failed: \(error)")
        }
    }
}

// MARK: - Clarification

extension StateMachine {
    /// Enter the CLARIFYING state: speak the question and wait for the user's
    /// next push-to-talk input (which will be a clarification response).
    ///
    /// Enforces max 2 clarification attempts per spec. After exceeding the max,
    /// gives up with a spoken message and returns to IDLE.
    ///
    /// Reference: technical-spec.md Component 3, "Ambiguous fallback";
    /// requirements.md AC-13.6
    private func enterClarifying(candidates: [String], question: String, originalText: String) {
        clarificationAttempts += 1

        if clarificationAttempts > 2 {
            Self.logger.info("Max clarification attempts reached; giving up")
            speakOperator("I couldn't figure out who that's for. Try saying the agent name.")
            enterIdle()
            return
        }

        transition(to: .clarifying)
        routingState.storeClarification(candidates: candidates, originalText: originalText)

        speakOperator(question)

        startTimeout(seconds: 8)
    }

    /// Process the user's response during the CLARIFYING state.
    private func handleClarificationResponse(text: String, clarification: PendingClarification) async {
        routingState.clearClarification()

        transition(to: .routing)
        startTimeout(seconds: 15)

        let result = await router.handleClarification(
            response: text,
            candidates: clarification.candidates,
            originalText: clarification.originalText
        )

        guard !Task.isCancelled else {
            return
        }
        guard currentState == .routing else {
            return
        }

        cancelTimeout()

        await applyClarificationResult(result, clarification: clarification)
    }

    /// Apply the resolved clarification result to the current interaction.
    private func applyClarificationResult(
        _ result: ClarificationResult,
        clarification: PendingClarification
    ) async {
        switch result {
        case let .resolved(session):
            Self.logger.info("Clarification resolved: routing to \(session)")
            await deliverMessage(clarification.originalText, to: session)

        case .sendToAll:
            Self.logger.info("Clarification: sending to all candidates")
            for candidate in clarification.candidates {
                await deliverToSessionDirect(clarification.originalText, to: candidate)
            }
            speakOperator("Sent to all.")
            enterIdle()

        case .cancelled:
            Self.logger.info("Clarification cancelled by user")
            speakOperator("OK, dropped it.")
            enterIdle()

        case .repeatQuestion:
            Self.logger.info("User wants clarification question repeated")
            let question = "Was that for \(clarification.candidates.joined(separator: " or "))?"
            enterClarifying(
                candidates: clarification.candidates,
                question: question,
                originalText: clarification.originalText
            )

        case .unresolved:
            Self.logger.info("Clarification unresolved; re-asking")
            let question = "I still couldn't tell. Was that for \(clarification.candidates.joined(separator: " or "))?"
            enterClarifying(
                candidates: clarification.candidates,
                question: question,
                originalText: clarification.originalText
            )
        }
    }
}

// MARK: - Operator Commands

extension StateMachine {
    /// Handle an operator voice command (status, list, replay, etc.).
    ///
    /// These are handled directly by the daemon without routing to any session.
    /// Reference: technical-spec.md Component 7
    private func handleOperatorCommand(_ command: OperatorCommand) async {
        Self.logger.info("Handling operator command: \(String(describing: command))")

        switch command {
        case .status:
            await handleStatusCommand()

        case .listAgents:
            await handleListAgentsCommand()

        case .replay:
            await handleReplayCommand()

        case let .replayAgent(name):
            await handleReplayAgentCommand(name: name)

        case .whatDidIMiss:
            await handleWhatDidIMissCommand()

        case .help:
            speakOperator(
                "Available commands: operator status, list agents, replay, what did I miss, and operator help."
            )
        }

        enterIdle()
    }

    /// Handle the "operator status" voice command.
    private func handleStatusCommand() async {
        let sessions = await registry.allSessions()
        if sessions.isEmpty {
            speakOperator("No agents are connected.")
        } else {
            var parts: [String] = ["\(sessions.count) agent\(sessions.count == 1 ? "" : "s") connected."]
            for session in sessions {
                var line = "\(session.name), working in \(session.cwd)"
                if !session.context.isEmpty {
                    line += ". \(session.context)"
                }
                parts.append(line)
            }
            speakOperator(parts.joined(separator: " "))
        }
    }

    /// Handle the "list agents" voice command.
    private func handleListAgentsCommand() async {
        let sessions = await registry.allSessions()
        if sessions.isEmpty {
            speakOperator("No agents are connected.")
        } else {
            let names = sessions.map { $0.name }
            let list = names.joined(separator: ", ")
            speakOperator("\(sessions.count) agent\(sessions.count == 1 ? "" : "s"): \(list).")
        }
    }

    /// Handle the "operator replay" voice command.
    private func handleReplayCommand() async {
        if let last = await audioQueue.lastSpokenMessage() {
            let voice = await registry.voiceFor(session: last.session)
            let pitch = await registry.pitchFor(session: last.session)
            await audioQueue.enqueue(
                AudioQueue.QueuedMessage(
                    sessionName: last.session,
                    text: last.text,
                    priority: .urgent,
                    voice: voice,
                    pitchMultiplier: pitch
                )
            )
        } else {
            speakOperator("Nothing to replay.")
        }
    }

    /// Handle the "what did [name] say" voice command.
    private func handleReplayAgentCommand(name: String) async {
        let sessions = await registry.allSessions()
        let canonical = sessions.first { $0.name.lowercased() == name.lowercased() }?.name
        if let agentName = canonical,
            let text = await audioQueue.lastMessage(forAgent: agentName) {
            let voice = await registry.voiceFor(session: agentName)
            let pitch = await registry.pitchFor(session: agentName)
            await audioQueue.enqueue(
                AudioQueue.QueuedMessage(
                    sessionName: agentName,
                    text: text,
                    priority: .urgent,
                    voice: voice,
                    pitchMultiplier: pitch
                )
            )
        } else if let agentName = canonical {
            speakOperator("I don't have a recent message from \(agentName).")
        } else {
            speakOperator("I don't know an agent called \(name).")
        }
    }

    /// Handle the "what did I miss" voice command.
    private func handleWhatDidIMissCommand() async {
        let pendingCount = await audioQueue.pendingCount
        if pendingCount > 0 {
            speakOperator("You have \(pendingCount) message\(pendingCount == 1 ? "" : "s") waiting.")
        } else if let interruption = routingState.pendingInterruption {
            speakOperator("You interrupted \(interruption.session). They were saying: \(interruption.heardText)")
        } else {
            speakOperator("You're all caught up.")
        }
    }
}

// MARK: - State Transitions & Infrastructure

extension StateMachine {
    /// Transition to a new state with logging.
    ///
    /// All state transitions flow through this method for consistent logging.
    private func transition(to newState: OperatorState) {
        let oldState = currentState
        currentState = newState
        Self.logger.info("State transition: \(oldState) -> \(newState)")
    }

    /// Enter the IDLE state: resume audio queue, play pending tone, fade waveform.
    ///
    /// This is the terminal state for every interaction cycle. The audio queue
    /// resumes playing any agent messages that arrived while the user was speaking.
    /// If messages are pending, AudioQueue plays the "pending" tone to alert the user.
    private func enterIdle() {
        cancelInFlightWork()
        transition(to: .idle)
        waveformPanel?.fadeOut()

        Task {
            await audioQueue.userStoppedSpeaking()
        }
    }

    /// Start a timeout for the current state.
    private func startTimeout(seconds: TimeInterval) {
        cancelTimeout()

        let state = currentState
        timeoutTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))

            await MainActor.run { [weak self] in
                self?.handleTimeout(seconds: seconds, forState: state)
            }
        }
    }

    /// Process a timeout on the main actor.
    private func handleTimeout(seconds: TimeInterval, forState state: OperatorState) {
        guard !Task.isCancelled else {
            return
        }
        guard currentState == state else {
            return
        }

        Self.logger.error("Timeout (\(seconds)s) in state \(state)")

        if state == .clarifying {
            routingState.clearClarification()
            speakOperator("No worries, I'll drop that message.")
            enterIdle()
        } else {
            speakOperator("Something went wrong. Let's try that again.")
            feedback.play(.error)
            transition(to: .error)
            scheduleErrorRecovery()
        }
    }

    /// Cancel any active timeout.
    private func cancelTimeout() {
        timeoutTask?.cancel()
        timeoutTask = nil
    }

    /// Schedule auto-recovery from ERROR to IDLE after 2 seconds.
    ///
    /// Reference: technical-spec.md Component 6, "ERROR state auto-recover to IDLE after 2s";
    /// requirements.md AC-15.7
    private func scheduleErrorRecovery() {
        currentTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 2_000_000_000)

            await MainActor.run { [weak self] in
                self?.performErrorRecovery()
            }
        }
    }

    /// Perform error recovery on the main actor.
    private func performErrorRecovery() {
        guard !Task.isCancelled else {
            return
        }
        guard currentState == .error else {
            return
        }

        Self.logger.info("Auto-recovering from ERROR to IDLE")
        enterIdle()
    }

    /// Cancel all in-flight async work (transcription, routing, delivery).
    private func cancelInFlightWork() {
        currentTask?.cancel()
        currentTask = nil
        cancelTimeout()
    }

    /// Speak a message using the Operator voice.
    private func speakOperator(_ text: String) {
        speechManager.speak(text, voice: voiceManager.operatorVoice, prefix: "Operator")
    }
}

// swiftlint:disable file_length
import Foundation

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

    /// A human-readable description of the state.
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
    private static let logger = Log.logger(for: "StateMachine")

    // MARK: - State

    /// Current operational state.
    ///
    /// All mutations go through transition(to:).
    public private(set) var currentState: OperatorState = .idle

    /// Routing state owned by the state machine.
    ///
    /// Passed to MessageRouter for routing decisions (affinity, history) and
    /// updated after successful routes.
    public private(set) var routingState = RoutingState()

    /// Handle to the current in-flight async work (transcription, routing, delivery).
    ///
    /// Cancelled on trigger start ("latest utterance wins") or timeout.
    private var currentTask: Task<Void, Never>?

    /// Handle to the timeout task for the current state.
    ///
    /// Each state with a timeout (TRANSCRIBING, ROUTING, DELIVERING, CLARIFYING)
    /// starts a timeout task that transitions to ERROR if the work does not
    /// complete in time.
    private var timeoutTask: Task<Void, Never>?

    /// Cached contextual strings for speech recognition, refreshed from the registry.
    private var cachedContextualStrings: [String] = []

    /// Handle to the in-flight contextual strings refresh task.
    private var contextRefreshTask: Task<Void, Never>?

    // MARK: - Dependencies

    private let transcriber: any SpeechTranscribing
    private let audioQueue: AudioQueue
    private let router: MessageRouter
    private let feedback: any AudioFeedbackProviding
    private let terminalBridge: any TerminalBridge
    private let registry: SessionRegistry
    private let voiceManager: VoiceManager
    private let waveformPanel: WaveformPanel?
    private let speechManager: any SpeechManaging
    private let commandHandler: OperatorCommandHandler
    private let interruptionHandler: InterruptionHandler
    private let deliveryCoordinator: DeliveryCoordinator
    private let bimodalEngine: BimodalDecisionEngine
    private let dictationDelivery: any DictationDelivering
    private let menuBarModel: MenuBarModel?

    // MARK: - Initialization

    /// - Parameters:
    ///   - transcriber: SFSpeechRecognizer wrapper for voice-to-text.
    ///   - audioQueue: Actor managing sequential speech playback.
    ///   - router: Message routing priority chain.
    ///   - feedback: Non-verbal audio tone player.
    ///   - terminalBridge: Bridge for terminal text delivery.
    ///   - registry: Actor managing registered Claude Code sessions.
    ///   - voiceManager: Voice selection for operator and agent speech.
    ///   - waveformPanel: Floating UI panel (nil in headless/test mode).
    ///   - speechManager: AVSpeechSynthesizer wrapper for direct speech control.
    ///   - bimodalEngine: Decision engine for dictation vs. agent routing.
    ///   - dictationDelivery: Pasteboard-based text insertion at the cursor.
    ///   - menuBarModel: Optional model for updating menu bar state display.
    public init(
        transcriber: any SpeechTranscribing,
        audioQueue: AudioQueue,
        router: MessageRouter,
        feedback: any AudioFeedbackProviding,
        terminalBridge: any TerminalBridge,
        registry: SessionRegistry,
        voiceManager: VoiceManager,
        waveformPanel: WaveformPanel?,
        speechManager: any SpeechManaging,
        bimodalEngine: BimodalDecisionEngine,
        dictationDelivery: any DictationDelivering,
        menuBarModel: MenuBarModel? = nil
    ) {
        self.transcriber = transcriber
        self.audioQueue = audioQueue
        self.router = router
        self.feedback = feedback
        self.terminalBridge = terminalBridge
        self.registry = registry
        self.voiceManager = voiceManager
        self.waveformPanel = waveformPanel
        self.speechManager = speechManager
        self.bimodalEngine = bimodalEngine
        self.dictationDelivery = dictationDelivery
        self.menuBarModel = menuBarModel
        self.interruptionHandler = InterruptionHandler(engine: router.engine)
        self.commandHandler = OperatorCommandHandler(
            registry: registry,
            audioQueue: audioQueue,
            voiceManager: voiceManager
        )
        self.deliveryCoordinator = DeliveryCoordinator(
            terminalBridge: terminalBridge,
            registry: registry,
            feedback: feedback
        )

        Self.logger.info("StateMachine initialized in IDLE state")
        refreshContextualStrings()
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
        if speechManager.isSpeaking {
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
            try transcriber.startListening(contextualStrings: cachedContextualStrings)
        } catch {
            Self.logger.error("Failed to start audio capture: \(error)")
            speakOperator("I can't access the microphone. Check System Settings.")
            transition(to: .error)
            scheduleErrorRecovery()
        }
    }

    /// Called when the user presses Escape to cancel the current recording.
    ///
    /// Only acts if in LISTENING or TRANSCRIBING state. Cancels in-flight work,
    /// plays cancel feedback, and returns to IDLE without processing.
    public func triggerCancel() {
        guard currentState == .listening || currentState == .transcribing else {
            Self.logger.warning(
                "Trigger CANCEL ignored: not in LISTENING or TRANSCRIBING state (current: \(self.currentState))"
            )
            return
        }

        Self.logger.info("Trigger CANCEL from state: \(self.currentState)")

        cancelInFlightWork()
        feedback.play(.error)
        enterIdle()
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

    /// Called when the user double-taps the push-to-talk key (two taps within 300ms).
    ///
    /// Replays the last dictation at the current cursor position. If in LISTENING
    /// state (first tap started listening), cancels listening without an error tone.
    /// Ignored from states other than IDLE or LISTENING.
    public func triggerDoubleTap() {
        guard currentState == .idle || currentState == .listening else {
            Self.logger.warning(
                "Double-tap ignored: not in IDLE or LISTENING state (current: \(self.currentState))"
            )
            return
        }

        Self.logger.info("Trigger DOUBLE-TAP from state: \(self.currentState)")

        if currentState == .listening {
            cancelInFlightWork()
            waveformPanel?.fadeOut()
        }

        currentTask = Task { @MainActor [weak self] in
            guard let self else {
                return
            }

            let result = await self.dictationDelivery.replayLast()

            guard !Task.isCancelled else {
                return
            }

            self.handleDictationResult(result)
            self.enterIdle()
        }
    }
}

// MARK: - Transcription Processing

extension StateMachine {
    /// Process the result of transcription on @MainActor.
    ///
    /// Runs the bimodal decision engine to choose between dictation (default)
    /// and agent routing (exception) before entering either pipeline.
    private func handleTranscriptionResult(_ text: String?) async {
        guard !Task.isCancelled else {
            return
        }
        guard currentState == .transcribing else {
            Self.logger.debug("Transcription completed but state changed to \(self.currentState); discarding")
            return
        }

        cancelTimeout()

        // Capture the audio filename before any async work clears it.
        let audioFile = transcriber.lastAudioFileURL?.lastPathComponent

        guard let text, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            Self.logger.info("Empty transcription; returning to IDLE")
            await postActivationTrace(
                text: "",
                step: .emptyTranscription,
                result: .notConfident(""),
                audioFile: audioFile
            )
            speakOperator("I didn't catch that. Try again?")
            enterIdle()
            return
        }

        Self.logger.info("Transcription result: \"\(text.prefix(80))\"")

        let decision = await bimodalEngine.decide(text: text, routingState: routingState)

        guard !Task.isCancelled else {
            return
        }
        guard currentState == .transcribing else {
            Self.logger.debug("State changed during bimodal decision; discarding")
            return
        }

        await dispatchBimodalDecision(decision, text: text, audioFile: audioFile)
    }

    private func dispatchBimodalDecision(
        _ decision: BimodalDecision,
        text: String,
        audioFile: String? = nil
    ) async {
        switch decision {
        case .routeToAgent:
            await processTranscription(text, audioFile: audioFile)

        case .dictate(let dictText):
            let trimmed = dictText.trimmingCharacters(in: .whitespacesAndNewlines)
            await postBimodalTrace(text: text, result: .notConfident(text), audioFile: audioFile)
            await performDictation(trimmed)

        case .noTextField:
            await postBimodalTrace(text: text, result: .noSessions, audioFile: audioFile)
            speakOperator("No text field detected. Move your cursor to a text input.")
            feedback.play(.error)
            enterIdle()

        case .permissionRequired:
            await postBimodalTrace(text: text, result: .noSessions, audioFile: audioFile)
            speakOperator(
                "Operator needs Accessibility permission for dictation. Check System Settings."
            )
            enterIdle()
        }
    }

    /// Deliver transcribed text at the cursor via pasteboard insertion.
    ///
    /// Called when the bimodal engine decides to dictate rather than route.
    /// No new state machine state is introduced; dictation completes in <200ms
    /// and transitions directly to IDLE.
    private func performDictation(_ text: String) async {
        let result = await dictationDelivery.deliver(text)

        guard !Task.isCancelled else {
            return
        }

        handleDictationResult(result)
        enterIdle()
    }

    /// Process a successful transcription by routing through the message router.
    ///
    /// Checks for pending interruptions and clarifications before normal routing.
    private func processTranscription(_ text: String, audioFile: String? = nil) async {
        if let interruption = routingState.pendingInterruption {
            routingState.clearInterruption()
            await handleInterruption(userUtterance: text, context: interruption)
            return
        }

        if let clarification = routingState.pendingClarification {
            await handleClarificationResponse(text: text, clarification: clarification)
            return
        }

        await performRouting(text, audioFile: audioFile)
    }

    /// Post a routing trace for bimodal decisions that skip the full routing chain.
    private func postBimodalTrace(
        text: String,
        result: RoutingResult,
        audioFile: String? = nil
    ) async {
        let step: RoutingTrace.RoutingStep =
            switch result {
            case .noSessions: .noSessions
            default: .heuristicScoring
            }
        await postActivationTrace(text: text, step: step, result: result, audioFile: audioFile)
    }

    /// Post a trace for any activation (routing, bimodal, error, empty transcription).
    private func postActivationTrace(
        text: String,
        step: RoutingTrace.RoutingStep,
        result: RoutingResult,
        audioFile: String? = nil
    ) async {
        let sessions = await registry.allSessions()
        await router.postTrace(
            text: text,
            sessions: sessions,
            step: step,
            result: result,
            routingState: routingState,
            audioFile: audioFile
        )
    }

    /// Run the message through the routing priority chain.
    private func performRouting(_ text: String, audioFile: String? = nil) async {
        transition(to: .routing)
        startTimeout(seconds: 15)

        let result = await router.route(text: text, routingState: routingState, audioFile: audioFile)

        guard !Task.isCancelled else {
            return
        }
        guard currentState == .routing else {
            Self.logger.debug("Routing completed but state changed to \(self.currentState); discarding")
            return
        }

        cancelTimeout()

        switch result {
        case .operatorCommand(let command):
            await handleOperatorCommand(command)

        case .route(let session, let message):
            await deliverMessage(message, to: session)

        case .clarify(let candidates, let question, let originalText):
            enterClarifying(candidates: candidates, question: question, originalText: originalText)

        case .noSessions:
            speakOperator("No agents are connected. Start a Claude Code session to get going.")
            enterIdle()

        case .cliNotFound:
            speakOperator("Claude CLI not found. I need it for smart routing.")
            feedback.play(.error)
            enterIdle()

        case .notConfident:
            Self.logger.warning("Received .notConfident from full routing path; entering idle")
            enterIdle()
        }
    }
}

// MARK: - Interruption Handling

extension StateMachine {
    /// Process the user's utterance in the context of an interrupted speech playback.
    ///
    /// Uses InterruptionHandler's fast-path keywords first, then falls back to
    /// the routing engine for genuinely ambiguous cases.
    private func handleInterruption(userUtterance: String, context: PendingInterruption) async {
        let interruptionContext = InterruptionContext(
            agentName: context.session,
            fullText: context.heardText + context.unheardText,
            heardText: context.heardText,
            unheardText: context.unheardText
        )

        let sessionNames = await registry.allSessions().map { $0.name }
        let action = await interruptionHandler.handle(
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

        case .route(let target, let message):
            Self.logger.info("Interruption: routing new message to \(target)")
            await deliverMessage(message, to: target)

        case .ask(let question):
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
    /// Enters DELIVERING state, delegates to DeliveryCoordinator, and handles
    /// state transitions based on the result.
    ///
    /// Reference: technical-spec.md Component 5; research doc "Proven Recipe"
    private func deliverMessage(_ message: String, to session: String) async {
        transition(to: .delivering)
        startTimeout(seconds: 5)

        let result = await deliveryCoordinator.deliver(message, to: session)

        guard !Task.isCancelled else {
            return
        }
        guard currentState == .delivering else {
            Self.logger.debug("Delivery completed but state changed; discarding")
            return
        }

        cancelTimeout()

        switch result {
        case .success(let msg, let sess):
            routingState.recordRoute(text: msg, session: sess)
            let sessionCount = await registry.sessionCount
            feedback.play(.delivered)
            if sessionCount > 1 {
                speakOperator("Sent to \(sess).")
            }
            enterIdle()

        case .sessionNotFound(let sess),
            .writeFailed(let sess),
            .deliveryError(let sess):
            speakOperator("Couldn't deliver to \(sess). The session may have closed.")
            feedback.play(.error)
            enterIdle()

        case .terminalNotRunning:
            speakOperator("The terminal doesn't seem to be running.")
            feedback.play(.error)
            enterIdle()
        }
    }

    /// Deliver a message to a session without state transitions (used for multi-send).
    private func deliverToSessionDirect(_ message: String, to session: String) async {
        if await deliveryCoordinator.deliverDirect(message, to: session) {
            routingState.recordRoute(text: message, session: session)
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
        routingState.clarificationAttempts += 1

        if routingState.clarificationAttempts > 2 {
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
        case .resolved(let session):
            Self.logger.info("Clarification resolved: routing to \(session)")
            await deliverMessage(clarification.originalText, to: session)

        case .sendToAll:
            Self.logger.info("Clarification: sending to all candidates")
            for candidate in clarification.candidates {
                await deliverToSessionDirect(clarification.originalText, to: candidate)
            }
            feedback.play(.delivered)
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
    /// Delegates to OperatorCommandHandler for the actual command logic,
    /// then speaks the result and returns to IDLE.
    /// Reference: technical-spec.md Component 7
    private func handleOperatorCommand(_ command: OperatorCommand) async {
        Self.logger.info("Handling operator command: \(String(describing: command))")

        let response = await commandHandler.handle(
            command,
            pendingInterruption: routingState.pendingInterruption
        )
        if let text = response {
            speakOperator(text)
        }

        enterIdle()
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
        menuBarModel?.update(state: newState.rawValue)
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

        refreshContextualStrings()
    }

    /// Start a timeout for the current state.
    private func startTimeout(seconds: TimeInterval) {
        cancelTimeout()

        let state = currentState
        timeoutTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
            self?.handleTimeout(seconds: seconds, forState: state)
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

        // Capture audio file before it's cleared by the next activation.
        let audioFile = transcriber.lastAudioFileURL?.lastPathComponent

        Task {
            await postActivationTrace(
                text: "[timeout]",
                step: .error,
                result: .notConfident("[timeout in \(state.rawValue) after \(Int(seconds))s]"),
                audioFile: audioFile
            )
        }

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
            self?.performErrorRecovery()
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

    /// Refresh cached contextual strings from the session registry and visible UI.
    ///
    /// Called on enterIdle so the cache is ready for the next triggerStart.
    /// Pulls session names and project directory names from the registry,
    /// plus visible vocabulary from HarnessCore (AX labels + OCR).
    /// Apple's contextualStrings is most effective with ~100 entries.
    private func refreshContextualStrings() {
        contextRefreshTask?.cancel()
        let axQuery = bimodalEngine.accessibilityQuery
        contextRefreshTask = Task {
            var strings = Set<String>()

            let sessions = await registry.allSessions()
            guard !Task.isCancelled else {
                return
            }
            for session in sessions {
                strings.insert(session.name)
                let projectName = URL(fileURLWithPath: session.cwd).lastPathComponent
                if !projectName.isEmpty {
                    strings.insert(projectName)
                }
            }

            let visible = await axQuery.visibleVocabulary()
            guard !Task.isCancelled else {
                return
            }
            strings.formUnion(visible)

            let sorted = strings.sorted { $0.count > $1.count }
            cachedContextualStrings = Array(sorted.prefix(100))
        }
    }

    /// Handle a DictationResult by playing the appropriate feedback/speech.
    ///
    /// Shared by `performDictation` (new dictation) and `triggerDoubleTap` (replay).
    /// Does NOT call `enterIdle()` — the caller is responsible for state transitions.
    private func handleDictationResult(_ result: DictationResult) {
        switch result {
        case .success:
            feedback.play(.dictation)

        case .noLastDictation:
            Self.logger.info("No lastDictation available")
            feedback.play(.error)

        case .pasteFailed(let reason):
            Self.logger.error("Dictation paste failed: \(reason)")
            speakOperator("Couldn't insert text at the cursor.")
            feedback.play(.error)

        case .noTextField:
            speakOperator("No text field detected.")
            feedback.play(.error)
        }
    }
}

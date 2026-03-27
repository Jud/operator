import AVFoundation

/// Thread-safe audio queue that manages sequential playback of agent speech messages.
///
/// AudioQueue is a Swift actor that owns the FIFO message queue and coordinates with
/// SpeechManager for actual speech synthesis. It enforces turn-taking: only one message
/// plays at a time, urgent messages jump to the front, and the queue pauses when the
/// user activates push-to-talk (mute-on-talk).
///
/// Playback flow:
/// 1. External code calls `enqueue(_:)` to add a message.
/// 2. If idle, `playNextIfIdle()` dequeues the front message and hands it to SpeechManager.
/// 3. SpeechManager yields to its `finishedSpeaking` AsyncStream when the utterance completes.
/// 4. The stream consumer calls `handleSpeechFinished()`, which transitions back to idle and
///    triggers `playNextIfIdle()` again for the next message.
///
/// Mute-on-talk:
/// - `userStartedSpeaking()` interrupts any current speech and pauses the queue.
/// - `userStoppedSpeaking()` resumes playback. If messages are pending, the "pending"
///   audio tone plays before resuming.
public actor AudioQueue {
    // MARK: - Subtypes

    /// Internal state tracking whether the queue is actively playing, idle, or paused
    /// because the user is speaking.
    public enum AudioState: Sendable {
        case idle
        case speaking
        case userSpeaking
    }

    /// A message waiting in the queue for sequential playback.
    public struct QueuedMessage: Sendable {
        /// Priority level for queued messages.
        public enum Priority: Sendable {
            /// May be dropped under saturation. Queued FIFO behind normal messages.
            case low
            /// Default priority. Queued FIFO.
            case normal
            /// Jumps to the front of the queue.
            case urgent
        }

        /// The session name associated with this message (full name, used for tracking).
        public let sessionName: String
        /// Short speakable name used as TTS prefix (defaults to sessionName).
        public let spokenName: String
        /// The text content to speak.
        public let text: String
        /// The priority level of this message.
        public let priority: Priority
        /// The voice to use for speech synthesis.
        public let voice: VoiceDescriptor
        /// The pitch multiplier for per-agent differentiation.
        public let pitchMultiplier: Float

        /// Creates a new queued message.
        public init(
            sessionName: String,
            text: String,
            priority: Priority,
            voice: VoiceDescriptor,
            pitchMultiplier: Float = 1.0,
            spokenName: String? = nil
        ) {
            self.sessionName = sessionName
            self.spokenName = spokenName ?? sessionName
            self.text = text
            self.priority = priority
            self.voice = voice
            self.pitchMultiplier = pitchMultiplier
        }
    }

    // MARK: - Type Properties

    private static let logger = Log.logger(for: "AudioQueue")

    // MARK: - Instance Properties

    private var queue: [QueuedMessage] = []
    private var state: AudioState = .idle
    private var speechManager: any SpeechManaging
    private let feedback: any AudioFeedbackProviding

    /// Tracks the last spoken message text per agent session name, for replay support.
    private var lastSpokenPerAgent: [String: String] = [:]

    /// The session name of the most recently played agent message.
    private var lastSpokenAgent: String?

    /// Handle to the listener task consuming the speech manager's finishedSpeaking stream.
    private var listenerTask: Task<Void, Never>?

    /// The number of messages currently waiting in the queue.
    public var pendingCount: Int {
        queue.count
    }

    /// Whether the queue is currently playing a message.
    public var isSpeaking: Bool {
        state == .speaking
    }

    /// Whether all agent speech is muted.
    ///
    /// When muted, enqueued messages are silently dropped. Feedback tones
    /// and operator messages are unaffected.
    private var muted: Bool = false

    /// Mute or unmute all agent speech.
    public func setMuted(_ value: Bool) {
        muted = value
        if value {
            let dropped = queue.count
            queue.removeAll()
            Self.logger.info("Muted — dropped \(dropped) pending messages")
        }
    }

    // MARK: - Initialization

    /// Create an AudioQueue with a speech manager and audio feedback player.
    ///
    /// - Parameters:
    ///   - speechManager: The speech synthesis manager.
    ///   - feedback: The audio feedback tone player.
    public init(speechManager: any SpeechManaging, feedback: any AudioFeedbackProviding) {
        self.speechManager = speechManager
        self.feedback = feedback
    }

    // MARK: - Setup

    /// Start consuming the speech manager's `finishedSpeaking` stream to drive sequential playback.
    ///
    /// Cancels any existing listener task before spawning a new one. The task awaits each
    /// stream element and calls `handleSpeechFinished()`. Ends automatically when the stream
    /// terminates (i.e., the speech manager deallocates) or the task is cancelled.
    /// Replace the speech manager (e.g. after Kokoro models finish downloading).
    ///
    /// Restarts the finished-speaking listener to observe the new manager's stream.
    public func replaceSpeechManager(_ newManager: any SpeechManaging) async {
        speechManager = newManager
        await startListening()
    }

    /// Start listening for speech-finished events from the speech manager.
    public func startListening() async {
        listenerTask?.cancel()
        let mgr = speechManager
        let stream = await MainActor.run { mgr.finishedSpeaking }
        listenerTask = Task { [weak self] in
            for await _ in stream {
                guard !Task.isCancelled else { break }
                await self?.handleSpeechFinished()
            }
        }
    }

    // MARK: - Queue Operations

    /// Add a message to the playback queue.
    ///
    /// Urgent-priority messages are inserted at the front of the queue.
    /// Normal messages are appended to the back (FIFO order).
    /// When muted, messages are silently dropped.
    public func enqueue(_ msg: QueuedMessage) async {
        if muted {
            Self.logger.debug("Muted — dropping message from \(msg.sessionName)")
            return
        }

        if msg.priority == .urgent {
            queue.insert(msg, at: 0)
            Self.logger.info(
                "Enqueued urgent message from \(msg.sessionName) at front (queue depth: \(self.queue.count))"
            )
        } else {
            queue.append(msg)
            Self.logger.info("Enqueued message from \(msg.sessionName) at back (queue depth: \(self.queue.count))")
        }
        await playNextIfIdle()
    }

    /// Pause the queue and interrupt current speech because the user activated push-to-talk.
    ///
    /// - Returns: Interruption info if speech was playing, nil if idle or paused.
    public func userStartedSpeaking() async -> InterruptInfo? {
        state = .userSpeaking
        Self.logger.info("User started speaking; queue paused (pending: \(self.queue.count))")

        let mgr = speechManager
        let info: InterruptInfo? = await MainActor.run {
            guard mgr.isSpeaking
            else { return nil }
            return mgr.interrupt()
        }
        if let info {
            Self.logger.info("Interrupted speech for session \(info.session) at char boundary")
        }
        return info
    }

    /// Resume queue playback after the user finishes speaking.
    public func userStoppedSpeaking() async {
        state = .idle
        Self.logger.info("User stopped speaking; resuming queue (pending: \(self.queue.count))")

        await playNextIfIdle()
    }

    /// Remove all pending messages from the queue without stopping current playback.
    public func clearQueue() {
        let count = queue.count
        queue.removeAll()
        Self.logger.info("Cleared \(count) pending messages from queue")
    }

    /// Retrieve the last spoken message text for a specific agent.
    ///
    /// - Parameter agent: The session name (case-sensitive).
    /// - Returns: The message text last spoken by this agent, or nil if none recorded.
    public func lastMessage(forAgent agent: String) -> String? {
        lastSpokenPerAgent[agent]
    }

    /// Retrieve the last spoken agent message (any agent).
    ///
    /// - Returns: A tuple of (sessionName, text) for the most recently played message,
    ///   or nil if no messages have been played.
    public func lastSpokenMessage() -> (session: String, text: String)? {
        guard let agent = lastSpokenAgent, let text = lastSpokenPerAgent[agent] else {
            return nil
        }
        return (session: agent, text: text)
    }

    // MARK: - Playback Engine

    /// Force the queue out of speaking state after an audio route change.
    ///
    /// Called by AudioHub when the output engine is restarted — the current
    /// utterance was interrupted and the completion callback won't fire.
    public func recoverFromEngineReset() {
        guard state == .speaking else {
            return
        }
        Self.logger.warning("Engine reset — recovering queue from speaking state")
        state = .idle
    }

    /// Called by SpeechManager when an utterance finishes, driving sequential playback.
    private func handleSpeechFinished() async {
        guard state == .speaking else {
            Self.logger.debug("Speech finished but state is \(String(describing: self.state)); not advancing")
            return
        }
        state = .idle
        Self.logger.debug("Speech finished; advancing queue (pending: \(self.queue.count))")
        await playNextIfIdle()
    }

    /// Dequeue and play the next message if idle.
    private func playNextIfIdle() async {
        guard state == .idle, !queue.isEmpty else {
            return
        }
        state = .speaking

        let msg = queue.removeFirst()

        Self.logger.info(
            "Playing message from \(msg.sessionName): \"\(msg.text.prefix(60))...\" (remaining: \(self.queue.count))"
        )

        lastSpokenPerAgent[msg.sessionName] = msg.text
        lastSpokenAgent = msg.sessionName

        let mgr = speechManager
        await MainActor.run {
            mgr.speak(
                msg.text,
                voice: msg.voice,
                prefix: msg.spokenName,
                pitchMultiplier: msg.pitchMultiplier
            )
        }
    }
}

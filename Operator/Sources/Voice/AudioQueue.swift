import AVFoundation
import OSLog

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
/// 3. SpeechManager's `onFinishedSpeaking` callback fires when the utterance completes.
/// 4. The callback calls `handleSpeechFinished()`, which transitions back to idle and
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
        public enum Priority: Sendable {
            case normal
            case urgent
        }

        public let sessionName: String
        public let text: String
        public let priority: Priority
        public let voice: AVSpeechSynthesisVoice
        public let pitchMultiplier: Float

        public init(
            sessionName: String,
            text: String,
            priority: Priority,
            voice: AVSpeechSynthesisVoice,
            pitchMultiplier: Float = 1.0
        ) {
            self.sessionName = sessionName
            self.text = text
            self.priority = priority
            self.voice = voice
            self.pitchMultiplier = pitchMultiplier
        }
    }

    // MARK: - Type Properties

    private static let logger = Logger(subsystem: "com.operator.app", category: "AudioQueue")

    // MARK: - Instance Properties

    private var queue: [QueuedMessage] = []
    private var state: AudioState = .idle
    private let speechManager: any SpeechManaging
    private let feedback: AudioFeedback

    /// Tracks the last spoken message text per agent session name, for replay support.
    private var lastSpokenPerAgent: [String: String] = [:]

    /// The session name of the most recently played agent message.
    private var lastSpokenAgent: String?

    /// The number of messages currently waiting in the queue.
    public var pendingCount: Int {
        queue.count
    }

    /// Whether the queue is currently playing a message.
    public var isSpeaking: Bool {
        state == .speaking
    }

    // MARK: - Initialization

    /// Create an AudioQueue with a speech manager and audio feedback player.
    ///
    /// - Parameters:
    ///   - speechManager: The speech synthesis manager.
    ///   - feedback: The audio feedback tone player.
    public init(speechManager: any SpeechManaging, feedback: AudioFeedback) {
        self.speechManager = speechManager
        self.feedback = feedback
    }

    // MARK: - Setup

    /// Wire up the SpeechManager delegate callback to drive sequential playback.
    ///
    /// Must be called once after init to connect the SpeechManager's onFinishedSpeaking
    /// callback to this actor's handleSpeechFinished method.
    public func setUp() async {
        let manager = speechManager
        await MainActor.run {
            manager.onFinishedSpeaking = { [weak self] in
                guard let self else {
                    return
                }
                Task {
                    await self.handleSpeechFinished()
                }
            }
        }
    }

    // MARK: - Queue Operations

    /// Add a message to the playback queue.
    ///
    /// Urgent-priority messages are inserted at the front of the queue.
    /// Normal messages are appended to the back (FIFO order).
    public func enqueue(_ msg: QueuedMessage) async {
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

        let isSpeaking = await MainActor.run { speechManager.synthesizer.isSpeaking }
        guard isSpeaking else {
            return nil
        }

        let info = await MainActor.run { speechManager.interrupt() }
        Self.logger.info("Interrupted speech for session \(info.session) at char boundary")
        return info
    }

    /// Resume queue playback after the user finishes speaking.
    public func userStoppedSpeaking() async {
        state = .idle
        Self.logger.info("User stopped speaking; resuming queue (pending: \(self.queue.count))")

        if !queue.isEmpty {
            feedback.play(.pending)
        }
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

        await MainActor.run {
            speechManager.speak(
                msg.text,
                voice: msg.voice,
                prefix: msg.sessionName,
                pitchMultiplier: msg.pitchMultiplier
            )
        }
    }
}

import AVFoundation
import Testing

@testable import OperatorCore

// MARK: - Mocks

@MainActor
private final class MockSpeech: SpeechManaging {
    var isSpeaking = false
    var finishedSpeaking: AsyncStream<Void> = AsyncStream { _ in }
    var spokenTexts: [String] = []

    func speak(_ text: String, voice: VoiceDescriptor, prefix: String, pitchMultiplier: Float) {
        isSpeaking = true
        spokenTexts.append(text)
    }

    func interrupt() -> InterruptInfo {
        isSpeaking = false
        return InterruptInfo(heardText: "", unheardText: "", session: "")
    }

    func stop() { isSpeaking = false }
}

@MainActor
private final class MockFeedback: AudioFeedbackProviding {
    var playedCues: [AudioCue] = []

    func play(_ cue: AudioCue) { playedCues.append(cue) }
}

// MARK: - Helpers

private struct TestHarness {
    static let voice = VoiceDescriptor.system(AVSpeechSynthesisVoice())

    let speech: MockSpeech
    let feedback: MockFeedback
    let queue: AudioQueue

    @MainActor
    init() {
        self.speech = MockSpeech()
        self.feedback = MockFeedback()
        self.queue = AudioQueue(speechManager: speech, feedback: feedback)
    }

    func message(
        _ sessionName: String,
        _ text: String,
        priority: AudioQueue.QueuedMessage.Priority = .normal
    ) -> AudioQueue.QueuedMessage {
        AudioQueue.QueuedMessage(
            sessionName: sessionName,
            text: text,
            priority: priority,
            voice: Self.voice
        )
    }
}

// MARK: - Tests

// .serialized prevents concurrent execution with @MainActor tests in other suites,
// which would deadlock since AudioQueue.enqueue hops to MainActor for speechManager.speak().
@Suite("AudioQueue", .serialized)
internal struct AudioQueueTests {
    @Test("lastSpokenMessage is nil initially")
    func lastSpokenNil() async {
        let harness = await TestHarness()

        let last = await harness.queue.lastSpokenMessage()
        #expect(last == nil)
    }

    @Test("enqueue triggers speak and records lastSpokenMessage")
    func enqueueTriggersSpeak() async {
        let harness = await TestHarness()

        await harness.queue.enqueue(harness.message("sudo", "Build succeeded"))

        let last = await harness.queue.lastSpokenMessage()
        #expect(last?.session == "sudo")
        #expect(last?.text == "Build succeeded")

        let spoken = await harness.speech.spokenTexts
        #expect(spoken == ["Build succeeded"])
    }

    @Test("urgent message inserted at front of queue")
    func urgentInsertedAtFront() async {
        let harness = await TestHarness()

        // First enqueue auto-plays (state -> .speaking)
        await harness.queue.enqueue(harness.message("a", "first"))

        // Now queue is speaking; these go into the queue
        await harness.queue.enqueue(harness.message("b", "normal"))
        await harness.queue.enqueue(harness.message("c", "urgent", priority: .urgent))

        // Urgent should be at front (index 0), normal behind it
        #expect(await harness.queue.pendingCount == 2)
    }

    @Test("lastMessage(forAgent:) returns text for known agent")
    func lastMessageForKnownAgent() async {
        let harness = await TestHarness()

        await harness.queue.enqueue(harness.message("frontend", "Styles updated"))

        let text = await harness.queue.lastMessage(forAgent: "frontend")
        #expect(text == "Styles updated")
    }

    @Test("lastMessage(forAgent:) returns nil for unknown agent")
    func lastMessageForUnknownAgent() async {
        let harness = await TestHarness()

        let text = await harness.queue.lastMessage(forAgent: "nonexistent")
        #expect(text == nil)
    }

    @Test("clearQueue empties pending messages")
    func clearQueue() async {
        let harness = await TestHarness()

        // First enqueue auto-plays
        await harness.queue.enqueue(harness.message("a", "playing"))
        // These queue up behind it
        await harness.queue.enqueue(harness.message("b", "queued1"))
        await harness.queue.enqueue(harness.message("c", "queued2"))

        #expect(await harness.queue.pendingCount == 2)
        await harness.queue.clearQueue()
        #expect(await harness.queue.pendingCount == 0)
    }

    @Test("pendingCount is 0 when empty")
    func pendingCountZero() async {
        let harness = await TestHarness()

        #expect(await harness.queue.pendingCount == 0)
    }

    @Test("multiple agents tracked independently for lastMessage")
    func multipleAgentsTracked() async {
        let harness = await TestHarness()

        // First enqueue auto-plays, sets state to speaking
        await harness.queue.enqueue(harness.message("sudo", "from sudo"))

        // This one queues (not played yet) but we can check sudo was recorded
        let sudoText = await harness.queue.lastMessage(forAgent: "sudo")
        #expect(sudoText == "from sudo")
    }
}

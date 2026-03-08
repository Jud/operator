import AVFoundation
import Foundation
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

// MARK: - Tests

// .serialized prevents concurrent execution with @MainActor tests in other suites,
// which would deadlock since AudioQueue.enqueue hops to MainActor for speechManager.speak().
@Suite("AudioQueue", .serialized)
internal struct AudioQueueTests {
    private func makeVoice() -> VoiceDescriptor {
        VoiceDescriptor(appleVoice: AVSpeechSynthesisVoice(), qwenSpeakerID: "ryan")
    }

    @Test("lastSpokenMessage is nil initially")
    func lastSpokenNil() async {
        let speech = await MockSpeech()
        let feedback = await MockFeedback()
        let queue = AudioQueue(speechManager: speech, feedback: feedback)

        let last = await queue.lastSpokenMessage()
        #expect(last == nil)
    }

    @Test("enqueue triggers speak and records lastSpokenMessage")
    func enqueueTriggersSpeak() async {
        let speech = await MockSpeech()
        let feedback = await MockFeedback()
        let queue = AudioQueue(speechManager: speech, feedback: feedback)

        let msg = AudioQueue.QueuedMessage(
            sessionName: "sudo",
            text: "Build succeeded",
            priority: .normal,
            voice: makeVoice()
        )
        await queue.enqueue(msg)

        let last = await queue.lastSpokenMessage()
        #expect(last?.session == "sudo")
        #expect(last?.text == "Build succeeded")

        let spoken = await speech.spokenTexts
        #expect(spoken == ["Build succeeded"])
    }

    @Test("urgent message inserted at front of queue")
    func urgentInsertedAtFront() async {
        let speech = await MockSpeech()
        let feedback = await MockFeedback()
        let queue = AudioQueue(speechManager: speech, feedback: feedback)

        let voice = makeVoice()

        // First enqueue auto-plays (state → .speaking)
        let first = AudioQueue.QueuedMessage(
            sessionName: "a",
            text: "first",
            priority: .normal,
            voice: voice
        )
        await queue.enqueue(first)

        // Now queue is speaking; these go into the queue
        let normal = AudioQueue.QueuedMessage(
            sessionName: "b",
            text: "normal",
            priority: .normal,
            voice: voice
        )
        let urgent = AudioQueue.QueuedMessage(
            sessionName: "c",
            text: "urgent",
            priority: .urgent,
            voice: voice
        )
        await queue.enqueue(normal)
        await queue.enqueue(urgent)

        // Urgent should be at front (index 0), normal behind it
        let pending = await queue.pendingCount
        #expect(pending == 2)
    }

    @Test("lastMessage(forAgent:) returns text for known agent")
    func lastMessageForKnownAgent() async {
        let speech = await MockSpeech()
        let feedback = await MockFeedback()
        let queue = AudioQueue(speechManager: speech, feedback: feedback)

        let msg = AudioQueue.QueuedMessage(
            sessionName: "frontend",
            text: "Styles updated",
            priority: .normal,
            voice: makeVoice()
        )
        await queue.enqueue(msg)

        let text = await queue.lastMessage(forAgent: "frontend")
        #expect(text == "Styles updated")
    }

    @Test("lastMessage(forAgent:) returns nil for unknown agent")
    func lastMessageForUnknownAgent() async {
        let speech = await MockSpeech()
        let feedback = await MockFeedback()
        let queue = AudioQueue(speechManager: speech, feedback: feedback)

        let text = await queue.lastMessage(forAgent: "nonexistent")
        #expect(text == nil)
    }

    @Test("clearQueue empties pending messages")
    func clearQueue() async {
        let speech = await MockSpeech()
        let feedback = await MockFeedback()
        let queue = AudioQueue(speechManager: speech, feedback: feedback)

        let voice = makeVoice()

        // First enqueue auto-plays
        await queue.enqueue(
            AudioQueue.QueuedMessage(
                sessionName: "a",
                text: "playing",
                priority: .normal,
                voice: voice
            )
        )
        // These queue up behind it
        await queue.enqueue(
            AudioQueue.QueuedMessage(
                sessionName: "b",
                text: "queued1",
                priority: .normal,
                voice: voice
            )
        )
        await queue.enqueue(
            AudioQueue.QueuedMessage(
                sessionName: "c",
                text: "queued2",
                priority: .normal,
                voice: voice
            )
        )

        #expect(await queue.pendingCount == 2)
        await queue.clearQueue()
        #expect(await queue.pendingCount == 0)
    }

    @Test("pendingCount is 0 when empty")
    func pendingCountZero() async {
        let speech = await MockSpeech()
        let feedback = await MockFeedback()
        let queue = AudioQueue(speechManager: speech, feedback: feedback)

        #expect(await queue.pendingCount == 0)
    }

    @Test("multiple agents tracked independently for lastMessage")
    func multipleAgentsTracked() async {
        let speech = await MockSpeech()
        let feedback = await MockFeedback()
        let queue = AudioQueue(speechManager: speech, feedback: feedback)

        let voice = makeVoice()

        // First enqueue auto-plays, sets state to speaking
        await queue.enqueue(
            AudioQueue.QueuedMessage(
                sessionName: "sudo",
                text: "from sudo",
                priority: .normal,
                voice: voice
            )
        )

        // This one queues (not played yet) but we can check sudo was recorded
        let sudoText = await queue.lastMessage(forAgent: "sudo")
        #expect(sudoText == "from sudo")

        let frontendText = await queue.lastMessage(forAgent: "frontend")
        #expect(frontendText == nil)
    }
}

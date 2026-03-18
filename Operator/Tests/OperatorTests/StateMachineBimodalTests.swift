import AVFoundation
import Foundation
import Testing

@testable import OperatorCore

/// Namespace for StateMachine bimodal tests.
private enum StateMachineBimodalTests {}

// MARK: - Mocks (internal to this test file; E2E mocks live in Tests/E2E/)

@MainActor
private final class TestSpeechTranscriber: SpeechTranscribing {
    private(set) var isListening = false
    private(set) var lastAudioFileURL: URL?
    var nextTranscription: String?

    func startListening(contextualStrings: [String] = []) throws {
        isListening = true
    }

    func stopListening() async -> String? {
        isListening = false
        let result = nextTranscription
        nextTranscription = nil
        return result
    }

    func stopListeningWithTimeout(seconds _: TimeInterval) async -> String? {
        await stopListening()
    }

    func replaceEngine(_ newEngine: any TranscriptionEngine) {}
}

@MainActor
private final class TestSpeechManager: SpeechManaging {
    let isSpeaking = false
    let finishedSpeaking: AsyncStream<Void>
    private let finishedContinuation: AsyncStream<Void>.Continuation
    var spokenMessages: [(text: String, prefix: String)] = []

    init() {
        let (stream, continuation) = AsyncStream<Void>.makeStream()
        self.finishedSpeaking = stream
        self.finishedContinuation = continuation
    }

    func speak(_ text: String, voice _: VoiceDescriptor, prefix: String, pitchMultiplier _: Float) {
        spokenMessages.append((text: text, prefix: prefix))
        finishedContinuation.yield()
    }

    func interrupt() -> InterruptInfo {
        InterruptInfo(heardText: "", unheardText: "", session: "")
    }

    func stop() {}
}

// MARK: - Test Infrastructure

/// Reusable context for StateMachine bimodal tests.
@MainActor
private struct BimodalTestContext {
    let stateMachine: StateMachine
    let mockTranscriber: TestSpeechTranscriber
    let mockSpeechManager: TestSpeechManager
    let mockDictation: MockDictationDelivery
    let mockFeedback: MockAudioFeedback
    let registry: SessionRegistry
    let router: MessageRouter
}

@MainActor
private func makeBimodalContext(
    isTerminal: Bool = false,
    isTextField: Bool = true,
    hasPermission: Bool = true,
    bundleID: String? = "com.example.app"
) -> BimodalTestContext {
    let mockAccessibility = MockAccessibilityQuery(
        bundleID: bundleID,
        isTerminal: isTerminal,
        isTextField: isTextField,
        hasPermission: hasPermission
    )
    let transcriber = TestSpeechTranscriber()
    let speechMgr = TestSpeechManager()
    let feedback = MockAudioFeedback()
    let dictation = MockDictationDelivery()
    let vm = VoiceManager()
    let reg = SessionRegistry(voiceManager: vm)
    let rt = MessageRouter(registry: reg)
    return buildContext(
        transcriber: transcriber,
        speechMgr: speechMgr,
        feedback: feedback,
        dictation: dictation,
        accessibility: mockAccessibility,
        vm: vm,
        reg: reg,
        rt: rt
    )
}

@MainActor
private func buildContext(
    transcriber: TestSpeechTranscriber,
    speechMgr: TestSpeechManager,
    feedback: MockAudioFeedback,
    dictation: MockDictationDelivery,
    accessibility: MockAccessibilityQuery,
    vm: VoiceManager,
    reg: SessionRegistry,
    rt: MessageRouter
) -> BimodalTestContext {
    let aq = AudioQueue(speechManager: speechMgr, feedback: feedback)
    let bde = BimodalDecisionEngine(
        accessibilityQuery: accessibility,
        router: rt,
        registry: reg
    )
    let sm = StateMachine(
        transcriber: transcriber,
        audioQueue: aq,
        router: rt,
        feedback: feedback,
        terminalBridge: MockTerminalBridge(),
        registry: reg,
        voiceManager: vm,
        waveformPanel: nil,
        speechManager: speechMgr,
        bimodalEngine: bde,
        dictationDelivery: dictation
    )
    sm.minimumRecordingDuration = .zero
    return BimodalTestContext(
        stateMachine: sm,
        mockTranscriber: transcriber,
        mockSpeechManager: speechMgr,
        mockDictation: dictation,
        mockFeedback: feedback,
        registry: reg,
        router: rt
    )
}

// MARK: - Bimodal Decision Flow Tests

@Suite("StateMachine - Bimodal Dictation Flow")
@MainActor
internal struct StateMachineBimodalDictationTests {
    @Test("transcription with non-terminal text field triggers dictation delivery")
    func transcriptionToDictation() async throws {
        let ctx = makeBimodalContext(isTerminal: false, isTextField: true)

        ctx.mockTranscriber.nextTranscription = "hello world"
        ctx.stateMachine.triggerStart()
        #expect(ctx.stateMachine.currentState == .listening)

        ctx.stateMachine.triggerStop()
        #expect(ctx.stateMachine.currentState == .transcribing)

        // Allow async transcription and bimodal decision to complete
        try await Task.sleep(nanoseconds: 500_000_000)

        #expect(ctx.mockDictation.deliverCallCount == 1)
        #expect(ctx.mockDictation.lastDeliveredText == "hello world")
        #expect(ctx.mockFeedback.playedCues.contains(.processing))
        #expect(ctx.mockFeedback.playedCues.contains(.dictation))
        #expect(ctx.stateMachine.currentState == .idle)
    }

    @Test("transcription with non-terminal no text field produces error")
    func transcriptionNoTextFieldError() async throws {
        let ctx = makeBimodalContext(isTerminal: false, isTextField: false)

        ctx.mockTranscriber.nextTranscription = "some message"
        ctx.stateMachine.triggerStart()
        ctx.stateMachine.triggerStop()

        try await Task.sleep(nanoseconds: 500_000_000)

        #expect(ctx.mockDictation.deliverCallCount == 0)
        #expect(ctx.mockFeedback.playedCues.contains(.error))
        #expect(ctx.stateMachine.currentState == .idle)
        // Should have spoken an error message about no text field
        #expect(
            ctx.mockSpeechManager.spokenMessages.contains {
                $0.text.contains("text field")
            }
        )
    }

    @Test("transcription with permission denied produces error")
    func transcriptionPermissionDenied() async throws {
        let ctx = makeBimodalContext(hasPermission: false)

        ctx.mockTranscriber.nextTranscription = "test message"
        ctx.stateMachine.triggerStart()
        ctx.stateMachine.triggerStop()

        try await Task.sleep(nanoseconds: 500_000_000)

        #expect(ctx.mockDictation.deliverCallCount == 0)
        #expect(ctx.stateMachine.currentState == .idle)
        #expect(
            ctx.mockSpeechManager.spokenMessages.contains {
                $0.text.contains("Accessibility")
            }
        )
    }
}

@Suite("StateMachine - Bimodal Agent Routing Flow")
@MainActor
internal struct StateMachineBimodalRoutingTests {
    @Test("transcription with keyword match routes to agent")
    func transcriptionKeywordRoutesToAgent() async throws {
        let ctx = makeBimodalContext(isTerminal: false, isTextField: true)
        await ctx.registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)

        ctx.mockTranscriber.nextTranscription = "tell sudo to fix the build"
        ctx.stateMachine.triggerStart()
        ctx.stateMachine.triggerStop()

        try await Task.sleep(nanoseconds: 200_000_000)

        // Should NOT have called dictation delivery
        #expect(ctx.mockDictation.deliverCallCount == 0)
        // Should have entered routing -> delivering pipeline
        // State should eventually settle to idle after delivery attempt
        #expect(ctx.stateMachine.currentState == .idle)
    }

    @Test("transcription with terminal single session routes to agent")
    func transcriptionTerminalSingleSessionRoutes() async throws {
        let ctx = makeBimodalContext(
            isTerminal: true,
            isTextField: false,
            bundleID: "com.googlecode.iterm2"
        )
        await ctx.registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)

        ctx.mockTranscriber.nextTranscription = "run the test suite"
        ctx.stateMachine.triggerStart()
        ctx.stateMachine.triggerStop()

        try await Task.sleep(nanoseconds: 200_000_000)

        #expect(ctx.mockDictation.deliverCallCount == 0)
        #expect(ctx.stateMachine.currentState == .idle)
    }

    @Test("active affinity routes to agent even from non-terminal")
    func affinityRoutesToAgentFromNonTerminal() async throws {
        let ctx = makeBimodalContext(isTerminal: false, isTextField: true)
        await ctx.registry.register(name: "sudo", tty: "/dev/ttys001", cwd: "/tmp", context: nil)

        // Simulate a previous route to establish affinity
        // Access routing state by doing a complete routing cycle first
        ctx.mockTranscriber.nextTranscription = "tell sudo fix the build"
        ctx.stateMachine.triggerStart()
        ctx.stateMachine.triggerStop()
        try await Task.sleep(nanoseconds: 200_000_000)

        // Now the follow-up should route via affinity
        ctx.mockTranscriber.nextTranscription = "actually, use flexbox instead"
        ctx.stateMachine.triggerStart()
        ctx.stateMachine.triggerStop()
        try await Task.sleep(nanoseconds: 200_000_000)

        #expect(ctx.mockDictation.deliverCallCount == 0)
        #expect(ctx.stateMachine.currentState == .idle)
    }
}

@Suite("StateMachine - Double-Tap Replay")
@MainActor
internal struct StateMachineDoubleTapTests {
    @Test("double-tap replays last dictation successfully")
    func doubleTapReplaySuccess() async throws {
        let ctx = makeBimodalContext(isTerminal: false, isTextField: true)

        // First, deliver some dictation to establish lastDictation
        _ = await ctx.mockDictation.deliver("meeting at 3pm")

        // Double-tap
        ctx.stateMachine.triggerDoubleTap()

        try await Task.sleep(nanoseconds: 500_000_000)

        #expect(ctx.mockDictation.replayCallCount == 1)
        #expect(ctx.mockFeedback.playedCues.contains(.dictation))
        #expect(ctx.stateMachine.currentState == .idle)
    }

    @Test("double-tap with no lastDictation plays error tone")
    func doubleTapNoLastDictationError() async throws {
        let ctx = makeBimodalContext()

        // No prior dictation
        ctx.stateMachine.triggerDoubleTap()

        try await Task.sleep(nanoseconds: 500_000_000)

        #expect(ctx.mockDictation.replayCallCount == 1)
        #expect(ctx.mockFeedback.playedCues.contains(.error))
        #expect(ctx.stateMachine.currentState == .idle)
    }

    @Test("double-tap from LISTENING cancels listening and replays")
    func doubleTapFromListeningCancelsAndReplays() async throws {
        let ctx = makeBimodalContext()

        // Establish lastDictation
        _ = await ctx.mockDictation.deliver("previous text")

        // Start listening
        ctx.stateMachine.triggerStart()
        #expect(ctx.stateMachine.currentState == .listening)

        // Double-tap should cancel listening and replay
        ctx.stateMachine.triggerDoubleTap()

        try await Task.sleep(nanoseconds: 500_000_000)

        #expect(ctx.mockDictation.replayCallCount == 1)
        #expect(ctx.stateMachine.currentState == .idle)
    }
}

@Suite("StateMachine - Dictation Feedback")
@MainActor
internal struct StateMachineDictationFeedbackTests {
    @Test("successful dictation plays processing then dictation tone, no spoken confirmation")
    func dictationPlaysBothTones() async throws {
        let ctx = makeBimodalContext(isTerminal: false, isTextField: true)

        ctx.mockTranscriber.nextTranscription = "hello world"
        ctx.stateMachine.triggerStart()
        ctx.stateMachine.triggerStop()

        try await Task.sleep(nanoseconds: 500_000_000)

        #expect(ctx.mockFeedback.playedCues.contains(.processing))
        #expect(ctx.mockFeedback.playedCues.contains(.dictation))
        // Should NOT have spoken a delivery confirmation
        let hasDeliveryConfirmation = ctx.mockSpeechManager.spokenMessages.contains {
            $0.text.contains("Sent to") || $0.text.contains("Delivered to")
        }
        #expect(!hasDeliveryConfirmation)
    }

    @Test("dictation paste failure speaks error message")
    func dictationPasteFailureSpeaksError() async throws {
        let ctx = makeBimodalContext(isTerminal: false, isTextField: true)
        ctx.mockDictation.deliverResult = .pasteFailed("CGEvent creation failed")

        ctx.mockTranscriber.nextTranscription = "test message"
        ctx.stateMachine.triggerStart()
        ctx.stateMachine.triggerStop()

        try await Task.sleep(nanoseconds: 500_000_000)

        #expect(ctx.mockFeedback.playedCues.contains(.error))
        #expect(
            ctx.mockSpeechManager.spokenMessages.contains {
                $0.text.contains("cursor")
            }
        )
        #expect(ctx.stateMachine.currentState == .idle)
    }

    @Test("empty transcription returns to idle with dismissed tone")
    func emptyTranscriptionReturnsToIdle() async throws {
        let ctx = makeBimodalContext()

        ctx.mockTranscriber.nextTranscription = nil
        ctx.stateMachine.triggerStart()
        ctx.stateMachine.triggerStop()

        try await Task.sleep(nanoseconds: 500_000_000)

        #expect(ctx.mockDictation.deliverCallCount == 0)
        #expect(ctx.stateMachine.currentState == .idle)
        #expect(ctx.mockFeedback.playedCues.contains(.dismissed))
    }
}

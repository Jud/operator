import AVFoundation
import Testing

@testable import OperatorCore

/// Namespace matching the file name to satisfy the file_name lint rule.
private enum AdaptiveSpeechManagerTests {}

// MARK: - Engine Switching

@Suite("AdaptiveSpeechManager - Engine Switching", .serialized)
internal struct AdaptiveTTSSwitchingTests {
    private func makeVoice() -> VoiceDescriptor {
        VoiceDescriptor(appleVoice: AVSpeechSynthesisVoice(), qwenSpeakerID: "ryan")
    }

    @MainActor
    private func makeEngine(preferLocal: Bool = true) -> AdaptiveSpeechManager {
        let cacheDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("asm-test-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)

        let modelManager = ModelManager(cacheDirectory: cacheDir)
        let localEngine = Qwen3TTSSpeechManager(modelManager: modelManager)
        let fallbackEngine = SpeechManager()

        return AdaptiveSpeechManager(
            localEngine: localEngine,
            fallbackEngine: fallbackEngine,
            modelManager: modelManager,
            preferLocal: preferLocal
        )
    }

    @Test("isUsingLocal reflects preference when no fallback has occurred")
    @MainActor
    func isUsingLocalReflectsPreference() {
        let engine = makeEngine(preferLocal: true)
        #expect(engine.isUsingLocal == true)

        engine.setUseLocal(false)
        #expect(engine.isUsingLocal == false)
    }

    @Test("setUseLocal toggles engine preference")
    @MainActor
    func setUseLocalToggles() {
        let engine = makeEngine(preferLocal: false)
        #expect(engine.isUsingLocal == false)

        engine.setUseLocal(true)
        #expect(engine.isUsingLocal == true)

        engine.setUseLocal(false)
        #expect(engine.isUsingLocal == false)
    }

    @Test("preferLocal false starts with Apple engine active")
    @MainActor
    func preferLocalFalseStartsWithApple() {
        let engine = makeEngine(preferLocal: false)
        #expect(engine.isUsingLocal == false)
    }

    @Test("speak delegates to fallback when useLocal is false")
    @MainActor
    func speakDelegatesToFallback() {
        let engine = makeEngine(preferLocal: false)
        let voice = makeVoice()
        engine.speak("hello", voice: voice, prefix: "test", pitchMultiplier: 1.0)
        // Verifying no crash when fallback engine handles the speak call
    }

    @Test("stop does not crash regardless of engine state")
    @MainActor
    func stopDoesNotCrash() {
        let engine = makeEngine(preferLocal: true)
        engine.stop()
    }

    @Test("interrupt returns InterruptInfo regardless of engine state")
    @MainActor
    func interruptReturnsInfo() {
        let engine = makeEngine(preferLocal: true)
        let info = engine.interrupt()
        // When nothing is playing, the result is valid (no crash)
        _ = info
    }
}

// MARK: - Fallback Behavior

@Suite("AdaptiveSpeechManager - Fallback Behavior", .serialized)
internal struct AdaptiveTTSFallbackTests {
    @MainActor
    private func makeEngine(preferLocal: Bool = true) -> AdaptiveSpeechManager {
        let cacheDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("asm-test-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)

        let modelManager = ModelManager(cacheDirectory: cacheDir)
        let localEngine = Qwen3TTSSpeechManager(modelManager: modelManager)
        let fallbackEngine = SpeechManager()

        return AdaptiveSpeechManager(
            localEngine: localEngine,
            fallbackEngine: fallbackEngine,
            modelManager: modelManager,
            preferLocal: preferLocal
        )
    }

    @Test("triggerFallback sets didFallBack and invokes onFallback callback")
    @MainActor
    func triggerFallbackSetsState() {
        let engine = makeEngine(preferLocal: true)

        var callbackMessage = ""
        engine.onFallback = { message in
            callbackMessage = message
        }

        engine.triggerFallback()
        #expect(engine.isUsingLocal == false)
        #expect(callbackMessage.contains("Falling back"))
        #expect(callbackMessage.contains("Apple"))
    }

    @Test("triggerFallback is idempotent - second call does not invoke callback again")
    @MainActor
    func triggerFallbackIdempotent() {
        let engine = makeEngine(preferLocal: true)

        var callbackCount = 0
        engine.onFallback = { _ in
            callbackCount += 1
        }

        engine.triggerFallback()
        engine.triggerFallback()
        #expect(callbackCount == 1)
    }

    @Test("sticky fallback stays on Apple until setUseLocal(true)")
    @MainActor
    func stickyFallback() {
        let engine = makeEngine(preferLocal: true)
        engine.onFallback = { _ in }

        engine.triggerFallback()
        #expect(engine.isUsingLocal == false)

        // Still false after checking again
        #expect(engine.isUsingLocal == false)
    }

    @Test("setUseLocal(true) resets fallback state")
    @MainActor
    func setUseLocalResetsFallback() {
        let engine = makeEngine(preferLocal: true)
        engine.onFallback = { _ in }

        engine.triggerFallback()
        #expect(engine.isUsingLocal == false)

        engine.setUseLocal(true)
        #expect(engine.isUsingLocal == true)
    }

    @Test("onFallback callback receives descriptive message mentioning Apple")
    @MainActor
    func fallbackMessageContent() {
        let engine = makeEngine(preferLocal: true)

        var receivedMessage = ""
        engine.onFallback = { message in
            receivedMessage = message
        }

        engine.triggerFallback()
        #expect(receivedMessage.contains("text-to-speech"))
        #expect(receivedMessage.contains("Local model unavailable"))
    }

    @Test("finishedSpeaking stream is available")
    @MainActor
    func finishedSpeakingStreamExists() {
        let engine = makeEngine(preferLocal: true)
        _ = engine.finishedSpeaking
    }
}

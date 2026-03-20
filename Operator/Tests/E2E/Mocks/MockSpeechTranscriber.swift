import Foundation
import OperatorCore

/// Mock speech transcriber that returns scripted transcriptions for E2E testing.
///
/// Set `nextTranscription` before each trigger cycle to control what text
/// the StateMachine receives from "transcription". The mock immediately
/// returns the scripted text without any audio processing.
@MainActor
internal final class MockSpeechTranscriber: SpeechTranscribing {
    private(set) var isListening = false
    private(set) var lastAudioFileURL: URL?

    /// The text to return from the next stopListening() / stopListeningWithTimeout() call.
    ///
    /// Set this before simulating a trigger stop to script the user's "speech".
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

    func stopListeningWithTimeout(seconds: TimeInterval) async -> String? {
        await stopListening()
    }

    func stopAndCheckSilence() async -> Bool {
        isListening = false
        return nextTranscription != nil
    }

    func finishTranscription() async -> String? {
        let result = nextTranscription
        nextTranscription = nil
        return result
    }

    func replaceEngine(_ newEngine: any TranscriptionEngine) {}
}

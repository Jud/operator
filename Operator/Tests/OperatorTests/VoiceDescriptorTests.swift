import AVFoundation
import Testing

@testable import OperatorCore

@Suite("VoiceDescriptor")
internal struct VoiceDescriptorTests {
    // MARK: - Construction

    @Test("creation stores Apple voice")
    func creationStoresFields() {
        let appleVoice = AVSpeechSynthesisVoice()
        let descriptor = VoiceDescriptor(appleVoice: appleVoice)

        #expect(descriptor.appleVoice.identifier == appleVoice.identifier)
    }

    // MARK: - Equality

    @Test("equal when Apple voice identifiers match")
    func equalWhenBothMatch() {
        let appleVoice = AVSpeechSynthesisVoice()
        let descriptorA = VoiceDescriptor(appleVoice: appleVoice)
        let descriptorB = VoiceDescriptor(appleVoice: appleVoice)

        #expect(descriptorA == descriptorB)
    }

    @Test("not equal when Apple voice identifiers differ")
    func notEqualDifferentAppleVoice() {
        let voices = AVSpeechSynthesisVoice.speechVoices().filter { $0.language.hasPrefix("en") }
        guard voices.count >= 2 else {
            return
        }

        let first = VoiceDescriptor(appleVoice: voices[0])
        let second = VoiceDescriptor(appleVoice: voices[1])

        #expect(first != second)
    }

    // MARK: - Sendable

    @Test("VoiceDescriptor is Sendable (compile-time check)")
    func sendableCompliance() async {
        let descriptor = VoiceDescriptor(appleVoice: AVSpeechSynthesisVoice())

        let value = await Task.detached { descriptor }.value
        #expect(value.appleVoice.identifier == descriptor.appleVoice.identifier)
    }
}

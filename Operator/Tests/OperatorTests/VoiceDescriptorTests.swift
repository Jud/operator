import AVFoundation
import Testing

@testable import OperatorCore

@Suite("VoiceDescriptor")
internal struct VoiceDescriptorTests {
    // MARK: - Construction

    @Test("creation stores both Apple voice and Qwen speaker ID")
    func creationStoresFields() {
        let appleVoice = AVSpeechSynthesisVoice()
        let descriptor = VoiceDescriptor(appleVoice: appleVoice, qwenSpeakerID: "ryan")

        #expect(descriptor.appleVoice.identifier == appleVoice.identifier)
        #expect(descriptor.qwenSpeakerID == "ryan")
    }

    @Test("different Qwen speaker IDs create distinct descriptors")
    func differentQwenSpeakerIDs() {
        let appleVoice = AVSpeechSynthesisVoice()
        let descriptorA = VoiceDescriptor(appleVoice: appleVoice, qwenSpeakerID: "ryan")
        let descriptorB = VoiceDescriptor(appleVoice: appleVoice, qwenSpeakerID: "emma")

        #expect(descriptorA != descriptorB)
    }

    // MARK: - Equality

    @Test("equal when both Apple voice identifier and Qwen speaker ID match")
    func equalWhenBothMatch() {
        let appleVoice = AVSpeechSynthesisVoice()
        let descriptorA = VoiceDescriptor(appleVoice: appleVoice, qwenSpeakerID: "vivian")
        let descriptorB = VoiceDescriptor(appleVoice: appleVoice, qwenSpeakerID: "vivian")

        #expect(descriptorA == descriptorB)
    }

    @Test("not equal when Qwen speaker IDs differ")
    func notEqualDifferentQwen() {
        let appleVoice = AVSpeechSynthesisVoice()
        let first = VoiceDescriptor(appleVoice: appleVoice, qwenSpeakerID: "lucas")
        let second = VoiceDescriptor(appleVoice: appleVoice, qwenSpeakerID: "olivia")

        #expect(first != second)
    }

    @Test("not equal when Apple voice identifiers differ")
    func notEqualDifferentAppleVoice() {
        let voices = AVSpeechSynthesisVoice.speechVoices().filter { $0.language.hasPrefix("en") }
        guard voices.count >= 2 else {
            return
        }

        let first = VoiceDescriptor(appleVoice: voices[0], qwenSpeakerID: "ryan")
        let second = VoiceDescriptor(appleVoice: voices[1], qwenSpeakerID: "ryan")

        #expect(first != second)
    }

    // MARK: - Sendable

    @Test("VoiceDescriptor is Sendable (compile-time check)")
    func sendableCompliance() async {
        let descriptor = VoiceDescriptor(
            appleVoice: AVSpeechSynthesisVoice(),
            qwenSpeakerID: "ethan"
        )

        let value = await Task.detached { descriptor }.value
        #expect(value.qwenSpeakerID == "ethan")
    }
}

import AVFoundation
import Testing

@testable import OperatorCore

@Suite("VoiceDescriptor")
internal struct VoiceDescriptorTests {
    // MARK: - Construction

    @Test("system voice stores Apple voice")
    func systemVoiceStoresFields() {
        let appleVoice = AVSpeechSynthesisVoice()
        let descriptor = VoiceDescriptor.system(appleVoice)

        #expect(descriptor.appleVoice?.identifier == appleVoice.identifier)
    }

    @Test("kokoro voice stores name")
    func kokoroVoiceStoresName() {
        let descriptor = VoiceDescriptor.kokoro(name: "af_heart")
        #expect(descriptor.kokoroName == "af_heart")
        #expect(descriptor.isKokoro)
    }

    // MARK: - Equality

    @Test("equal when both system voices match")
    func equalWhenBothMatch() {
        let appleVoice = AVSpeechSynthesisVoice()
        let descriptorA = VoiceDescriptor.system(appleVoice)
        let descriptorB = VoiceDescriptor.system(appleVoice)

        #expect(descriptorA == descriptorB)
    }

    @Test("equal when both kokoro names match")
    func equalKokoroNames() {
        let voice1 = VoiceDescriptor.kokoro(name: "af_heart")
        let voice2 = VoiceDescriptor.kokoro(name: "af_heart")
        #expect(voice1 == voice2)
    }

    @Test("not equal when Apple voice identifiers differ")
    func notEqualDifferentAppleVoice() {
        let voices = AVSpeechSynthesisVoice.speechVoices().filter { $0.language.hasPrefix("en") }
        guard voices.count >= 2 else {
            return
        }

        let first = VoiceDescriptor.system(voices[0])
        let second = VoiceDescriptor.system(voices[1])

        #expect(first != second)
    }

    @Test("not equal kokoro vs system")
    func notEqualMixed() {
        let kokoro = VoiceDescriptor.kokoro(name: "af_heart")
        let system = VoiceDescriptor.system(AVSpeechSynthesisVoice())
        #expect(kokoro != system)
    }

    // MARK: - Sendable

    @Test("VoiceDescriptor is Sendable (compile-time check)")
    func sendableCompliance() async {
        let descriptor = VoiceDescriptor.system(AVSpeechSynthesisVoice())

        let value = await Task.detached { descriptor }.value
        #expect(value.appleVoice?.identifier == descriptor.appleVoice?.identifier)
    }

    // MARK: - Display Name

    @Test("display name for kokoro voice")
    func displayNameKokoro() {
        let descriptor = VoiceDescriptor.kokoro(name: "am_adam")
        #expect(descriptor.displayName == "am_adam")
    }
}

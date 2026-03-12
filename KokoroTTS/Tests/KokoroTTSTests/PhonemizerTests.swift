import Testing
@testable import KokoroTTS

@Suite("Phonemizer")
struct PhonemizerTests {
    @Test("Special cases produce correct phonemes")
    func specialCases() {
        let phonemizer = Phonemizer()
        #expect(phonemizer.textToPhonemes("the").contains("ð"))
        #expect(phonemizer.textToPhonemes("a") == "ɐ")
        #expect(phonemizer.textToPhonemes("I") == "aɪ")
    }

    @Test("Contraction expansion")
    func contractions() {
        let phonemizer = Phonemizer()
        // "don't" should be expanded to "do not" — result should not contain apostrophe
        let result = phonemizer.textToPhonemes("don't")
        // After expansion: "do not" → phonemes for "do" + " " + phonemes for "not"
        #expect(result.contains(" "))
    }

    @Test("Punctuation preserved as phoneme tokens")
    func punctuation() {
        let phonemizer = Phonemizer()
        let result = phonemizer.textToPhonemes("hello, world.")
        #expect(result.contains(","))
        #expect(result.contains("."))
    }

    @Test("Empty input")
    func emptyInput() {
        let phonemizer = Phonemizer()
        let result = phonemizer.textToPhonemes("")
        #expect(result.isEmpty)
    }

    @Test("Multiple spaces collapsed")
    func multipleSpaces() {
        let phonemizer = Phonemizer()
        let result = phonemizer.textToPhonemes("hello    world")
        // Should have at most one space between words
        #expect(!result.contains("  "))
    }
}

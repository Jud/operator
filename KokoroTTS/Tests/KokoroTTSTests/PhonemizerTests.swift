import Testing
@testable import KokoroTTS

@Suite("Phonemizer")
struct PhonemizerTests {
    @Test("Special cases produce correct phonemes")
    func specialCases() {
        let phonemizer = Phonemizer()
        #expect(phonemizer.textToPhonemes("the").contains("ð"))
        // Standalone "a" without context → letter name "eɪ"; in "a cat" → determiner "ɐ"
        #expect(phonemizer.textToPhonemes("a") == "eɪ")
        #expect(phonemizer.textToPhonemes("I") == "aɪ")
    }

    @Test("Number expansion")
    func numberExpansion() {
        let phonemizer = Phonemizer()
        // "42" should be expanded to "forty-two" before phonemization
        let result = phonemizer.textToPhonemes("42")
        // Should contain phonemes, not raw digits
        #expect(!result.contains("4"))
        #expect(!result.contains("2"))
        #expect(!result.isEmpty)
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

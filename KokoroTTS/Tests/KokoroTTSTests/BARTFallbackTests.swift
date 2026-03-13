import Foundation
import Testing
@testable import KokoroTTS

@Suite("BARTFallback")
struct BARTFallbackTests {
    @Test("BART loads from bundle")
    func loads() {
        #expect(BARTFallback.fromBundle() != nil)
    }

    @Test("Known words match Python reference (greedy decoding is deterministic)")
    func referenceMatch() throws {
        let bart = try #require(BARTFallback.fromBundle())
        let expected: [(String, String)] = [
            ("hello", "hˈɛlO"),
            ("cat", "kˈæt"),
            ("world", "wˈɜɹld"),
        ]
        for (word, ref) in expected {
            let result = bart.predict(word)
            #expect(result == ref, "BART(\(word)): got \(result ?? "nil"), expected \(ref)")
        }
    }

    @Test("OOV technical terms produce valid phonemes via G2P pipeline")
    func oovThroughG2P() {
        let g2p = EnglishG2P(british: false)
        let words = ["kubernetes", "nginx", "graphql", "anthropic"]
        for word in words {
            let (phonemes, _) = g2p.phonemize(text: word)
            #expect(!phonemes.isEmpty, "\(word) produced empty phonemes")
            #expect(!phonemes.contains("❓"), "\(word) hit unknown fallback")
        }
    }

    @Test("Edge cases: empty and too-long inputs")
    func edgeCases() throws {
        let bart = try #require(BARTFallback.fromBundle())
        #expect(bart.predict("") == nil)
        let tooLong = String(repeating: "a", count: 100)
        #expect(bart.predict(tooLong) == nil)
    }
}

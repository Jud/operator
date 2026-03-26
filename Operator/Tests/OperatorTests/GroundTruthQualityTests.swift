import Testing

@testable import OperatorCore

/// Tests for the ground truth quality gate to prevent regressions
/// in how batch decode results are assembled and compared.
internal enum GroundTruthQualityTests {
    @Suite("Ground Truth - VAD Chunk Assembly")
    internal struct ChunkAssemblyTests {
        @Test("multiple VAD chunks must be joined, not just first taken")
        func multipleChunksJoined() {
            // Simulates what VAD chunking returns for a long recording:
            // multiple text segments that must be concatenated.
            let chunks = [
                "I think we should do this",
                "and also maybe that",
                "and then finally the last part"
            ]

            // This is the pattern that was broken — results.first only got chunk 1.
            let brokenFirstOnly = chunks.first ?? ""
            let correctJoined = chunks.joined(separator: " ")

            // The joined version must contain all chunks.
            for chunk in chunks {
                #expect(correctJoined.contains(chunk))
            }

            // The first-only version must NOT contain later chunks.
            #expect(!brokenFirstOnly.contains("also maybe that"))
            #expect(!brokenFirstOnly.contains("finally the last part"))

            // Similarity between first-only and full text should be low
            // (this is what triggered the false 33% quality alert).
            let similarity = WhisperKitEngine.levenshteinSimilarity(
                correctJoined,
                brokenFirstOnly
            )
            #expect(similarity < 0.5, "Taking only first chunk should produce low similarity")
        }

        @Test("single VAD chunk works correctly")
        func singleChunk() {
            let chunks = ["Hello world this is a short utterance"]
            let joined = chunks.joined(separator: " ")
            #expect(joined == chunks[0])
        }

        @Test("empty chunks produce empty string")
        func emptyChunks() {
            let chunks: [String] = []
            let joined = chunks.joined(separator: " ")
            #expect(joined.isEmpty)
        }
    }

    @Suite("Ground Truth - Levenshtein Similarity")
    internal struct LevenshteinTests {
        @Test("identical strings return 1.0")
        func identicalStrings() {
            let sim = WhisperKitEngine.levenshteinSimilarity("hello world", "hello world")
            #expect(sim == 1.0)
        }

        @Test("completely different strings return near 0.0")
        func completelyDifferent() {
            let sim = WhisperKitEngine.levenshteinSimilarity("aaa", "zzz")
            #expect(sim == 0.0)
        }

        @Test("partial match returns intermediate value")
        func partialMatch() {
            let sim = WhisperKitEngine.levenshteinSimilarity(
                "the quick brown fox",
                "the quick brown dog"
            )
            #expect(sim > 0.8)
            #expect(sim < 1.0)
        }

        @Test("empty strings return 1.0")
        func bothEmpty() {
            let sim = WhisperKitEngine.levenshteinSimilarity("", "")
            #expect(sim == 1.0)
        }

        @Test("one empty string returns 0.0")
        func oneEmpty() {
            let sim = WhisperKitEngine.levenshteinSimilarity("hello", "")
            #expect(sim == 0.0)
        }

        @Test("truncated text has low similarity to full text")
        func truncatedVsFull() {
            let full = "I think we should do this and also maybe that and finally the last part"
            let truncated = "I think we should do this"
            let sim = WhisperKitEngine.levenshteinSimilarity(full, truncated)
            #expect(sim < 0.5, "Truncated result must score below quality threshold")
        }
    }
}

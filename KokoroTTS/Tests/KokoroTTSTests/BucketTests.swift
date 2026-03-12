import Testing
@testable import KokoroTTS

@Suite("Bucket")
struct BucketTests {
    @Test("Unified bucket selection prefers smallest fit")
    func unifiedPreferSmallest() {
        let bucket = UnifiedBucket.select(forTokenCount: 100)
        #expect(bucket == .v21_5s)
        let large = UnifiedBucket.select(forTokenCount: 200)
        #expect(large == .v24_10s)
    }

    @Test("Unified bucket handles overflow")
    func unifiedOverflow() {
        let bucket = UnifiedBucket.select(forTokenCount: 500)
        #expect(bucket == nil)
    }

    @Test("All unified buckets have model names")
    func modelNames() {
        for bucket in UnifiedBucket.allCases {
            #expect(!bucket.modelName.isEmpty)
        }
    }

    @Test("Max tokens are positive and ordered")
    func maxTokensOrdered() {
        let sorted = UnifiedBucket.allCases.sorted(by: { $0.maxTokens < $1.maxTokens })
        for bucket in sorted {
            #expect(bucket.maxTokens > 0)
        }
        // Verify ordering: v21_5s < v21_10s < v24_10s
        #expect(UnifiedBucket.v21_5s.maxTokens < UnifiedBucket.v21_10s.maxTokens)
        #expect(UnifiedBucket.v21_10s.maxTokens < UnifiedBucket.v24_10s.maxTokens)
    }
}

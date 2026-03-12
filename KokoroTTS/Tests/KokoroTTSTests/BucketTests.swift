import Testing
@testable import KokoroTTS

@Suite("Bucket")
struct BucketTests {
    @Test("Bucket selection accepts valid token counts")
    func selectValid() {
        #expect(UnifiedBucket.select(forTokenCount: 100) == .v24_10s)
        #expect(UnifiedBucket.select(forTokenCount: 242) == .v24_10s)
    }

    @Test("Bucket selection rejects overflow")
    func selectOverflow() {
        #expect(UnifiedBucket.select(forTokenCount: 500) == nil)
    }

    @Test("Model name is set")
    func modelName() {
        #expect(!UnifiedBucket.v24_10s.modelName.isEmpty)
    }

    @Test("Max tokens matches constant")
    func maxTokens() {
        #expect(UnifiedBucket.v24_10s.maxTokens == UnifiedBucket.maxTokenCount)
    }
}

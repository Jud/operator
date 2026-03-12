import Testing
@testable import KokoroTTS

@Suite("Bucket")
struct BucketTests {
    @Test("Max token count is set")
    func maxTokenCount() {
        #expect(UnifiedBucket.maxTokenCount == 242)
    }

    @Test("Model name is set")
    func modelName() {
        #expect(!UnifiedBucket.modelName.isEmpty)
    }
}

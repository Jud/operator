import Testing
@testable import KokoroTTS

@Suite("Bucket")
struct BucketTests {
    @Test("Short text selects 3s bucket")
    func shortText() {
        let bucket = Bucket.select(forTokenCount: 20)
        #expect(bucket == .threeSecond)
    }

    @Test("Medium text selects 10s bucket")
    func mediumText() {
        let bucket = Bucket.select(forTokenCount: 100)
        #expect(bucket == .tenSecond)
    }

    @Test("Boundary: exactly at 3s limit")
    func boundary3s() {
        let bucket = Bucket.select(forTokenCount: 60)
        #expect(bucket == .threeSecond)
    }

    @Test("Boundary: just over 3s limit")
    func boundary3sOver() {
        let bucket = Bucket.select(forTokenCount: 61)
        #expect(bucket == .tenSecond)
    }

    @Test("Unified bucket selection prefers v24")
    func unifiedPreferV24() {
        let bucket = UnifiedBucket.select(forTokenCount: 100)
        #expect(bucket == .v24_10s)
    }

    @Test("Unified bucket handles overflow")
    func unifiedOverflow() {
        let bucket = UnifiedBucket.select(forTokenCount: 500)
        #expect(bucket == nil)
    }

    @Test("All HAR buckets have decoder model names")
    func decoderNames() {
        for bucket in Bucket.allCases {
            #expect(!bucket.decoderModelName.isEmpty)
        }
    }

    @Test("Max frames are positive")
    func maxFrames() {
        for bucket in Bucket.allCases {
            #expect(bucket.maxFrames > 0)
        }
    }
}

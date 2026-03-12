/// Unified (end-to-end) model bucket.
///
/// Each bucket corresponds to a CoreML model compiled for a fixed input shape.
enum UnifiedBucket: CaseIterable, Sendable, Hashable {
    case v24_10s

    /// Largest token count any bucket accepts.
    static let maxTokenCount = 242

    var modelName: String { "kokoro_24_10s" }

    var maxTokens: Int { 242 }

    /// Select a bucket that fits the given token count, or nil if too long.
    static func select(forTokenCount tokens: Int) -> UnifiedBucket? {
        tokens <= maxTokenCount ? .v24_10s : nil
    }
}

import Foundation

/// Two-stage HAR decoder bucket sizes.
///
/// Only 3s and 10s — sufficient for 1-3 sentence utterances.
public enum Bucket: CaseIterable, Sendable, Hashable {
    /// 3 second bucket (~72 frames, ~43,200 samples)
    case threeSecond
    /// 10 second bucket (~240 frames, ~240,000 samples)
    case tenSecond

    /// Frame count for text features at ~24fps.
    var maxFrames: Int {
        switch self {
        case .threeSecond: return 72
        case .tenSecond: return 240
        }
    }

    /// Maximum output audio samples at 24kHz.
    var maxSamples: Int {
        switch self {
        case .threeSecond: return 43_200   // 3s * 24000 * 0.6
        case .tenSecond: return 240_000    // 10s * 24000
        }
    }

    /// CoreML model filename (without extension).
    var decoderModelName: String {
        switch self {
        case .threeSecond: return "kokoro_har_3s"
        case .tenSecond: return "kokoro_har_10s"
        }
    }

    /// Token count threshold for bucket selection.
    var maxTokens: Int {
        switch self {
        case .threeSecond: return 60
        case .tenSecond: return 168
        }
    }

    /// Select the smallest bucket that fits the given token count.
    public static func select(forTokenCount tokens: Int) -> Bucket {
        if tokens <= Bucket.threeSecond.maxTokens {
            return .threeSecond
        }
        return .tenSecond
    }
}

/// Unified (end-to-end) model bucket sizes, matching FluidInference models.
///
/// Used as fallback when two-stage split models aren't available.
enum UnifiedBucket: CaseIterable, Sendable, Hashable {
    case v21_5s
    case v21_10s
    case v24_10s

    var modelName: String {
        switch self {
        case .v21_5s:  return "kokoro_21_5s"
        case .v21_10s: return "kokoro_21_10s"
        case .v24_10s: return "kokoro_24_10s"
        }
    }

    var maxTokens: Int {
        switch self {
        case .v21_5s:  return 124
        case .v21_10s: return 168
        case .v24_10s: return 242
        }
    }

    var maxSamples: Int {
        switch self {
        case .v21_5s:  return 175_800
        case .v21_10s: return 253_200
        case .v24_10s: return 240_000
        }
    }

    /// Select the best bucket for the given token count.
    ///
    /// Prefers v2.4 for quality. Falls back to v2.1 buckets only if v2.4 can't fit.
    static func select(forTokenCount tokens: Int) -> UnifiedBucket? {
        if tokens <= UnifiedBucket.v24_10s.maxTokens { return .v24_10s }
        // v2.1 buckets have smaller maxTokens than v2.4 (124, 168 < 242),
        // so this path is only reachable if a future bucket exceeds v2.4 capacity.
        return allCases.sorted(by: { $0.maxTokens < $1.maxTokens })
            .first { tokens <= $0.maxTokens }
    }
}

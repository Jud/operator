import Foundation

/// Unified (end-to-end) model bucket sizes, matching FluidInference models.
///
/// Each bucket corresponds to a CoreML model compiled for a fixed input shape.
/// Smaller buckets are faster — the engine selects the smallest that fits.
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

    /// Select the smallest bucket that fits the given token count.
    ///
    /// Prefers smaller models for speed — v21_5s for short utterances,
    /// v24_10s for longer text. Falls back to the largest available model.
    static func select(forTokenCount tokens: Int) -> UnifiedBucket? {
        allCases.sorted(by: { $0.maxTokens < $1.maxTokens })
            .first { tokens <= $0.maxTokens }
    }
}

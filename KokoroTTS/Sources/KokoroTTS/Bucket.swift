import Foundation

/// Unified (end-to-end) model bucket sizes.
///
/// Each bucket corresponds to a CoreML model compiled for a fixed input shape.
/// Smaller buckets are faster — the engine selects the smallest that fits.
enum UnifiedBucket: CaseIterable, Sendable, Hashable {
    case v21_5s
    case v21_10s
    case v24_10s

    /// Largest token count any bucket accepts.
    static let maxTokenCount = sortedBySize.last!.maxTokens

    /// Buckets sorted smallest-to-largest by token capacity.
    static let sortedBySize = allCases.sorted(by: { $0.maxTokens < $1.maxTokens })

    var modelName: String {
        switch self {
        case .v21_5s:  "kokoro_21_5s"
        case .v21_10s: "kokoro_21_10s"
        case .v24_10s: "kokoro_24_10s"
        }
    }

    var maxTokens: Int {
        switch self {
        case .v21_5s:  124
        case .v21_10s: 168
        case .v24_10s: 242
        }
    }

    /// Select the smallest bucket that fits the given token count.
    static func select(forTokenCount tokens: Int) -> UnifiedBucket? {
        sortedBySize.first { tokens <= $0.maxTokens }
    }
}

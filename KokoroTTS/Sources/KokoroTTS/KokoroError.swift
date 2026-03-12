import Foundation

/// Errors thrown by the KokoroTTS framework.
public enum KokoroError: Error, Sendable {
    /// CoreML model failed to load.
    case modelLoadFailed(String)
    /// CoreML inference returned unexpected output.
    case inferenceFailed(String)
    /// Requested voice not found.
    case voiceNotFound(String)
    /// Model files not present at expected path.
    case modelsNotAvailable(URL)
    /// Model download failed.
    case downloadFailed(String)
    /// Text too long for available model buckets.
    case textTooLong(tokenCount: Int, maxTokens: Int)
    /// Alignment math produced invalid output.
    case alignmentFailed(String)
}

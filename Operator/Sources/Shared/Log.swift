import OSLog

/// Centralized logger factory for the Operator subsystem.
///
/// Replaces per-file Logger construction with a single factory method,
/// ensuring consistent subsystem naming across all modules.
public enum Log {
    /// Create a logger for the given category within the Operator subsystem.
    ///
    /// - Parameter category: The logging category (e.g., "StateMachine", "AudioQueue").
    /// - Returns: A configured Logger instance.
    public static func logger(for category: String) -> Logger {
        Logger(subsystem: "com.operator.app", category: category)
    }
}

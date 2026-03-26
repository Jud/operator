import Foundation

/// Result of matching an explicit agent name in a user utterance.
///
/// Contains the canonical session name and the extracted message text.
public struct AgentNameMatch: Sendable {
    /// The canonical session name (preserving original casing from the registry).
    public let session: String

    /// The message text extracted after the agent name, or the full utterance if no message part was found.
    public let message: String
}

/// Shared utility for detecting explicit agent targeting in user utterances.
///
/// Matches the pattern "tell <name>", "hey <name>", or "@<name>" followed by
/// an optional message. Used by both MessageRouter (keyword extraction) and
/// InterruptionHandler (agent target detection) to avoid duplicating the regex
/// and matching logic.
public enum AgentNameMatcher {
    /// Regex pattern for explicit agent targeting in user utterances.
    ///
    /// Captures: group 1 = agent name, group 2 = remaining message text.
    private static let pattern: NSRegularExpression? = {
        try? NSRegularExpression(
            pattern: #"(?:tell |hey |@)(\w+)[,:]?\s*(.*)"#,
            options: [.caseInsensitive]
        )
    }()

    /// Attempt to match an explicit agent name in the given utterance.
    ///
    /// The extracted name is matched case-insensitively against the provided
    /// session names and nicknames. Nicknames are checked first since they are
    /// the short speakable names the user is most likely to say. If a match is
    /// found, the canonical session name is returned along with the message text.
    ///
    /// - Parameters:
    ///   - utterance: The user's utterance to search for an agent name pattern.
    ///   - sessionNames: The list of registered session names to match against.
    ///   - nicknames: Optional map of nickname → canonical session name for matching short spoken names.
    /// - Returns: An `AgentNameMatch` if a registered agent name was found, nil otherwise.
    public static func match(
        in utterance: String,
        sessionNames: [String],
        nicknames: [String: String] = [:]
    ) -> AgentNameMatch? {
        guard let pattern else {
            return nil
        }

        let range = NSRange(utterance.startIndex..., in: utterance)
        guard let result = pattern.firstMatch(in: utterance, range: range) else {
            return nil
        }

        guard let nameRange = Range(result.range(at: 1), in: utterance) else {
            return nil
        }

        let extractedName = String(utterance[nameRange]).lowercased()

        // Try nickname match first (more likely in voice interaction).
        let canonical: String? =
            nicknames.first { $0.key.lowercased() == extractedName }?.value
            ?? sessionNames.first { $0.lowercased() == extractedName }

        guard let canonical else {
            return nil
        }

        if let messageRange = Range(result.range(at: 2), in: utterance) {
            let message = String(utterance[messageRange]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !message.isEmpty {
                return AgentNameMatch(session: canonical, message: message)
            }
        }

        return AgentNameMatch(session: canonical, message: utterance)
    }
}

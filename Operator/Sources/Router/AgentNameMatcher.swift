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
    /// session names. If a match is found, the canonical name (preserving
    /// original casing) is returned along with the message text.
    ///
    /// - Parameters:
    ///   - utterance: The user's utterance to search for an agent name pattern.
    ///   - sessionNames: The list of registered session names to match against.
    /// - Returns: An `AgentNameMatch` if a registered agent name was found, nil otherwise.
    public static func match(
        in utterance: String,
        sessionNames: [String]
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

        guard let canonical = sessionNames.first(where: { $0.lowercased() == extractedName }) else {
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

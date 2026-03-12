import Foundation

// MARK: - Operator Command Matching

extension MessageRouter {
    /// Regex pattern for "what did [name] say" operator command.
    private static let whatDidSayPattern: NSRegularExpression? = {
        try? NSRegularExpression(
            pattern: #"what did (\w+) say"#,
            options: [.caseInsensitive]
        )
    }()

    /// Operator command patterns matched against the lowercased user utterance.
    private static let operatorCommandMatchers: [(pattern: String, command: OperatorCommand)] = [
        ("operator status", .status),
        ("operator what", .status),
        ("what's everyone working on", .status),
        ("whats everyone working on", .status),
        ("what is everyone working on", .status),
        ("list agents", .listAgents),
        ("list sessions", .listAgents),
        ("who's running", .listAgents),
        ("whos running", .listAgents),
        ("who is running", .listAgents),
        ("operator replay", .replay),
        ("what did i miss", .whatDidIMiss),
        ("operator help", .help)
    ]

    /// Find the canonical session name matching `name` case-insensitively.
    static func canonicalSession(
        _ name: String,
        in sessionNames: [String]
    ) -> String? {
        let lowered = name.lowercased()
        return sessionNames.first { $0.lowercased() == lowered }
    }

    /// Build the default clarification question from a list of session names.
    static func clarificationQuestion(for sessionNames: [String]) -> String {
        "Which agent is this for? \(sessionNames.joined(separator: " or "))?"
    }

    /// Build a `.clarify` result using the default question format.
    static func clarifyResult(
        candidates: [String],
        originalText: String
    ) -> RoutingResult {
        .clarify(
            candidates: candidates,
            question: clarificationQuestion(for: candidates),
            originalText: originalText
        )
    }

    /// Match the user's lowercased utterance against known operator command patterns.
    func matchOperatorCommand(_ lowered: String) -> OperatorCommand? {
        let range = NSRange(lowered.startIndex..., in: lowered)
        if let pattern = Self.whatDidSayPattern,
            let match = pattern.firstMatch(in: lowered, range: range),
            let nameRange = Range(match.range(at: 1), in: lowered)
        {
            let agentName = String(lowered[nameRange])
            if agentName != "i" {
                return .replayAgent(name: agentName)
            }
        }

        for matcher in Self.operatorCommandMatchers where lowered.contains(matcher.pattern) {
            return matcher.command
        }
        return nil
    }
}

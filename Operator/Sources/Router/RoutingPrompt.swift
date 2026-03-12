import Foundation

/// Modular prompt template for routing engines.
///
/// Extracts the routing prompt construction into a configurable module so that routing
/// heuristics can be iterated without code changes to the engine. Provides shared
/// context formatting used by both the deterministic heuristics and any future
/// routing engine (local model, Claude CLI, etc.).
public enum RoutingPrompt {
    private static let currentUserMessageHeading = "Current user message:"

    // MARK: - System Prompt

    /// System instruction for routing classification.
    public static let systemInstruction = """
        Route the user message to the best active session.
        Match using the provided session name, working directory, project context, and recent messages.
        Base the decision on those session details, not on assumptions \
        about common team names; if the match is weak or ambiguous, \
        return an empty session with confident=false.
        Reply only with JSON: {"session":"<name or empty>","confident":true|false}
        """

    // MARK: - JSON Schema

    /// JSON schema for routing engine output.
    public static let outputSchemaString = """
        {
          "type": "object",
          "properties": {
            "session": {
              "type": "string",
              "description": "Target session name, or empty string if ambiguous"
            },
            "confident": {
              "type": "boolean",
              "description": "Whether the routing decision is confident"
            }
          },
          "required": ["session", "confident"],
          "additionalProperties": false
        }
        """

    // MARK: - Prompt Construction

    /// Build the static routing context section shared by MessageRouter and benchmarks.
    ///
    /// This excludes the current user utterance so local-model routing can prefill and
    /// reuse the session snapshot across repeated routing calls.
    public static func buildContextPrompt(sessions: [SessionState]) -> String {
        var lines: [String] = [
            "Routing context:",
            "Active Claude Code sessions:"
        ]

        for (index, session) in sessions.enumerated() {
            var sessionLine = "\(index + 1). \"\(session.name)\" (\(session.cwd))"
            if !session.context.isEmpty {
                sessionLine += " - \(session.context)"
            }
            let recent = session.recentMessages.suffix(2)
            if !recent.isEmpty {
                let summary =
                    recent
                    .map { "\($0.role): \"\($0.text.prefix(60))\"" }
                    .joined(separator: " | ")
                sessionLine += ", recent: \(summary)"
            }
            lines.append(sessionLine)
        }

        lines.append("")
        lines.append("Use only the session details above to choose the best target.")
        return lines.joined(separator: "\n")
    }

    /// Build the routing request body shared by MessageRouter and benchmarks.
    public static func buildPrompt(
        text: String,
        sessions: [SessionState],
        routingState: RoutingState
    ) -> String {
        let context = buildContextPrompt(sessions: sessions)
        return "\(context)\n\n\(currentUserMessageHeading)\n\(text)"
    }

    /// Split a flat routing prompt into its reusable context and dynamic utterance.
    public static func splitPrompt(_ routingPrompt: String) -> (context: String, currentMessage: String)? {
        let marker = "\n\n\(currentUserMessageHeading)\n"
        guard let range = routingPrompt.range(of: marker) else {
            return nil
        }
        return (
            context: String(routingPrompt[..<range.lowerBound]),
            currentMessage: String(routingPrompt[range.upperBound...])
        )
    }
}

import Foundation

/// Modular prompt template for the local routing model.
///
/// Extracts the routing prompt construction into a configurable module so that routing
/// heuristics can be iterated without code changes to the engine. The local Qwen 3.5 0.8B
/// model receives the same session context and message as the Claude CLI routing, but
/// wrapped with a system instruction optimized for fast, direct classification.
///
/// The JSON output schema is defined here as the source of truth for both the grammar
/// constraint (in ``MLXRoutingEngine``) and the format expected by ``MessageRouter``.
public enum RoutingPrompt {
    private static let currentUserMessageHeading = "Current user message:"

    // MARK: - System Prompt

    /// System instruction optimized for the 0.8B model's classification task.
    ///
    /// Key characteristics:
    /// - Concise and direct to maximize accuracy from a small model
    /// - Explicitly states the JSON output format
    /// - Requests non-thinking mode for fast direct answers (Qwen 3 `/no_think`)
    public static let systemInstruction = """
        Route the user message to the best active session.
        Match using the provided session name, working directory, project context, and recent messages.
        Base the decision on those session details, not on assumptions about common team names; if the match is weak or ambiguous, return an empty session with confident=false.
        Reply only with JSON: {"session":"<name or empty>","confident":true|false}
        """

    // MARK: - JSON Schema

    /// Raw JSON schema string for grammar-constrained decoding.
    ///
    /// Defines the output structure that ``MLXRoutingEngine`` enforces via xgrammar.
    /// Minimal schema (session + confident only) to minimize token generation.
    /// ``MessageRouter`` has fallback defaults for candidates and question fields
    /// when they're absent from the response.
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

    /// Wrap a routing prompt from ``MessageRouter`` for optimal results with the local model.
    ///
    /// Appends the Qwen non-thinking flag. This is retained for compatibility with
    /// callers that still provide the routing request as a single flat string.
    ///
    /// - Parameter routingPrompt: The prompt string built by MessageRouter.
    /// - Returns: A wrapped prompt suitable for the local 0.8B model.
    public static func wrapForLocalModel(_ routingPrompt: String) -> String {
        "\(routingPrompt)\n/no_think"
    }

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
            let recentMessages = session.recentMessages.suffix(2)
            if !recentMessages.isEmpty {
                let recentSummary = recentMessages.map { message in
                    "\(message.role): \"\(message.text.prefix(60))\""
                }
                .joined(separator: " | ")
                sessionLine += ", recent: \(recentSummary)"
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
        _ = routingState
        var lines: [String] = [buildContextPrompt(sessions: sessions)]
        lines.append("")
        lines.append(currentUserMessageHeading)
        lines.append(text)

        return lines.joined(separator: "\n")
    }

    /// Split a flat routing prompt into its reusable context and dynamic utterance.
    public static func splitForLocalModel(_ routingPrompt: String) -> (context: String, currentMessage: String)? {
        let marker = "\n\n\(currentUserMessageHeading)\n"
        guard let range = routingPrompt.range(of: marker) else {
            return nil
        }
        return (
            context: String(routingPrompt[..<range.lowerBound]),
            currentMessage: String(routingPrompt[range.upperBound...])
        )
    }

    /// Build the dynamic user turn appended to a prefetched local-model routing context.
    public static func buildLocalModelCurrentTurn(_ currentMessage: String) -> String {
        "\(currentUserMessageHeading)\n\(currentMessage)\n/no_think"
    }
}

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
    // MARK: - System Prompt

    /// System instruction optimized for the 0.8B model's classification task.
    ///
    /// Key characteristics:
    /// - Concise and direct to maximize accuracy from a small model
    /// - Explicitly states the JSON output format
    /// - Requests non-thinking mode for fast direct answers (Qwen 3 `/no_think`)
    public static let systemInstruction = """
        You are a fast message router. Given active Claude Code sessions and a user message, \
        classify which session should receive the message.

        Rules:
        - If you can confidently determine the target session, set "confident": true and \
        "session" to the session name.
        - If ambiguous, set "confident": false, "session" to "", list likely "candidates", \
        and provide a brief "question" asking the user to clarify.
        - Match based on session context (working directory, recent messages, project type).
        - Consider routing history for conversation continuity.

        Reply ONLY as JSON matching this schema:
        {"session": "<name or empty>", "confident": true/false, "candidates": ["name1", ...], \
        "question": "clarification question"}
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
    /// Prepends the system instruction and appends non-thinking mode flag. The incoming
    /// prompt already contains session context, routing history, and the user message
    /// (built by ``MessageRouter.buildRoutingPrompt()``).
    ///
    /// - Parameter routingPrompt: The prompt string built by MessageRouter.
    /// - Returns: A wrapped prompt suitable for the local 0.8B model.
    public static func wrapForLocalModel(_ routingPrompt: String) -> String {
        // Append /no_think to disable Qwen 3's thinking mode for faster inference
        "\(routingPrompt)\n/no_think"
    }
}

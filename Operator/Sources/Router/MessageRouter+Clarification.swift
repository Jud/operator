import Foundation

// MARK: - Clarification Response Handling

extension MessageRouter {
    /// Ordinal patterns mapped to candidate indices.
    private static let ordinalPatterns: [(patterns: [String], index: Int)] = [
        (["the first one", "first one", "the first", "first", "number one", "number 1", "1"], 0),
        (["the second one", "second one", "the second", "second", "number two", "number 2", "2"], 1),
        (["the third one", "third one", "the third", "third", "number three", "number 3", "3"], 2)
    ]

    /// Patterns indicating the user wants to cancel the clarification.
    private static let cancelPatterns = [
        "cancel", "never mind", "nevermind", "forget it", "forget about it", "nah", "nope"
    ]

    /// Patterns indicating the user wants to send to all candidates.
    ///
    /// Checked via `contains` for multi-word phrases, plus exact match for "all".
    private static let multiPatterns = [
        "both", "all of them", "all of 'em", "all agents", "everyone"
    ]

    /// Patterns indicating the user wants the question repeated (exact match only).
    private static let repeatPatterns: Set<String> = [
        "repeat", "say that again", "what?", "what", "come again", "huh"
    ]

    /// Process a user's response during the CLARIFYING state.
    ///
    /// - Parameters:
    ///   - response: The user's transcribed clarification response.
    ///   - candidates: The candidate session names from the original clarification request.
    ///   - originalText: The original user message that triggered clarification.
    /// - Returns: A `ClarificationResult` indicating how to proceed.
    public func handleClarification(
        response: String,
        candidates: [String],
        originalText: String
    ) async -> ClarificationResult {
        let trimmed = response.trimmingCharacters(in: .whitespacesAndNewlines)
        let lowered = trimmed.lowercased()

        Self.logger.info("Processing clarification response: \"\(trimmed.prefix(80))\"")

        let allSessions = await registry.allSessions()
        let allNames = allSessions.map(\.name)

        for name in allNames {
            let nameLower = name.lowercased()
            if lowered == nameLower || lowered.contains(nameLower) {
                Self.logger.info("Clarification keyword match: '\(name)'")
                return .resolved(session: name)
            }
        }

        if let metaResult = matchMetaIntent(lowered: lowered, candidates: candidates) {
            return metaResult
        }

        Self.logger.info("No meta-intent match; falling back to claude -p for clarification")
        return await claudePipeClarification(
            response: trimmed,
            candidates: candidates,
            originalText: originalText,
            allSessionNames: allNames,
            allSessions: allSessions
        )
    }

    /// Match meta-intents in clarification response.
    func matchMetaIntent(lowered: String, candidates: [String]) -> ClarificationResult? {
        for pattern in Self.cancelPatterns where lowered.contains(pattern) {
            Self.logger.info("Clarification cancelled by user")
            return .cancelled
        }

        if lowered == "all" || Self.multiPatterns.contains(where: { lowered.contains($0) }) {
            Self.logger.info("Clarification: user wants to send to all")
            return .sendToAll
        }

        if Self.repeatPatterns.contains(lowered) {
            Self.logger.info("Clarification: user wants repeat")
            return .repeatQuestion
        }

        if let resolved = matchOrdinal(lowered: lowered, candidates: candidates) {
            Self.logger.info("Clarification ordinal match: '\(resolved)'")
            return .resolved(session: resolved)
        }

        return nil
    }

    /// Match ordinal references against candidates.
    func matchOrdinal(lowered: String, candidates: [String]) -> String? {
        for (patterns, index) in Self.ordinalPatterns {
            for pattern in patterns where lowered.contains(pattern) {
                return candidates.indices.contains(index) ? candidates[index] : nil
            }
        }
        return nil
    }

    /// Fall back to claude -p for interpreting an unrecognized clarification response.
    func claudePipeClarification(
        response: String,
        candidates: [String],
        originalText: String,
        allSessionNames: [String],
        allSessions: [SessionState]
    ) async -> ClarificationResult {
        guard let engine else {
            Self.logger.info("No routing engine; clarification unresolved")
            return .unresolved
        }

        let candidateLines =
            allSessions
            .filter { candidates.contains($0.name) }
            .map { session -> String in
                var line = "- \"\(session.name)\" (\(session.cwd))"
                if !session.context.isEmpty {
                    line += " - \(session.context)"
                }
                return line
            }
            .joined(separator: "\n")

        let prompt = """
            A user was asked which Claude Code session a message was for.
            The original message was: "\(originalText)"
            The candidates are:
            \(candidateLines)

            The user responded: "\(response)"

            Reply ONLY as JSON: {"session": "name"} with the session the user meant.
            Or if still unclear: {"session": null}
            """

        do {
            let json = try await engine.run(prompt: prompt)

            if let sessionName = json["session"] as? String,
                let canonical = Self.canonicalSession(sessionName, in: allSessionNames)
            {
                Self.logger.info("claude -p clarification resolved to '\(canonical)'")
                return .resolved(session: canonical)
            }

            Self.logger.info("claude -p clarification returned no match")
            return .unresolved
        } catch {
            Self.logger.error("claude -p clarification failed: \(error)")
            return .unresolved
        }
    }
}

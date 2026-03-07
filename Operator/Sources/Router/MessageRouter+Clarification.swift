import Foundation
import OSLog

// MARK: - Clarification Response Handling

extension MessageRouter {
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
        let allNames = allSessions.map { $0.name }

        for name in allNames where lowered == name.lowercased() || lowered.contains(name.lowercased()) {
            Self.logger.info("Clarification keyword match: '\(name)'")
            return .resolved(session: name)
        }

        if let metaResult = matchMetaIntent(lowered: lowered, candidates: candidates) {
            return metaResult
        }

        Self.logger.info("No meta-intent match; falling back to claude -p for clarification")
        return await claudePipeClarification(
            response: trimmed,
            candidates: candidates,
            originalText: originalText,
            allSessionNames: allNames
        )
    }

    /// Match meta-intents in clarification response.
    func matchMetaIntent(lowered: String, candidates: [String]) -> ClarificationResult? {
        let cancelPatterns = ["cancel", "never mind", "nevermind", "forget it", "forget about it", "nah", "nope"]
        for pattern in cancelPatterns where lowered.contains(pattern) {
            Self.logger.info("Clarification cancelled by user")
            return .cancelled
        }

        let multiPatterns = ["both", "all of them", "all of 'em", "all agents", "everyone"]
        for pattern in multiPatterns where lowered.contains(pattern) {
            Self.logger.info("Clarification: user wants to send to all")
            return .sendToAll
        }
        if lowered == "all" {
            Self.logger.info("Clarification: user wants to send to all")
            return .sendToAll
        }

        let repeatPatterns = ["repeat", "say that again", "what?", "what", "come again", "huh"]
        for pattern in repeatPatterns where lowered == pattern {
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
        let firstPatterns = ["the first one", "first one", "the first", "first", "number one", "number 1", "1"]
        let secondPatterns = ["the second one", "second one", "the second", "second", "number two", "number 2", "2"]
        let thirdPatterns = ["the third one", "third one", "the third", "third", "number three", "number 3", "3"]

        for pattern in firstPatterns where lowered == pattern || lowered.contains(pattern) {
            return candidates.indices.contains(0) ? candidates[0] : nil
        }

        for pattern in secondPatterns where lowered == pattern || lowered.contains(pattern) {
            return candidates.indices.contains(1) ? candidates[1] : nil
        }

        for pattern in thirdPatterns where lowered == pattern || lowered.contains(pattern) {
            return candidates.indices.contains(2) ? candidates[2] : nil
        }

        return nil
    }

    /// Fall back to claude -p for interpreting an unrecognized clarification response.
    func claudePipeClarification(
        response: String,
        candidates: [String],
        originalText: String,
        allSessionNames: [String]
    ) async -> ClarificationResult {
        let sessions = await registry.allSessions()
        let candidateSessions = sessions.filter { candidates.contains($0.name) }

        var promptLines: [String] = [
            "A user was asked which Claude Code session a message was for.",
            "The original message was: \"\(originalText)\"",
            "The candidates are:"
        ]

        for session in candidateSessions {
            var line = "- \"\(session.name)\" (\(session.cwd))"
            if !session.context.isEmpty {
                line += " - \(session.context)"
            }
            promptLines.append(line)
        }

        promptLines.append("")
        promptLines.append("The user responded: \"\(response)\"")
        promptLines.append("")
        promptLines.append("Reply ONLY as JSON: {\"session\": \"name\"} with the session the user meant.")
        promptLines.append("Or if still unclear: {\"session\": null}")

        let prompt = promptLines.joined(separator: "\n")

        do {
            let json = try await ClaudePipe.run(prompt: prompt)

            if let sessionName = json["session"] as? String,
                allSessionNames.contains(where: { $0.lowercased() == sessionName.lowercased() }) {
                let canonical =
                    allSessionNames.first { $0.lowercased() == sessionName.lowercased() }
                    ?? sessionName
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

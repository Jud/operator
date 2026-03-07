import Foundation

/// The action determined by the InterruptionHandler when the user interrupts
/// an agent's speech by activating push-to-talk.
///
/// Reference: technical-spec.md Component 3, "Interruption Handler"
public enum InterruptionAction: Sendable {
    /// Skip the interrupted message entirely. The user does not want to hear the rest.
    case skip

    /// Replay the interrupted message from the beginning.
    case replay

    /// The user's utterance is a new message directed at a specific agent.
    /// The interrupted message is implicitly skipped.
    case route(target: String, message: String)

    /// The interruption is ambiguous; ask the user a clarifying question.
    case ask(question: String)
}

/// Context about the interrupted speech, provided by the AudioQueue / SpeechManager
/// when push-to-talk interrupts playback.
///
/// This context is passed to the InterruptionHandler so it can build an accurate
/// prompt for claude -p when fast-path matching fails.
public struct InterruptionContext: Sendable {
    /// Name of the agent whose speech was interrupted (e.g., "sudo").
    let agentName: String

    /// The full text of the message that was being spoken.
    let fullText: String

    /// The portion of the message that had been spoken (heard by the user).
    ///
    /// Based on word-level tracking from AVSpeechSynthesizer's
    /// willSpeakRangeOfSpeechString delegate.
    let heardText: String

    /// The portion of the message that had not yet been spoken (unheard).
    let unheardText: String
}

/// Handles user interruptions during agent speech playback.
///
/// When the user presses push-to-talk while an agent's response is being spoken,
/// this handler determines what should happen next using a fast-path keyword check
/// followed by a claude -p fallback for genuinely ambiguous cases.
///
/// Priority chain:
/// 1. Fast-path skip keywords ("skip", "move on", "next") -- no claude -p needed
/// 2. Fast-path replay keywords ("replay", "say that again", "what?") -- no claude -p needed
/// 3. Explicit agent target in user utterance -- skip interrupted, route new message
/// 4. Fallback to claude -p with full interruption context
///
/// Reference: technical-spec.md Component 3, "Interruption Handler";
/// design.md Section 3.1.7
public enum InterruptionHandler {
    private static let logger = Log.logger(for: "InterruptionHandler")

    /// Keywords that indicate the user wants to skip the interrupted message.
    ///
    /// Matched case-insensitively against the full (trimmed) user utterance.
    private static let skipKeywords: Set<String> = [
        "skip",
        "move on",
        "next"
    ]

    /// Keywords that indicate the user wants to replay the interrupted message.
    ///
    /// Matched case-insensitively against the full (trimmed) user utterance.
    private static let replayKeywords: Set<String> = [
        "replay",
        "say that again",
        "what?",
        "what"
    ]

    /// Determine what action to take when the user interrupts agent speech.
    ///
    /// - Parameters:
    ///   - userUtterance: What the user said after pressing push-to-talk.
    ///   - context: Information about the interrupted speech (agent, heard/unheard text).
    ///   - registeredSessionNames: Names of all currently registered sessions.
    /// - Returns: The determined action (skip, replay, route, or ask).
    public static func handle(
        userUtterance: String,
        context: InterruptionContext,
        registeredSessionNames: [String]
    ) async -> InterruptionAction {
        let trimmed = userUtterance.trimmingCharacters(in: .whitespacesAndNewlines)
        let lowered = trimmed.lowercased()

        if skipKeywords.contains(lowered) {
            logger.info("Fast-path skip: user said '\(trimmed)'")
            return .skip
        }

        if replayKeywords.contains(lowered) {
            logger.info("Fast-path replay: user said '\(trimmed)'")
            return .replay
        }

        if let (target, routeAction) = matchAgentTarget(
            utterance: trimmed,
            registeredSessionNames: registeredSessionNames
        ) {
            logger.info("Agent target detected in interruption, routing to \(target)")
            return routeAction
        }

        logger.info("No fast-path match; falling back to claude -p for interruption handling")
        return await claudePipeFallback(userUtterance: trimmed, context: context)
    }

    /// Check if the user's utterance contains an explicit agent target pattern
    /// referencing a registered session name.
    ///
    /// - Parameters:
    ///   - utterance: The user's trimmed utterance.
    ///   - registeredSessionNames: Names of registered sessions.
    /// - Returns: A tuple of (target name, `.route` action) if found, nil otherwise.
    private static func matchAgentTarget(
        utterance: String,
        registeredSessionNames: [String]
    ) -> (target: String, action: InterruptionAction)? {
        guard let result = AgentNameMatcher.match(in: utterance, sessionNames: registeredSessionNames) else {
            return nil
        }
        return (target: result.session, action: .route(target: result.session, message: result.message))
    }

    /// Fall back to claude -p for genuinely ambiguous interruptions.
    private static func claudePipeFallback(
        userUtterance: String,
        context: InterruptionContext
    ) async -> InterruptionAction {
        let prompt = buildInterruptionPrompt(userUtterance: userUtterance, context: context)

        do {
            let json = try await ClaudePipe.run(prompt: prompt)

            guard let action = json["action"] as? String else {
                logger.warning("claude -p response missing 'action' field")
                return .ask(question: "Did you want to hear the rest of \(context.agentName)'s message?")
            }

            switch action {
            case "skip":
                let reason = json["reason"] as? String ?? "no reason given"
                logger.info("claude -p decided: skip (\(reason))")
                return .skip

            case "replay":
                let reason = json["reason"] as? String ?? "no reason given"
                logger.info("claude -p decided: replay (\(reason))")
                return .replay

            case "ask":
                let question =
                    json["question"] as? String
                    ?? "Did you want to hear the rest of \(context.agentName)'s message?"
                logger.info("claude -p decided: ask (\(question))")
                return .ask(question: question)

            default:
                logger.warning("claude -p returned unknown action: \(action)")
                return .ask(question: "Did you want to hear the rest of \(context.agentName)'s message?")
            }
        } catch let error as ClaudePipeError {
            logger.error("claude -p failed for interruption handling: \(error.description)")
            return .ask(question: "Did you want to hear the rest of \(context.agentName)'s message?")
        } catch {
            logger.error("Unexpected error in interruption handling: \(error)")
            return .ask(question: "Did you want to hear the rest of \(context.agentName)'s message?")
        }
    }

    /// Build the claude -p prompt for interruption handling.
    private static func buildInterruptionPrompt(
        userUtterance: String,
        context: InterruptionContext
    ) -> String {
        """
        An agent was speaking and the user interrupted.
        Agent ("\(context.agentName)") was saying: "\(context.fullText)"
        User heard up to: "\(context.heardText)"
        User then said: "\(userUtterance)"

        What should happen? Reply ONLY as JSON:
        {"action": "skip", "reason": "User heard the key content and explicitly objected"}
        OR {"action": "replay", "reason": "User barely heard anything"}
        OR {"action": "ask", "question": "Did you want to hear the rest of \(context.agentName)'s message?"}
        """
    }
}

import Foundation

// MARK: - Heuristic Routing

extension MessageRouter {
    struct HeuristicCandidate {
        let session: String
        let score: Int
        let directHits: Int
    }

    static let heuristicStopWords: Set<String> = [
        "a", "an", "and", "the", "to", "for", "of", "on", "in", "at", "with", "from", "new",
        "fix", "add", "update", "make", "check", "look", "into", "that", "this", "it", "my",
        "your", "their", "our", "please", "need", "still", "just", "agent", "session"
    ]

    static let heuristicConceptGroups: [[String]] = [
        [
            "react", "tsx", "typescript", "frontend", "client", "ui", "css", "layout",
            "dashboard", "shell", "button", "spacing", "skeleton", "spinner", "loading",
            "component"
        ],
        [
            "api", "backend", "server", "endpoint", "rest", "route", "service", "python",
            "py", "sql", "postgres", "database", "db", "query", "middleware", "auth",
            "billing", "refund", "profile", "sync", "requirements"
        ],
        [
            "infra", "terraform", "tf", "deploy", "deployment", "staging", "prod",
            "production", "cluster", "eks", "vpc", "subnet", "network", "iam", "policy",
            "s3", "logging", "rollout"
        ]
    ]

    static let heuristicConceptMap: [String: Set<String>] = {
        var map: [String: Set<String>] = [:]
        for group in heuristicConceptGroups {
            let expanded = Set(group)
            for token in group {
                map[token] = expanded
            }
        }
        return map
    }()
}

// MARK: - Heuristic Scoring

extension MessageRouter {
    static func scoreHeuristicRoute(
        messageTokens: Set<String>,
        session: SessionState
    ) -> HeuristicCandidate {
        let nameTokens = heuristicExpandedTokens(
            from: session.name,
            dropStopWords: false
        )
        let cwdTokens = heuristicExpandedTokens(from: session.cwd)
        let contextTokens = heuristicExpandedTokens(from: session.context)
        let recentTokens = heuristicExpandedTokens(
            from: session.recentMessages.map(\.text).joined(separator: " ")
        )

        let tokenWeights = [
            (nameTokens, 8),
            (cwdTokens, 6),
            (contextTokens, 4),
            (recentTokens, 3)
        ]

        var score = 0
        var directHits = 0

        for token in messageTokens {
            var matched = false
            for (sourceTokens, weight) in tokenWeights where sourceTokens.contains(token) {
                score += weight
                matched = true
            }

            if matched {
                directHits += 1
            }
        }

        return HeuristicCandidate(session: session.name, score: score, directHits: directHits)
    }

    static func heuristicMessageTokens(from text: String) -> Set<String> {
        heuristicExpandedTokens(from: text)
    }

    static func heuristicExpandedTokens(
        from text: String,
        dropStopWords: Bool = true
    ) -> Set<String> {
        let baseTokens = heuristicBaseTokens(from: text)
        var expanded: Set<String> = []

        for token in baseTokens {
            if dropStopWords && heuristicStopWords.contains(token) {
                continue
            }
            expanded.insert(token)
            if let conceptTokens = heuristicConceptMap[token] {
                expanded.formUnion(conceptTokens)
            }
        }

        return expanded
    }

    static func heuristicBaseTokens(from text: String) -> Set<String> {
        let separators = CharacterSet.alphanumerics.inverted
        let lowered =
            text
            .replacingOccurrences(
                of: "([a-z0-9])([A-Z])",
                with: "$1 $2",
                options: .regularExpression
            )
            .lowercased()

        let rawTokens =
            lowered
            .components(separatedBy: separators)
            .filter { !$0.isEmpty }

        var tokens = Set(rawTokens)
        for token in rawTokens where token.count > 3 && token.hasSuffix("s") {
            tokens.insert(String(token.dropLast()))
        }
        return tokens
    }

    func heuristicRoute(
        text: String,
        sessions: [SessionState]
    ) -> String? {
        let messageTokens = Self.heuristicMessageTokens(from: text)
        guard !messageTokens.isEmpty else {
            return nil
        }

        let ranked =
            sessions
            .map { session in
                Self.scoreHeuristicRoute(messageTokens: messageTokens, session: session)
            }
            .filter { $0.score > 0 }
            .sorted {
                if $0.score == $1.score {
                    return $0.directHits > $1.directHits
                }
                return $0.score > $1.score
            }

        guard let best = ranked.first else {
            return nil
        }

        let runnerUpScore = ranked.dropFirst().first?.score ?? 0
        let strongDirectMatch = best.directHits >= 2 && best.score >= 10
        let strongMargin = best.score >= 12 && best.score >= runnerUpScore + 4
        guard strongDirectMatch || strongMargin else {
            return nil
        }

        return best.session
    }
}

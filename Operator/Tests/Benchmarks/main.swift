import AVFoundation
import OperatorCore

private enum BenchmarkTarget: String, CaseIterable {
    case routing
    case routingAccuracy = "routing-accuracy"
    case routingLatency = "routing-latency"
}

private let benchmarkRunnerEnvKey = "OPERATOR_BENCHMARK_RUNNER"

private func printBenchmarkUsage(listOnly: Bool = false) {
    print("Usage: ./scripts/run-benchmarks.sh [--debug|--release] [--no-build] [target...]")
    print("")
    print("Targets:")
    print("  routing           Run routing accuracy and latency benchmarks (heuristic only)")
    print("  routing-accuracy  Run only routing accuracy benchmark")
    print("  routing-latency   Run only routing latency benchmark")
    if listOnly {
        return
    }
    print("")
    print("Examples:")
    print("  ./scripts/run-benchmarks.sh routing")
    print("  ./scripts/run-benchmarks.sh --no-build routing-accuracy")
}

private func makeBenchmarkVoice() -> VoiceDescriptor? {
    guard let appleVoice = AVSpeechSynthesisVoice(language: "en-US") else {
        return nil
    }
    return .system(appleVoice)
}

private func makeRoutingBenchmarkSessions(voice: VoiceDescriptor) -> [SessionState] {
    [
        SessionState(
            name: "ui-shell",
            tty: "/dev/benchmark1",
            cwd: "/Users/jud/work/operator-ui",
            context:
                "Branch feat/settings-loading. Editing web/src/routes/settings.tsx, web/src/components/LoadingCard.tsx, and web/src/styles/dashboard.css for React loading states, layout polish, and responsive CSS.",
            recentMessages: [
                SessionMessage(
                    role: "user",
                    text:
                        "Fix the janky loading state on the settings dashboard and make the cards stop shifting on mobile."
                ),
                SessionMessage(
                    role: "assistant",
                    text:
                        "I updated LoadingCard.tsx, SettingsRoute.tsx, and dashboard.css; next I am wiring the pending query state and skeleton layout."
                )
            ],
            status: .idle,
            lastActivity: Date(),
            voice: voice,
            pitchMultiplier: 1.0
        ),
        SessionState(
            name: "profile-api",
            tty: "/dev/benchmark2",
            cwd: "/Users/jud/work/operator-api",
            context:
                "Branch feat/profile-endpoints. Editing api/routes/profile.py, services/user_profile.py, and db/queries/profile.sql for REST endpoints, auth middleware, and Postgres query performance.",
            recentMessages: [
                SessionMessage(
                    role: "user",
                    text:
                        "The user profile endpoint is timing out under load and we still need a new REST route for profile preferences."
                ),
                SessionMessage(
                    role: "assistant",
                    text:
                        "I traced the slowdown to profile.sql and the connection pool, and I am adding the preferences endpoint in profile.py."
                )
            ],
            status: .idle,
            lastActivity: Date(),
            voice: voice,
            pitchMultiplier: 1.0
        ),
        SessionState(
            name: "staging-infra",
            tty: "/dev/benchmark3",
            cwd: "/Users/jud/work/operator-infra",
            context:
                "Branch chore/staging-network. Editing infra/envs/staging/main.tf, modules/vpc, and modules/iam_policy for Terraform apply failures, private subnet layout, IAM policies, and S3 logging.",
            recentMessages: [
                SessionMessage(
                    role: "user",
                    text:
                        "The staging VPC rollout failed after the Terraform change and logging still needs the new S3 bucket policy."
                ),
                SessionMessage(
                    role: "assistant",
                    text:
                        "I am updating the private subnet module, the IAM policy document, and the S3 logging resources in staging Terraform."
                )
            ],
            status: .idle,
            lastActivity: Date(),
            voice: voice,
            pitchMultiplier: 1.0
        )
    ]
}

private func makeBenchmarkRouter(
    sessions: [SessionState]
) async -> MessageRouter {
    let registry = SessionRegistry(voiceManager: VoiceManager())
    for session in sessions {
        await registry.register(name: session.name, tty: session.tty, cwd: session.cwd, context: session.context)
        await registry.updateContext(
            tty: session.tty,
            summary: session.context,
            recentMessages: session.recentMessages
        )
    }
    return MessageRouter(registry: registry)
}

// MARK: - Accuracy Test Types

private struct AccuracyTestCase {
    let message: String
    let expected: String
}

private struct AccuracyScenario {
    let name: String
    let sessions: [SessionState]
    let cases: [AccuracyTestCase]
}

// MARK: - Routing Benchmark

private func benchmarkRouting(runLatency: Bool, runAccuracy: Bool) async {
    guard runLatency || runAccuracy else { return }

    print("\n=== ROUTING BENCHMARK (Heuristic Only) ===\n")

    guard let benchmarkVoice = makeBenchmarkVoice() else {
        print("FAIL: Could not create AVSpeechSynthesisVoice")
        return
    }
    let benchmarkSessions = makeRoutingBenchmarkSessions(voice: benchmarkVoice)

    if runLatency {
        await runLatencyBenchmark(sessions: benchmarkSessions)
    }

    if runAccuracy {
        await runAccuracyBenchmark(sessions: benchmarkSessions)
    }
}

private func runLatencyBenchmark(sessions: [SessionState]) async {
    let router = await makeBenchmarkRouter(sessions: sessions)
    let testMessages = [
        "fix the database migration",
        "update the terraform module for the new VPC",
        "add dark mode toggle to the settings page",
        "optimize the SQL query for user lookups",
        "deploy the staging environment",
        "refactor the authentication middleware",
        "add unit tests for the payment module",
        "fix the CSS grid layout on mobile"
    ]

    var latencies: [Double] = []

    print("--- Routing Latency (Heuristic) ---")
    for message in testMessages {
        let start = CFAbsoluteTimeGetCurrent()
        _ = await router.route(text: message, routingState: RoutingState())
        let durationMs = (CFAbsoluteTimeGetCurrent() - start) * 1000
        latencies.append(durationMs)
        print("  \(String(format: "%5.2f", durationMs))ms | \"\(message)\"")
    }

    let sorted = latencies.sorted()
    let avg = latencies.reduce(0, +) / Double(latencies.count)
    let p50 = sorted[sorted.count / 2]
    let p95 = sorted[Int(Double(sorted.count - 1) * 0.95)]
    let maxMs = sorted.last ?? 0

    print("")
    print(
        "Latency:  avg=\(String(format: "%.2f", avg))ms  p50=\(String(format: "%.2f", p50))ms  p95=\(String(format: "%.2f", p95))ms  max=\(String(format: "%.2f", maxMs))ms"
    )
}

private func runAccuracyBenchmark(sessions: [SessionState]) async {
    let scenarios: [AccuracyScenario] = [
        AccuracyScenario(
            name: "operator stack",
            sessions: sessions,
            cases: [
                AccuracyTestCase(message: "add a loading spinner to the React dashboard", expected: "ui-shell"),
                AccuracyTestCase(message: "fix the CSS grid layout", expected: "ui-shell"),
                AccuracyTestCase(message: "add a new REST endpoint for user profiles", expected: "profile-api"),
                AccuracyTestCase(message: "fix the database connection pool", expected: "profile-api"),
                AccuracyTestCase(message: "optimize the SQL query", expected: "profile-api"),
                AccuracyTestCase(message: "update the terraform module for the new VPC", expected: "staging-infra"),
                AccuracyTestCase(message: "add a new S3 bucket for logs", expected: "staging-infra"),
                AccuracyTestCase(message: "fix the AWS IAM permissions", expected: "staging-infra"),
                AccuracyTestCase(message: "add TypeScript types for the API response", expected: "ui-shell"),
                AccuracyTestCase(message: "update the Python requirements.txt", expected: "profile-api")
            ]
        )
    ]

    print("\n--- Routing Accuracy (Heuristic) ---\n")
    var totalCorrect = 0
    var totalCases = 0

    for scenario in scenarios {
        let router = await makeBenchmarkRouter(sessions: scenario.sessions)

        print("Scenario: \(scenario.name)")
        var correct = 0

        for testCase in scenario.cases {
            let result = await router.route(text: testCase.message, routingState: RoutingState())
            let routed: String
            switch result {
            case .route(let session, _):
                routed = session
            case .clarify:
                routed = "<clarify>"
            case .notConfident:
                routed = "<not-confident>"
            default:
                routed = "<other>"
            }

            let ok = routed == testCase.expected
            if ok { correct += 1 }
            let marker = ok ? "OK" : "MISS"
            print("  [\(marker)] \"\(testCase.message)\" -> \(routed) (expected: \(testCase.expected))")
        }

        let accuracy = Double(correct) / Double(scenario.cases.count) * 100
        print("  Score: \(correct)/\(scenario.cases.count) (\(String(format: "%.0f", accuracy))%)\n")
        totalCorrect += correct
        totalCases += scenario.cases.count
    }

    let overall = Double(totalCorrect) / Double(totalCases) * 100
    print("Overall: \(totalCorrect)/\(totalCases) (\(String(format: "%.0f", overall))%)")
}

// MARK: - Main

@main
enum BenchmarkRunner {
    static func main() async {
        guard ProcessInfo.processInfo.environment[benchmarkRunnerEnvKey] != nil else {
            print("Run benchmarks via: ./scripts/run-benchmarks.sh [target...]")
            return
        }

        let args = Array(CommandLine.arguments.dropFirst())

        if args.isEmpty || args.contains("list") || args.contains("help") {
            printBenchmarkUsage(listOnly: args.contains("list"))
            return
        }

        let targets: Set<BenchmarkTarget> = Set(args.compactMap { BenchmarkTarget(rawValue: $0) })

        if targets.isEmpty {
            print(
                "No recognized targets. Available: \(BenchmarkTarget.allCases.map(\.rawValue).joined(separator: ", "))"
            )
            return
        }

        let includesRouting = targets.contains(.routing)
        if includesRouting || targets.contains(.routingLatency) || targets.contains(.routingAccuracy) {
            await benchmarkRouting(
                runLatency: includesRouting || targets.contains(.routingLatency),
                runAccuracy: includesRouting || targets.contains(.routingAccuracy)
            )
        }

        print("\nDone.")
    }
}

import AVFoundation
import Darwin
import Foundation
import OperatorCore

private enum BenchmarkTarget: String, CaseIterable {
    case routing
    case routingSingle = "routing-single"
    case routingLatency = "routing-latency"
    case routingAccuracy = "routing-accuracy"
    case tts
    case ttsTTFA = "tts-ttfa"
    case stt
    case sttLatency = "stt-latency"
    case sttLong = "stt-long"
    case sttStreaming = "stt-streaming"
    case sttSoak = "stt-soak"
    case memory
}

private let benchmarkHelpTargets = BenchmarkTarget.allCases.map(\.rawValue).joined(separator: ", ")
private let benchmarkRunnerEnvKey = "OPERATOR_BENCHMARK_RUNNER"

private func printBenchmarkUsage(listOnly: Bool = false) {
    print("Usage: ./scripts/run-benchmarks.sh [--debug|--release] [--no-build] [target...]")
    print("No .app bundle is required.")
    print("")
    print("Targets:")
    print("  routing           Run routing latency and routing accuracy benchmarks")
    print("  routing-single    Run one profiled routing inference with phase timings")
    print("  routing-latency   Run only routing latency benchmark")
    print("  routing-accuracy  Run only routing accuracy benchmark")
    print("  tts               Run TTS time-to-first-audio benchmark")
    print("  tts-ttfa          Run only TTS time-to-first-audio benchmark")
    print("  stt               Run STT latency and long-utterance benchmarks")
    print("  stt-latency       Run only STT 5s transcription benchmark")
    print("  stt-long          Run only STT long-utterance benchmark")
    print("  stt-streaming     Run STT streaming/finalize benchmark with synthesized speech")
    print("  stt-soak          Run a long paced STT soak benchmark (set OPERATOR_STT_SOAK_MINUTES)")
    print("  memory            Run routing-model memory benchmark")
    if listOnly {
        return
    }
    print("")
    print("Examples:")
    print("  ./scripts/run-benchmarks.sh routing")
    print("  ./scripts/run-benchmarks.sh routing-single")
    print("  ./scripts/run-benchmarks.sh --no-build routing-latency")
    print("  ./scripts/run-benchmarks.sh stt-long")
    print("  ./scripts/run-benchmarks.sh stt-streaming")
    print("  OPERATOR_STT_SOAK_MINUTES=5 ./scripts/run-benchmarks.sh stt-soak")
    print("  ./scripts/run-benchmarks.sh --release memory")
}

private func startModelProgressObserver(
    modelManager: ModelManager,
    type: ModelManager.ModelType,
    label: String
) async -> Task<Void, Never> {
    let stream = await modelManager.stateChanged

    return Task {
        var lastDownloadBucket: Int?
        var lastStage: String?

        for await (changedType, state) in stream {
            if Task.isCancelled {
                return
            }
            guard changedType == type else {
                continue
            }

            switch state {
            case .notDownloaded:
                continue

            case .downloading(let progress):
                let percent = max(0, min(100, Int(progress * 100)))
                let bucket = (percent / 10) * 10
                guard lastDownloadBucket != bucket else {
                    continue
                }
                lastDownloadBucket = bucket
                print("  \(label): download \(bucket)%")

            case .ready:
                guard lastStage != "ready" else {
                    continue
                }
                lastStage = "ready"
                print("  \(label): download complete")

            case .loading:
                guard lastStage != "loading" else {
                    continue
                }
                lastStage = "loading"
                print("  \(label): loading model")

            case .loaded:
                guard lastStage != "loaded" else {
                    continue
                }
                lastStage = "loaded"
                print("  \(label): model loaded")

            case .unloaded:
                guard lastStage != "unloaded" else {
                    continue
                }
                lastStage = "unloaded"
                print("  \(label): model unloaded")

            case .error(let message):
                print("  \(label): error \(message)")
            }
        }
    }
}

private func makeBenchmarkVoice() -> VoiceDescriptor? {
    guard let appleVoice = AVSpeechSynthesisVoice(language: "en-US") else {
        return nil
    }
    return VoiceDescriptor(appleVoice: appleVoice, qwenSpeakerID: "benchmark")
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

private func makeRoutingBenchmarkPrompt(
    text: String,
    sessions: [SessionState]
) -> String {
    RoutingPrompt.buildPrompt(
        text: text,
        sessions: sessions,
        routingState: RoutingState()
    )
}

private func makeBenchmarkRouter(
    sessions: [SessionState],
    engine: any RoutingEngine
) async -> MessageRouter {
    let voiceManager = VoiceManager()
    let registry = SessionRegistry(voiceManager: voiceManager)

    for session in sessions {
        _ = await registry.register(
            name: session.name,
            tty: session.tty,
            cwd: session.cwd,
            context: session.context
        )
        await registry.updateContext(
            tty: session.tty,
            summary: session.context,
            recentMessages: session.recentMessages
        )
    }

    return MessageRouter(registry: registry, engine: engine)
}

private func wordCount(in text: String) -> Int {
    text.split(whereSeparator: \.isWhitespace).count
}

@MainActor
func runBenchmarks() async {
    guard ProcessInfo.processInfo.environment[benchmarkRunnerEnvKey] == "1" else {
        fputs("Run benchmarks via ./scripts/run-benchmarks.sh. No .app bundle is required.\n", stderr)
        exit(2)
    }

    let args = Set(
        CommandLine.arguments
            .dropFirst()
            .map { $0.lowercased().trimmingCharacters(in: .whitespacesAndNewlines) }
    )

    if args.contains("help") || args.contains("--help") || args.contains("-h") {
        printBenchmarkUsage()
        return
    }
    if args.contains("list") {
        printBenchmarkUsage(listOnly: true)
        return
    }

    let knownArgs = Set(BenchmarkTarget.allCases.map(\.rawValue)).union(["all"])
    let unknownArgs = args.subtracting(knownArgs)
    guard unknownArgs.isEmpty else {
        print("Unknown benchmark target(s): \(unknownArgs.sorted().joined(separator: ", "))")
        print("")
        printBenchmarkUsage()
        return
    }

    let runAll = args.isEmpty || args.contains("all")
    let includes = { (target: BenchmarkTarget) in
        runAll || args.contains(target.rawValue)
    }

    if includes(.routingSingle) {
        await benchmarkRoutingSingle()
    }
    if includes(.routing) || includes(.routingLatency) || includes(.routingAccuracy) {
        await benchmarkRouting(
            runLatency: includes(.routing) || includes(.routingLatency),
            runAccuracy: includes(.routing) || includes(.routingAccuracy)
        )
    }
    if includes(.tts) || includes(.ttsTTFA) {
        await benchmarkTTS()
    }
    if includes(.stt) || includes(.sttLatency) || includes(.sttLong) || includes(.sttStreaming) || includes(.sttSoak) {
        await benchmarkSTT(
            runLatency: includes(.stt) || includes(.sttLatency),
            runLong: includes(.stt) || includes(.sttLong),
            runStreaming: includes(.stt) || includes(.sttStreaming),
            runSoak: includes(.sttSoak)
        )
    }
    if includes(.memory) {
        await benchmarkMemory()
    }
}

// MARK: - Routing Benchmarks (AC-007.1, AC-007.5)

@MainActor
func benchmarkRoutingSingle() async {
    print("\n=== ROUTING SINGLE INFERENCE ===\n")

    let modelManager = ModelManager()
    let engine = MLXRoutingEngine(modelManager: modelManager)
    guard let benchmarkVoice = makeBenchmarkVoice() else {
        print("FAIL: Could not create AVSpeechSynthesisVoice")
        return
    }

    let sessions = makeRoutingBenchmarkSessions(voice: benchmarkVoice)
    let coldMessage = "add a new REST endpoint for user profiles"
    let coldPrompt = makeRoutingBenchmarkPrompt(text: coldMessage, sessions: sessions)

    let routingProgress = await startModelProgressObserver(
        modelManager: modelManager,
        type: .routing,
        label: "Routing"
    )

    do {
        print("Cold message: \"\(coldMessage)\"")
        print("Cold prompt: \(coldPrompt.count) chars, \(wordCount(in: coldPrompt)) words")

        let result = try await engine.runProfiled(prompt: coldPrompt, timeout: 120)
        routingProgress.cancel()

        let metrics = result.metrics
        let session = result.json["session"] as? String ?? ""
        let confident = result.json["confident"] as? Bool ?? false

        print("Cold output: session=\(session.isEmpty ? "<empty>" : session) confident=\(confident)")
        print("Cold raw JSON: \(result.rawOutput)")
        print("")
        print("Cold phases:")
        print("  model load:      \(String(format: "%.1f", metrics.modelLoadMs))ms")
        print("  grammar setup:   \(String(format: "%.1f", metrics.grammarSetupMs))ms")
        print("  prefix cache:    \(String(format: "%.1f", metrics.promptPrefixCacheMs))ms")
        print("  inference RTT:   \(String(format: "%.1f", metrics.inferenceRoundTripMs))ms")
        print("  task grp ovhd:   \(String(format: "%.1f", metrics.taskGroupOverheadMs))ms")
        print("  perform RTT:     \(String(format: "%.1f", metrics.containerPerformRoundTripMs))ms")
        print("  perform queue:   \(String(format: "%.1f", metrics.containerPerformSchedulingMs))ms")
        print("  closure total:   \(String(format: "%.1f", metrics.containerClosureMs))ms")
        print("  inf. overhead:   \(String(format: "%.1f", metrics.inferenceOverheadMs))ms")
        print("  prompt prepare:  \(String(format: "%.1f", metrics.promptPreparationMs))ms")
        print("  iterator init:   \(String(format: "%.1f", metrics.tokenIteratorInitMs))ms")
        print("  token generate:  \(String(format: "%.1f", metrics.generationMs))ms")
        print("  json parse:      \(String(format: "%.1f", metrics.jsonParseMs))ms")
        print("  total:           \(String(format: "%.1f", metrics.totalMs))ms")
        print("")
        print("Cold flags:")
        print("  cached grammar:  \(metrics.usedCachedGrammar ? "yes" : "no")")
        print("  prefix cache:    \(metrics.usedPromptPrefixCache ? "yes" : "no")")
        print("  prefix cache hit:\(metrics.promptPrefixCacheHit ? "yes" : "no")")
        print("  model preloaded: \(metrics.modelWasLoadedBeforeRun ? "yes" : "no")")
        print("  cached context:  \(metrics.cachedContextCharacters) chars, \(metrics.cachedContextWords) words")
        print("  runtime prompt:  \(metrics.runtimePromptCharacters) chars, \(metrics.runtimePromptWords) words")

        let warmMessage = "fix the CSS grid layout"
        let warmPrompt = makeRoutingBenchmarkPrompt(text: warmMessage, sessions: sessions)
        let warmResult = try await engine.runProfiled(prompt: warmPrompt, timeout: 30)
        let warmMetrics = warmResult.metrics
        let warmSession = warmResult.json["session"] as? String ?? ""
        let warmConfident = warmResult.json["confident"] as? Bool ?? false

        print("")
        print("Warm message: \"\(warmMessage)\"")
        print("Warm prompt: \(warmPrompt.count) chars, \(wordCount(in: warmPrompt)) words")
        print("Warm output: session=\(warmSession.isEmpty ? "<empty>" : warmSession) confident=\(warmConfident)")
        print("Warm raw JSON: \(warmResult.rawOutput)")
        print("")
        print("Warm phases:")
        print("  model load:      \(String(format: "%.1f", warmMetrics.modelLoadMs))ms")
        print("  grammar setup:   \(String(format: "%.1f", warmMetrics.grammarSetupMs))ms")
        print("  prefix cache:    \(String(format: "%.1f", warmMetrics.promptPrefixCacheMs))ms")
        print("  inference RTT:   \(String(format: "%.1f", warmMetrics.inferenceRoundTripMs))ms")
        print("  task grp ovhd:   \(String(format: "%.1f", warmMetrics.taskGroupOverheadMs))ms")
        print("  perform RTT:     \(String(format: "%.1f", warmMetrics.containerPerformRoundTripMs))ms")
        print("  perform queue:   \(String(format: "%.1f", warmMetrics.containerPerformSchedulingMs))ms")
        print("  closure total:   \(String(format: "%.1f", warmMetrics.containerClosureMs))ms")
        print("  inf. overhead:   \(String(format: "%.1f", warmMetrics.inferenceOverheadMs))ms")
        print("  prompt prepare:  \(String(format: "%.1f", warmMetrics.promptPreparationMs))ms")
        print("  iterator init:   \(String(format: "%.1f", warmMetrics.tokenIteratorInitMs))ms")
        print("  token generate:  \(String(format: "%.1f", warmMetrics.generationMs))ms")
        print("  json parse:      \(String(format: "%.1f", warmMetrics.jsonParseMs))ms")
        print("  total:           \(String(format: "%.1f", warmMetrics.totalMs))ms")
        print("")
        print("Warm flags:")
        print("  cached grammar:  \(warmMetrics.usedCachedGrammar ? "yes" : "no")")
        print("  prefix cache:    \(warmMetrics.usedPromptPrefixCache ? "yes" : "no")")
        print("  prefix cache hit:\(warmMetrics.promptPrefixCacheHit ? "yes" : "no")")
        print("  model preloaded: \(warmMetrics.modelWasLoadedBeforeRun ? "yes" : "no")")
        print(
            "  cached context:  \(warmMetrics.cachedContextCharacters) chars, \(warmMetrics.cachedContextWords) words"
        )
        print(
            "  runtime prompt:  \(warmMetrics.runtimePromptCharacters) chars, \(warmMetrics.runtimePromptWords) words"
        )
    } catch {
        routingProgress.cancel()
        print("FAIL: \(error)")
    }
}

@MainActor
func benchmarkRouting(runLatency: Bool, runAccuracy: Bool) async {
    guard runLatency || runAccuracy else {
        return
    }

    print("\n=== ROUTING BENCHMARK ===\n")

    let modelManager = ModelManager()
    let engine = MLXRoutingEngine(modelManager: modelManager)
    guard let benchmarkVoice = makeBenchmarkVoice() else {
        print("FAIL: Could not create AVSpeechSynthesisVoice")
        return
    }
    let benchmarkSessions = makeRoutingBenchmarkSessions(voice: benchmarkVoice)

    let prompt = makeRoutingBenchmarkPrompt(
        text: "add a loading spinner to the dashboard",
        sessions: benchmarkSessions
    )

    // Cold start (includes model download + load)
    print("Loading routing model (first run downloads ~622MB)...")
    let routingProgress = await startModelProgressObserver(
        modelManager: modelManager,
        type: .routing,
        label: "Routing"
    )
    let coldStart = CFAbsoluteTimeGetCurrent()
    do {
        _ = try await engine.run(prompt: prompt, timeout: 120)
    } catch {
        routingProgress.cancel()
        print("FAIL: Cold start failed: \(error)")
        return
    }
    routingProgress.cancel()
    let coldMs = (CFAbsoluteTimeGetCurrent() - coldStart) * 1000
    print("Cold start (download+load+inference): \(String(format: "%.0f", coldMs))ms")

    if runLatency {
        let router = await makeBenchmarkRouter(
            sessions: benchmarkSessions,
            engine: engine
        )
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

        print("\n--- Routing Latency ---")
        for message in testMessages {
            let start = CFAbsoluteTimeGetCurrent()
            _ = await router.route(text: message, routingState: RoutingState())
            let durationMs = (CFAbsoluteTimeGetCurrent() - start) * 1000
            latencies.append(durationMs)
            print("  \(String(format: "%5.0f", durationMs))ms | \"\(message)\"")
        }

        let sorted = latencies.sorted()
        let avg = latencies.reduce(0, +) / Double(latencies.count)
        let p50 = sorted[sorted.count / 2]
        let p95 = sorted[Int(Double(sorted.count - 1) * 0.95)]
        let maxMs = sorted.last ?? 0

        print("")
        let latencyLine =
            "Latency:  avg=\(String(format: "%.0f", avg))ms"
            + "  p50=\(String(format: "%.0f", p50))ms"
            + "  p95=\(String(format: "%.0f", p95))ms"
            + "  max=\(String(format: "%.0f", maxMs))ms"
        print(latencyLine)
        print("AC-007.1 (latency <300ms):  \(p95 < 300 ? "PASS" : "FAIL") (p95=\(String(format: "%.0f", p95))ms)")
    }

    if runAccuracy {
        struct SessionCase {
            let name: String
            let cwd: String
            let context: String
            let recentMessages: [SessionMessage]
        }

        struct TestCase {
            let message: String
            let expected: String
        }

        struct Scenario {
            let name: String
            let sessions: [SessionCase]
            let cases: [TestCase]
        }

        let scenarios: [Scenario] = [
            Scenario(
                name: "operator stack",
                sessions: [
                    SessionCase(
                        name: "ui-shell",
                        cwd: "/Users/jud/work/operator-ui",
                        context:
                            "Branch feat/settings-loading. Editing web/src/routes/settings.tsx, web/src/components/LoadingCard.tsx, and web/src/styles/dashboard.css for React loading states, responsive layout bugs, and dashboard polish.",
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
                        ]
                    ),
                    SessionCase(
                        name: "profile-api",
                        cwd: "/Users/jud/work/operator-api",
                        context:
                            "Branch feat/profile-endpoints. Editing api/routes/profile.py, services/user_profile.py, and db/queries/profile.sql for REST endpoints, auth middleware, and Postgres performance.",
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
                        ]
                    ),
                    SessionCase(
                        name: "staging-infra",
                        cwd: "/Users/jud/work/operator-infra",
                        context:
                            "Branch chore/staging-network. Editing infra/envs/staging/main.tf, modules/vpc, and modules/iam_policy for Terraform apply failures, private subnet layout, IAM policy changes, and S3 logging.",
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
                        ]
                    )
                ],
                cases: [
                    TestCase(message: "add a loading spinner to the React dashboard", expected: "ui-shell"),
                    TestCase(message: "fix the CSS grid layout", expected: "ui-shell"),
                    TestCase(message: "add a new REST endpoint for user profiles", expected: "profile-api"),
                    TestCase(message: "fix the database connection pool", expected: "profile-api"),
                    TestCase(message: "optimize the SQL query", expected: "profile-api"),
                    TestCase(message: "update the terraform module for the new VPC", expected: "staging-infra"),
                    TestCase(message: "add a new S3 bucket for logs", expected: "staging-infra"),
                    TestCase(message: "fix the AWS IAM permissions", expected: "staging-infra"),
                    TestCase(message: "add TypeScript types for the API response", expected: "ui-shell"),
                    TestCase(message: "update the Python requirements.txt", expected: "profile-api")
                ]
            ),
            Scenario(
                name: "agent handles",
                sessions: [
                    SessionCase(
                        name: "jules",
                        cwd: "/Users/jud/sessions/jules",
                        context:
                            "Branch feat/onboarding-shell. Editing app/routes/onboarding.tsx, app/components/StepCard.tsx, and app/styles/forms.css for React onboarding UI, form spacing, and loading transitions.",
                        recentMessages: [
                            SessionMessage(
                                role: "user",
                                text:
                                    "The onboarding cards are misaligned on smaller screens and the loading placeholders flash."
                            ),
                            SessionMessage(
                                role: "assistant",
                                text:
                                    "I refined StepCard.tsx, onboarding.tsx, and forms.css to stabilize the skeletons and spacing."
                            )
                        ]
                    ),
                    SessionCase(
                        name: "marco",
                        cwd: "/Users/jud/sessions/marco",
                        context:
                            "Branch feat/billing-api. Editing api/routes/billing.py, services/refunds.py, and db/queries/invoices.sql for Python endpoints, refund status APIs, and invoice query tuning.",
                        recentMessages: [
                            SessionMessage(
                                role: "user",
                                text: "Refund status still needs an endpoint and invoice totals are off in Postgres."
                            ),
                            SessionMessage(
                                role: "assistant",
                                text:
                                    "I am updating billing.py, refunds.py, and invoices.sql to fix the endpoint and invoice aggregation."
                            )
                        ]
                    ),
                    SessionCase(
                        name: "tess",
                        cwd: "/Users/jud/sessions/tess",
                        context:
                            "Branch chore/cluster-rollout. Editing infra/prod/cluster.tf, modules/network, and modules/iam_role for EKS rollout issues, private subnets, IAM roles, and S3 access policies.",
                        recentMessages: [
                            SessionMessage(
                                role: "user",
                                text:
                                    "The cluster rollout still needs subnet changes and the IAM role is blocking S3 access."
                            ),
                            SessionMessage(
                                role: "assistant",
                                text:
                                    "I am editing cluster.tf, the network module, and the IAM role policy to finish the rollout."
                            )
                        ]
                    )
                ],
                cases: [
                    TestCase(message: "add skeleton states to the onboarding flow", expected: "jules"),
                    TestCase(message: "fix the CSS grid layout", expected: "jules"),
                    TestCase(message: "add an endpoint for refund status", expected: "marco"),
                    TestCase(message: "fix the database connection pool", expected: "marco"),
                    TestCase(message: "update the terraform module for the new VPC", expected: "tess"),
                    TestCase(message: "fix the AWS IAM permissions", expected: "tess")
                ]
            ),
            Scenario(
                name: "project codenames",
                sessions: [
                    SessionCase(
                        name: "mercury",
                        cwd: "/Users/jud/projects/mercury",
                        context:
                            "Branch feat/workspace-shell. Editing client/src/shell/AppFrame.tsx, client/src/settings/PreferencesPanel.tsx, and client/src/styles/shell.css for workspace UI, keyboard shortcuts, and TypeScript component polish.",
                        recentMessages: [
                            SessionMessage(
                                role: "user",
                                text:
                                    "The preferences screen needs skeleton states, tighter spacing, and better keyboard shortcut hints."
                            ),
                            SessionMessage(
                                role: "assistant",
                                text:
                                    "I am polishing AppFrame.tsx, PreferencesPanel.tsx, and shell.css to fix the UI shell and settings flow."
                            )
                        ]
                    ),
                    SessionCase(
                        name: "saturn",
                        cwd: "/Users/jud/projects/saturn",
                        context:
                            "Branch feat/identity-sync. Editing server/routes/users.py, server/services/sync.py, and sql/user_sync.sql for user APIs, background sync jobs, and Postgres query correctness.",
                        recentMessages: [
                            SessionMessage(
                                role: "user",
                                text:
                                    "User sync still needs a status endpoint and the Postgres query for profiles is too slow."
                            ),
                            SessionMessage(
                                role: "assistant",
                                text:
                                    "I am adding the sync status endpoint in users.py and tightening the user_sync.sql query plan."
                            )
                        ]
                    ),
                    SessionCase(
                        name: "neptune",
                        cwd: "/Users/jud/projects/neptune",
                        context:
                            "Branch chore/deploy-networking. Editing deploy/envs/prod/main.tf, deploy/modules/vpc, and deploy/modules/iam for production networking, IAM hardening, and rollout automation.",
                        recentMessages: [
                            SessionMessage(
                                role: "user",
                                text:
                                    "The production rollout still needs IAM hardening, subnet changes, and the logging bucket policy update."
                            ),
                            SessionMessage(
                                role: "assistant",
                                text:
                                    "I am updating main.tf, the VPC module, and the IAM module to unblock the production deploy."
                            )
                        ]
                    )
                ],
                cases: [
                    TestCase(message: "add skeleton states to the settings screen", expected: "mercury"),
                    TestCase(message: "fix the postgres query for user sync", expected: "saturn"),
                    TestCase(message: "add an endpoint for sync status", expected: "saturn"),
                    TestCase(message: "rotate the IAM policy for the cluster", expected: "neptune"),
                    TestCase(message: "update the Terraform module for the private subnets", expected: "neptune"),
                    TestCase(message: "tighten the button spacing in the React shell", expected: "mercury")
                ]
            )
        ]

        var correct = 0
        var total = 0

        print("\n--- Routing Accuracy ---")

        for scenario in scenarios {
            print("  Scenario: \(scenario.name)")

            let sessions = scenario.sessions.enumerated().map { index, session in
                SessionState(
                    name: session.name,
                    tty: "/dev/benchmark\(index + 1)",
                    cwd: session.cwd,
                    context: session.context,
                    recentMessages: session.recentMessages,
                    status: .idle,
                    lastActivity: Date(),
                    voice: benchmarkVoice,
                    pitchMultiplier: 1.0
                )
            }
            let router = await makeBenchmarkRouter(sessions: sessions, engine: engine)

            for tc in scenario.cases {
                let result = await router.route(text: tc.message, routingState: RoutingState())
                let resolvedSession: String
                switch result {
                case .route(let session, _):
                    resolvedSession = session
                default:
                    resolvedSession = "<unresolved>"
                }
                let match = resolvedSession == tc.expected
                total += 1
                if match { correct += 1 }
                let mark = match ? "✓" : "✗"
                print("    \(mark) \"\(tc.message)\" -> \(resolvedSession) (expected: \(tc.expected))")
            }
        }

        let accuracy = Double(correct) / Double(total) * 100
        print("")
        print("Accuracy: \(correct)/\(total) (\(String(format: "%.0f", accuracy))%)")
        print("AC-007.5 (accuracy >=90%%): \(accuracy >= 90 ? "PASS" : "FAIL") (\(String(format: "%.0f", accuracy))%%)")
    }
}

// MARK: - TTS Benchmarks (AC-003.1)

@MainActor
func benchmarkTTS() async {
    print("\n=== TTS BENCHMARK ===\n")

    let modelManager = ModelManager()
    let ttsManager = Qwen3TTSSpeechManager(modelManager: modelManager)
    guard let appleVoice = AVSpeechSynthesisVoice(language: "en-US") else {
        print("FAIL: Could not create AVSpeechSynthesisVoice")
        return
    }
    let voice = VoiceDescriptor(
        appleVoice: appleVoice,
        qwenSpeakerID: "vivian"
    )

    // Cold start
    print("Loading TTS model (first run downloads ~2GB)...")
    let ttsProgress = await startModelProgressObserver(
        modelManager: modelManager,
        type: .tts,
        label: "TTS"
    )
    let coldStart = CFAbsoluteTimeGetCurrent()
    ttsManager.speak(
        "Hello, this is a test.",
        voice: voice,
        prefix: "test",
        pitchMultiplier: 1.0
    )

    var waited = 0.0
    while !ttsManager.isSpeaking && waited < 120 {
        try? await Task.sleep(nanoseconds: 10_000_000)
        waited += 0.01
    }
    ttsProgress.cancel()
    let coldMs = (CFAbsoluteTimeGetCurrent() - coldStart) * 1000
    print("Cold start (download+load+first-audio): \(String(format: "%.0f", coldMs))ms")

    // Wait for cold utterance to finish naturally (avoids stream cancellation bug)
    for await _ in ttsManager.finishedSpeaking {
        break
    }
    try? await Task.sleep(nanoseconds: 200_000_000)

    // Warm TTFA tests — let each utterance play out fully before starting the next.
    // Interrupting mid-stream causes the old MLX computation to hog the GPU
    // (upstream speech-swift stream cancellation issue).
    let testTexts = [
        "The quick brown fox jumps over the lazy dog.",
        "Please check your email for the confirmation link.",
        "I've updated the configuration file with the new settings.",
        "The deployment was successful. All services are running.",
        "There was an error processing your request."
    ]

    var warmLatencies: [Double] = []

    for text in testTexts {
        let start = CFAbsoluteTimeGetCurrent()
        ttsManager.speak(text, voice: voice, prefix: "agent", pitchMultiplier: 1.0)

        waited = 0.0
        while !ttsManager.isSpeaking && waited < 30 {
            try? await Task.sleep(nanoseconds: 1_000_000)
            waited += 0.001
        }
        let ttfaMs = (CFAbsoluteTimeGetCurrent() - start) * 1000
        warmLatencies.append(ttfaMs)
        print("  \(String(format: "%5.0f", ttfaMs))ms | \"\(text.prefix(50))...\"")

        // Wait for utterance to finish naturally before starting next
        for await _ in ttsManager.finishedSpeaking {
            break
        }
        try? await Task.sleep(nanoseconds: 200_000_000)
    }

    let sorted = warmLatencies.sorted()
    let avg = warmLatencies.reduce(0, +) / Double(warmLatencies.count)
    let p95 = sorted[Int(Double(sorted.count - 1) * 0.95)]

    print("\n--- Results ---")
    print("TTFA:  avg=\(String(format: "%.0f", avg))ms  p95=\(String(format: "%.0f", p95))ms")
    print("")
    print("AC-003.1 (TTFA <200ms): \(p95 < 200 ? "PASS" : "FAIL") (p95=\(String(format: "%.0f", p95))ms)")
}

// MARK: - STT Benchmarks (AC-001.3, AC-001.4)

@MainActor
func benchmarkSTT(
    runLatency: Bool,
    runLong: Bool,
    runStreaming: Bool,
    runSoak: Bool
) async {
    guard runLatency || runLong || runStreaming || runSoak else {
        return
    }

    print("\n=== STT BENCHMARK ===\n")

    let modelManager = ModelManager()
    let engine = ParakeetEngine(modelManager: modelManager)

    // Generate synthetic audio (440Hz sine wave at 16kHz)
    func makeSineBuffer(seconds: Double) -> AVAudioPCMBuffer? {
        let sampleRate: Double = 16000
        let frameCount = AVAudioFrameCount(seconds * sampleRate)
        guard
            let format = AVAudioFormat(
                standardFormatWithSampleRate: sampleRate,
                channels: 1
            )
        else { return nil }
        guard
            let buffer = AVAudioPCMBuffer(
                pcmFormat: format,
                frameCapacity: frameCount
            )
        else { return nil }
        buffer.frameLength = frameCount
        guard let data = buffer.floatChannelData?[0] else {
            return nil
        }
        for i in 0..<Int(frameCount) {
            data[i] =
                sin(
                    2.0 * .pi * 440.0 * Float(i) / Float(sampleRate)
                ) * 0.5
        }
        return buffer
    }

    if runLatency {
        print("Loading STT model (first run downloads ~400MB)...")
        let sttProgress = await startModelProgressObserver(
            modelManager: modelManager,
            type: .stt,
            label: "STT"
        )
        guard let buffer5s = makeSineBuffer(seconds: 5.0) else {
            sttProgress.cancel()
            print("FAIL: Could not create audio buffer")
            return
        }
        do {
            try engine.prepare()
        } catch {
            sttProgress.cancel()
            print("FAIL: prepare() failed: \(error)")
            return
        }
        engine.append(buffer5s)

        let coldStart = CFAbsoluteTimeGetCurrent()
        _ = await engine.finishAndTranscribe()
        sttProgress.cancel()
        let coldMs = (CFAbsoluteTimeGetCurrent() - coldStart) * 1000
        print("Cold transcription (5s audio, includes download+load): \(String(format: "%.0f", coldMs))ms")

        var latencies5s: [Double] = []
        for i in 1...3 {
            guard let buf = makeSineBuffer(seconds: 5.0) else {
                continue
            }
            do { try engine.prepare() } catch { continue }
            engine.append(buf)

            let start = CFAbsoluteTimeGetCurrent()
            _ = await engine.finishAndTranscribe()
            let ms = (CFAbsoluteTimeGetCurrent() - start) * 1000
            latencies5s.append(ms)
            print("  Run \(i): \(String(format: "%.0f", ms))ms (5s audio)")
        }

        let avg5s = latencies5s.isEmpty ? 0 : latencies5s.reduce(0, +) / Double(latencies5s.count)
        let max5s = latencies5s.max() ?? 0

        print("\n--- STT 5s Results ---")
        print("5s audio:  avg=\(String(format: "%.0f", avg5s))ms  max=\(String(format: "%.0f", max5s))ms")
        print("AC-001.3 (5s in <2s):     \(max5s < 2000 ? "PASS" : "FAIL") (max=\(String(format: "%.0f", max5s))ms)")
    }

    if runLong {
        print("\nLong utterance test (20s audio)...")
        guard let buffer20s = makeSineBuffer(seconds: 20.0) else {
            print("FAIL: Could not create 20s audio buffer")
            return
        }
        do { try engine.prepare() } catch {
            print("FAIL: prepare() failed for long utterance: \(error)")
            return
        }
        engine.append(buffer20s)

        let longStart = CFAbsoluteTimeGetCurrent()
        let longResult = await engine.finishAndTranscribe()
        let longMs = (CFAbsoluteTimeGetCurrent() - longStart) * 1000
        print("Long utterance (20s audio): \(String(format: "%.0f", longMs))ms, result: \(longResult ?? "nil")")
        print("AC-001.4 (20s no crash):  PASS (completed in \(String(format: "%.0f", longMs))ms)")
    }

    if runStreaming {
        print("\nStreaming finalize test (paced normal speech)...")
        await engine.prewarm()

        let warmupText = "Operator warmup. The agent is preparing the streaming transcription benchmark."
        print("Rendering warmup speech fixture...")
        guard let warmupAudio = await synthesizeSpeechAudio(text: warmupText) else {
            print("FAIL: Could not synthesize warmup audio")
            return
        }
        let warmupBuffers = makePCMChunks(
            samples: warmupAudio.samples,
            sampleRate: warmupAudio.sampleRate,
            chunkSeconds: 0.5
        )
        do { try engine.prepare() } catch {
            print("FAIL: prepare() failed for streaming warmup: \(error)")
            return
        }
        await feedBuffersRealTime(warmupBuffers, into: engine)
        _ = await engine.finishAndTranscribe()

        let speechText =
            """
            I reviewed the agent session, checked the changed Swift files, reran the focused benchmarks, and found that the routing path is warm and stable. Next I am streaming transcription in chunks so the final route only waits on the tail of the utterance instead of the full conversation. After that I will route the request immediately and send it to the right session.
            """
        print("Rendering benchmark speech fixture...")
        guard let speechAudio = await synthesizeSpeechAudio(text: speechText) else {
            print("FAIL: Could not synthesize streaming benchmark audio")
            return
        }
        let speechBuffers = makePCMChunks(
            samples: speechAudio.samples,
            sampleRate: speechAudio.sampleRate,
            chunkSeconds: 0.35
        )

        do { try engine.prepare() } catch {
            print("FAIL: prepare() failed for streaming benchmark: \(error)")
            return
        }

        let feedStart = CFAbsoluteTimeGetCurrent()
        await feedBuffersRealTime(speechBuffers, into: engine)
        let feedMs = (CFAbsoluteTimeGetCurrent() - feedStart) * 1000

        let finalizeStart = CFAbsoluteTimeGetCurrent()
        let result = await engine.finishAndTranscribe()
        let finalizeMs = (CFAbsoluteTimeGetCurrent() - finalizeStart) * 1000

        let durationSeconds = Double(speechAudio.samples.count) / speechAudio.sampleRate
        let metrics = engine.lastStreamingMetrics
        let tailDurationMs = Double(metrics.tailSamplesAtFinalize) / 16_000 * 1000
        print(
            "Streaming audio: \(String(format: "%.1f", durationSeconds))s"
                + " in \(speechBuffers.count) chunks"
        )
        print("Paced capture:    \(String(format: "%.0f", feedMs))ms")
        print("Finalize:         \(String(format: "%.0f", finalizeMs))ms")
        print("Segments queued:  \(metrics.queuedSegmentCount)")
        print("Forced segments:  \(metrics.forcedSegmentCount)")
        print("Tail at finalize: \(String(format: "%.0f", tailDurationMs))ms")
        print("Transcript chars: \(result?.count ?? 0)")
    }

    if runSoak {
        let soakMinutes =
            ProcessInfo.processInfo.environment["OPERATOR_STT_SOAK_MINUTES"]
            .flatMap(Double.init)
            .map { max(0.25, $0) }
            ?? 5.0
        let soakProgressInterval = 30.0
        print("\nStreaming soak test (\(String(format: "%.1f", soakMinutes)) min paced speech)...")
        await engine.prewarm()

        let soakText =
            """
            I am continuing the operator session, inspecting changed files, routing work across frontend, backend, and infrastructure tasks, and keeping the transcript current while I speak. The benchmark is simulating a long agent interaction so we can verify memory stays bounded, segments keep flushing during speech, and the final route does not wait on a large leftover tail.
            """
        print("Rendering soak speech fixture...")
        guard let soakAudio = await synthesizeSpeechAudio(text: soakText) else {
            print("FAIL: Could not synthesize soak benchmark audio")
            return
        }
        let soakBuffers = makePCMChunks(
            samples: soakAudio.samples,
            sampleRate: soakAudio.sampleRate,
            chunkSeconds: 0.35
        )
        let fixtureDurationSeconds = Double(soakAudio.samples.count) / soakAudio.sampleRate
        guard fixtureDurationSeconds > 0 else {
            print("FAIL: Soak fixture had zero duration")
            return
        }

        let targetDurationSeconds = soakMinutes * 60
        let baselineFootprintMB = physFootprintMB()
        var peakFootprintMB = baselineFootprintMB
        var progressCheckpoint = soakProgressInterval
        var totalChunksFed = 0
        var repetitionsCompleted = 0

        do { try engine.prepare() } catch {
            print("FAIL: prepare() failed for streaming soak: \(error)")
            return
        }

        let soakStart = CFAbsoluteTimeGetCurrent()
        while true {
            let elapsedSeconds = CFAbsoluteTimeGetCurrent() - soakStart
            let remainingSeconds = targetDurationSeconds - elapsedSeconds
            guard remainingSeconds > 0 else {
                break
            }

            let buffersToFeed = prefixBuffers(soakBuffers, maxDurationSeconds: remainingSeconds)
            await feedBuffersRealTime(buffersToFeed, into: engine)
            totalChunksFed += buffersToFeed.count
            repetitionsCompleted += 1

            let currentElapsedSeconds = min(
                targetDurationSeconds,
                CFAbsoluteTimeGetCurrent() - soakStart
            )
            let footprintMB = physFootprintMB()
            peakFootprintMB = max(peakFootprintMB, footprintMB)

            if currentElapsedSeconds >= progressCheckpoint || currentElapsedSeconds >= targetDurationSeconds {
                print(
                    "  Soak progress: \(String(format: "%.0f", currentElapsedSeconds))s"
                        + " | peak RSS +\(String(format: "%.0f", peakFootprintMB - baselineFootprintMB))MB"
                )
                progressCheckpoint += soakProgressInterval
            }
        }

        let captureMs = (CFAbsoluteTimeGetCurrent() - soakStart) * 1000
        let finalizeStart = CFAbsoluteTimeGetCurrent()
        let result = await engine.finishAndTranscribe()
        let finalizeMs = (CFAbsoluteTimeGetCurrent() - finalizeStart) * 1000
        let metrics = engine.lastStreamingMetrics
        let tailDurationMs = Double(metrics.tailSamplesAtFinalize) / 16_000 * 1000

        print("Soak capture:     \(String(format: "%.0f", captureMs))ms")
        print("Fixture duration: \(String(format: "%.1f", fixtureDurationSeconds))s x \(repetitionsCompleted)")
        print("Chunks fed:       \(totalChunksFed)")
        print("Finalize:         \(String(format: "%.0f", finalizeMs))ms")
        print("Segments queued:  \(metrics.queuedSegmentCount)")
        print("Forced segments:  \(metrics.forcedSegmentCount)")
        print("Tail at finalize: \(String(format: "%.0f", tailDurationMs))ms")
        print(
            "Peak RSS delta:   +\(String(format: "%.0f", peakFootprintMB - baselineFootprintMB))MB"
        )
        print("Transcript chars: \(result?.count ?? 0)")
    }
}

@MainActor
private func synthesizeSpeechAudio(
    text: String
) async -> (samples: [Float], sampleRate: Double)? {
    let outputURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("operator-benchmark-\(UUID().uuidString)")
        .appendingPathExtension("aiff")

    defer {
        try? FileManager.default.removeItem(at: outputURL)
    }

    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/say")
    process.arguments = ["-o", outputURL.path, text]

    do {
        try process.run()
        process.waitUntilExit()
    } catch {
        print("FAIL: could not launch say: \(error)")
        return nil
    }

    guard process.terminationStatus == 0 else {
        print("FAIL: say exited with status \(process.terminationStatus)")
        return nil
    }

    do {
        let audioFile = try AVAudioFile(forReading: outputURL)
        let format = audioFile.processingFormat
        let frameCount = AVAudioFrameCount(audioFile.length)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            return nil
        }
        try audioFile.read(into: buffer)
        return (samples: monoSamples(from: buffer), sampleRate: format.sampleRate)
    } catch {
        print("FAIL: could not load synthesized speech: \(error)")
        return nil
    }
}

private func monoSamples(from buffer: AVAudioPCMBuffer) -> [Float] {
    let frameCount = Int(buffer.frameLength)
    let channelCount = Int(buffer.format.channelCount)

    if let channelData = buffer.floatChannelData {
        if channelCount == 1 {
            return Array(UnsafeBufferPointer(start: channelData[0], count: frameCount))
        }

        return (0..<frameCount).map { frame in
            var sum: Float = 0
            for channel in 0..<channelCount {
                sum += channelData[channel][frame]
            }
            return sum / Float(channelCount)
        }
    }

    return []
}

private func makePCMChunks(
    samples: [Float],
    sampleRate: Double,
    chunkSeconds: Double
) -> [AVAudioPCMBuffer] {
    guard
        let format = AVAudioFormat(
            standardFormatWithSampleRate: sampleRate,
            channels: 1
        )
    else {
        return []
    }

    let chunkSize = max(1, Int(sampleRate * chunkSeconds))
    var chunks: [AVAudioPCMBuffer] = []
    var offset = 0

    while offset < samples.count {
        let end = min(samples.count, offset + chunkSize)
        let slice = samples[offset..<end]
        let frameCount = AVAudioFrameCount(slice.count)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            break
        }
        buffer.frameLength = frameCount
        if let channelData = buffer.floatChannelData?[0] {
            for (index, value) in slice.enumerated() {
                channelData[index] = value
            }
        }
        chunks.append(buffer)
        offset = end
    }

    return chunks
}

private func prefixBuffers(
    _ buffers: [AVAudioPCMBuffer],
    maxDurationSeconds: Double
) -> [AVAudioPCMBuffer] {
    guard maxDurationSeconds > 0 else {
        return []
    }

    var selected: [AVAudioPCMBuffer] = []
    var accumulatedSeconds = 0.0

    for buffer in buffers {
        let durationSeconds = bufferDurationMs(buffer) / 1000
        guard accumulatedSeconds < maxDurationSeconds else {
            break
        }
        selected.append(buffer)
        accumulatedSeconds += durationSeconds
    }

    return selected.isEmpty ? [buffers[0]] : selected
}

@MainActor
private func feedBuffersRealTime(
    _ buffers: [AVAudioPCMBuffer],
    into engine: ParakeetEngine,
    driftToleranceMs: Double = 8
) async {
    let start = CFAbsoluteTimeGetCurrent()
    var scheduledElapsedMs = 0.0

    for buffer in buffers {
        engine.append(buffer)
        scheduledElapsedMs += bufferDurationMs(buffer)

        let actualElapsedMs = (CFAbsoluteTimeGetCurrent() - start) * 1000
        let remainingMs = scheduledElapsedMs - actualElapsedMs
        if remainingMs > driftToleranceMs {
            try? await Task.sleep(nanoseconds: UInt64(remainingMs * 1_000_000))
        } else {
            await Task.yield()
        }
    }
}

private func bufferDurationMs(_ buffer: AVAudioPCMBuffer) -> Double {
    guard buffer.format.sampleRate > 0 else {
        return 0
    }
    return (Double(buffer.frameLength) / buffer.format.sampleRate) * 1000
}

private func physFootprintMB() -> Double {
    var rusage = rusage_info_v4()
    let result = withUnsafeMutablePointer(to: &rusage) {
        $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
            proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0)
        }
    }
    guard result == 0 else {
        return 0
    }
    return Double(rusage.ri_phys_footprint) / 1_048_576
}

// MARK: - Memory Benchmark (AC-011.5)

@MainActor
func benchmarkMemory() async {
    print("\n=== MEMORY BENCHMARK ===\n")

    let baseline = physFootprintMB()
    print("Baseline: \(String(format: "%.0f", baseline))MB")

    // Load routing model
    let modelManager = ModelManager()
    let routingProgress = await startModelProgressObserver(
        modelManager: modelManager,
        type: .routing,
        label: "Routing"
    )
    let routingEngine = MLXRoutingEngine(modelManager: modelManager)
    let routingPrompt = """
        Sessions:
        1. "test" (cwd: /tmp/test) - test project
        User message: "hello"
        """
    do {
        _ = try await routingEngine.run(prompt: routingPrompt, timeout: 120)
    } catch {
        print("Routing model load failed: \(error)")
    }
    routingProgress.cancel()
    let afterRouting = physFootprintMB()
    let routingDelta = afterRouting - baseline
    print(
        "After routing model: \(String(format: "%.0f", afterRouting))MB"
            + " (+\(String(format: "%.0f", routingDelta))MB)"
    )

    // Load TTS model
    let ttsProgress = await startModelProgressObserver(
        modelManager: modelManager,
        type: .tts,
        label: "TTS"
    )
    let ttsManager = Qwen3TTSSpeechManager(modelManager: modelManager)
    guard let memVoice = AVSpeechSynthesisVoice(language: "en-US") else {
        print("FAIL: Could not create AVSpeechSynthesisVoice")
        return
    }
    let voice = VoiceDescriptor(
        appleVoice: memVoice,
        qwenSpeakerID: "vivian"
    )
    ttsManager.speak(
        "Memory test.",
        voice: voice,
        prefix: "test",
        pitchMultiplier: 1.0
    )
    // Wait for utterance to complete naturally so GPU buffers settle
    for await _ in ttsManager.finishedSpeaking {
        break
    }
    ttsProgress.cancel()
    ttsManager.stop()
    try? await Task.sleep(nanoseconds: 1_000_000_000)

    let afterTTS = physFootprintMB()
    let ttsDelta = afterTTS - afterRouting
    print(
        "After TTS model:     \(String(format: "%.0f", afterTTS))MB"
            + " (+\(String(format: "%.0f", ttsDelta))MB)"
    )

    // STT stays loaded between transcriptions for performance.
    // Peak = routing + TTS + STT during transcription.
    let sttProgress = await startModelProgressObserver(
        modelManager: modelManager,
        type: .stt,
        label: "STT"
    )
    let sttEngine = ParakeetEngine(modelManager: modelManager)
    guard
        let format = AVAudioFormat(
            standardFormatWithSampleRate: 16000,
            channels: 1
        )
    else {
        print("FAIL: Could not create AVAudioFormat")
        return
    }
    guard
        let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: 16000
        )
    else {
        print("FAIL: Could not create AVAudioPCMBuffer")
        return
    }
    buffer.frameLength = 16000
    guard let data = buffer.floatChannelData?[0] else {
        print("FAIL: Could not access floatChannelData")
        return
    }
    for i in 0..<16000 {
        data[i] =
            sin(
                2.0 * .pi * 440.0 * Float(i) / 16000.0
            ) * 0.5
    }
    do { try sttEngine.prepare() } catch {}
    sttEngine.append(buffer)
    _ = await sttEngine.finishAndTranscribe()
    sttProgress.cancel()

    let peakWithSTT = physFootprintMB()
    let sttDelta = peakWithSTT - afterTTS
    print(
        "Peak (all models):   \(String(format: "%.0f", peakWithSTT))MB"
            + " (+\(String(format: "%.0f", sttDelta))MB for STT)"
    )

    // Explicitly unload STT to show post-unload memory
    sttEngine.unloadModel()
    try? await Task.sleep(nanoseconds: 200_000_000)
    let afterUnload = physFootprintMB()
    print("After STT unload:    \(String(format: "%.0f", afterUnload))MB")

    let peak = max(afterRouting, afterTTS, peakWithSTT)
    print("\n--- Results ---")
    print("Peak resident memory: \(String(format: "%.0f", peak))MB")
    print("")
    print("AC-011.5 (peak <3GB): \(peak < 3072 ? "PASS" : "FAIL") (\(String(format: "%.0f", peak))MB)")
}

// Entry point
await runBenchmarks()

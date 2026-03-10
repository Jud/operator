# Plan: Local Routing Engine (Qwen 3.5 0.8B)

Replace `claude -p` subprocess routing with a local Qwen 3.5 0.8B model for faster, more reliable message routing that doesn't depend on the Claude CLI.

## Current State

The `MessageRouter` uses a priority chain for routing user messages to Claude Code sessions:

1. **Operator commands** — regex match ("operator status", "list agents", etc.)
2. **Keyword extraction** — regex: `/(?:tell |hey |@)(\w+)[,:]?\s*(.*)/i`
3. **Single-session bypass** — if only 1 session, route directly
4. **Session affinity** — if last routed <15s ago, reuse target
5. **`claude -p` smart routing** — shells out to Claude CLI, parses JSON response
6. **Ambiguous fallback** — ask the user to clarify

Steps 1-4 are fast (regex/lookup). Step 5 spawns a full `claude -p` subprocess, which:
- Takes 2-5 seconds (cold) or 1-2s (warm)
- Depends on Claude CLI being installed and authenticated
- Uses the user's API quota for a trivial classification task
- Has a 10-second timeout

## Why Qwen 3.5 0.8B

This is a **classification task**, not a creative task. Given 2-5 sessions with names, cwds, and context, pick which one a message belongs to. A small local model is ideal:

- **Qwen 3.5 0.8B** — newest architecture (hybrid Gated Delta Networks + sparse MoE), natively multimodal
- **622MB download** (4-bit quantized via `mlx-community/Qwen3.5-0.8B-MLX-4bit`)
- **~500MB RAM** resident, ~50-200ms inference for short prompts on Apple Silicon
- No subprocess spawn overhead, no API key, no network latency
- Deterministic output with temperature 0 (ArgMaxSampler)
- Shares MLX runtime with speech-swift

## SPM Dependencies

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift-lm/", .upToNextMinor(from: "2.29.1")),
    .package(url: "https://github.com/petrukha-ivan/mlx-swift-structured", from: "0.0.2"),
]

// Target
.product(name: "MLXLLM", package: "mlx-swift-lm"),
.product(name: "MLXSwiftStructured", package: "mlx-swift-structured"),
```

## Architecture

### New: `MLXRoutingEngine`

```swift
import MLXLLM
import MLXLMCommon
import MLXSwiftStructured

public actor MLXRoutingEngine: RoutingEngine {
    private var container: ModelContainer?

    private static let routingSchema = JSONSchema.object(
        properties: [
            "session": .oneOf([.string(), .null()]),
            "confident": .boolean(),
            "candidates": .array(items: .string()),
            "question": .string(),
        ],
        required: ["session", "confident"]
    )

    public init() {}

    /// Load the model on first use, keep resident for fast repeated inference.
    private func ensureLoaded() async throws -> ModelContainer {
        if let container { return container }
        let container = try await LLMModelFactory.shared.loadContainer(
            configuration: .init(id: "mlx-community/Qwen3.5-0.8B-MLX-4bit")
        ) { progress in
            // Could surface download progress to UI
        }
        self.container = container
        return container
    }

    public func run(prompt: String, timeout: TimeInterval) async throws -> [String: Any] {
        let container = try await ensureLoaded()
        let grammar = try Grammar.schema(Self.routingSchema)

        let result = try await container.perform { context in
            let input = try await context.processor.prepare(
                input: UserInput(prompt: prompt)
            )
            return try MLXLMCommon.generate(
                input: input,
                parameters: GenerateParameters(temperature: 0.0),
                context: context
            ) { tokens in
                tokens.count >= 100 ? .stop : .more
            }
        }

        let output = result.output
        guard let data = output.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            throw ClaudePipeError.invalidResponse(output: output)
        }
        return json
    }
}
```

### Integration Point

The `RoutingEngine` protocol is already in place. `MessageRouter` takes any `RoutingEngine` via init injection:

```swift
// Current:
MessageRouter(registry: registry, engine: ClaudePipeRoutingEngine())

// New:
MessageRouter(registry: registry, engine: MLXRoutingEngine())
```

Zero changes to `MessageRouter` routing logic — it just gets JSON back faster.

## Implementation Plan

### Phase 1: SPM + Model Loading (Day 1)

- [ ] Add `mlx-swift-lm` and `mlx-swift-structured` to Package.swift
- [ ] Verify SPM resolution (no conflicts with Hummingbird or speech-swift deps)
- [ ] Create `MLXRoutingEngine` actor conforming to `RoutingEngine`
- [ ] Implement lazy model loading with `LLMModelFactory.shared.loadContainer()`
- [ ] Test model download + first inference round-trip

### Phase 2: Constrained Decoding + Prompt (Day 2)

- [ ] Define JSON schema for routing response via `mlx-swift-structured`
- [ ] Use grammar-constrained decoding to guarantee valid JSON (no need for `ClaudePipe.extractJSON`)
- [ ] Optimize prompt for 0.8B model with few-shot examples:
  ```
  You are a message router. Route the user's message to the correct session.

  Sessions:
  1. "frontend" (/Users/jud/web-app) - React dashboard
  2. "backend" (/Users/jud/api-server) - FastAPI endpoints

  Examples:
  User: "add a loading spinner" -> {"session": "frontend", "confident": true}
  User: "fix the database query" -> {"session": "backend", "confident": true}
  User: "deploy everything" -> {"session": null, "confident": false, "candidates": ["frontend", "backend"], "question": "Was that for frontend or backend?"}

  User: "[actual message]" ->
  ```
- [ ] Use non-thinking mode for fast direct answers
- [ ] Temperature 0 with ArgMaxSampler for deterministic output

### Phase 3: Fallback + Wiring (Day 3)

- [ ] Wire `MLXRoutingEngine` into `OperatorApp.swift`
- [ ] Add setting: "Routing engine" → Local / Claude CLI
- [ ] Fallback to `ClaudePipeRoutingEngine` if model fails to load
- [ ] Log routing latency for comparison
- [ ] Route clarification responses through same engine (`MessageRouter+Clarification.swift`)

### Phase 4: Testing (Day 4)

- [ ] Test with 2-5 sessions with varying contexts
- [ ] Test edge cases: ambiguous messages, very short messages, non-English
- [ ] Test ordinal clarification responses ("the first one")
- [ ] Benchmark: measure p50/p95 latency on M1/M4
- [ ] Verify constrained decoding always produces parseable JSON

## Shared MLX Runtime

If we're already loading MLX for speech-swift (Qwen3-TTS), we share the runtime:

```
MLX Runtime (shared)
├── Qwen3-TTS model (~2GB) — loaded for TTS
├── Parakeet CoreML model (~400MB) — loaded for STT (runs on ANE, not MLX)
└── Qwen 3.5 0.8B (~622MB) — loaded for routing
```

Total MLX memory: ~2.6GB. Parakeet runs on Neural Engine separately.

## Performance Targets

| Metric | Current (claude -p) | Target (Qwen 3.5 0.8B) |
|--------|---------------------|--------------------------|
| Latency | 1-5 seconds | 50-200ms |
| Cold start | 2-5 seconds | ~500ms (model load, one-time) |
| Reliability | Depends on CLI + auth | Always available |
| API cost | Uses user's quota | Zero |
| JSON validity | Needs extraction from free text | Guaranteed (constrained decoding) |
| Accuracy | Very high (Claude) | High for classification |

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| 0.8B too small for nuanced routing | Low | This is simple classification; few-shot + constrained decoding handles it. Fall back to claude -p if needed |
| Model adds ~500MB memory | Low | Acceptable; can unload when single-session (routing not needed) |
| Constrained decoding adds latency | Low | 15-25% overhead on ~100ms = negligible |
| MLX dep conflicts with speech-swift | Low | Both use mlx-swift; compatible |
| First-run model download UX | Medium | Surface progress to Settings panel; ~622MB download |

## Files to Modify

| File | Change |
|------|--------|
| `Package.swift` | Add mlx-swift-lm + mlx-swift-structured deps |
| `Router/RoutingEngine.swift` | No change (protocol already generic) |
| `Router/MessageRouter.swift` | No change (uses injected engine) |
| `App/OperatorApp.swift` | Wire `MLXRoutingEngine` instead of `ClaudePipeRoutingEngine` |
| `UI/SettingsView.swift` | Add routing engine toggle |

## New Files

| File | Purpose |
|------|---------|
| `Router/MLXRoutingEngine.swift` | Qwen 3.5 0.8B MLX inference + constrained decoding |

## Future: Apple Foundation Models (macOS 26+)

When Operator's deployment target moves to macOS 26, we can replace this with Apple's built-in Foundation Models framework:
```swift
let session = LanguageModelSession()
@Generable struct RoutingResult { ... }
let result = try await session.respond(to: prompt, generating: RoutingResult.self)
```
Zero dependencies, built-in constrained decoding via `@Generable`, no model download. But macOS 26 isn't stable yet.

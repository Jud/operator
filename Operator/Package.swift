// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Operator",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.0.0"),
        .package(url: "https://github.com/jud/speech-swift.git", branch: "operator-fork"),
        .package(
            url: "https://github.com/ml-explore/mlx-swift-lm.git",
            branch: "main"
        ),
        .package(
            url: "https://github.com/Jud/mlx-swift-structured.git",
            branch: "main"
        ),
    ],
    targets: [
        .target(
            name: "OperatorShared",
            path: "Sources/OperatorShared"
        ),
        .target(
            name: "OperatorCore",
            dependencies: [
                "OperatorShared",
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "AudioCommon", package: "speech-swift"),
                .product(name: "Qwen3TTS", package: "speech-swift"),
                .product(name: "ParakeetASR", package: "speech-swift"),
                .product(name: "SpeechVAD", package: "speech-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXStructured", package: "mlx-swift-structured"),
            ],
            path: "Sources",
            exclude: [
                "App/OperatorApp.swift",
                "OperatorShared/",
                "MCPServer/",
            ],
            resources: [
                .process("Resources"),
            ]
        ),
        .executableTarget(
            name: "Operator",
            dependencies: ["OperatorCore"],
            path: "Sources/App"
        ),
        .target(
            name: "OperatorMCPCore",
            dependencies: ["OperatorShared"],
            path: "Sources/MCPServer",
            exclude: ["Entry/OperatorMCPApp.swift"]
        ),
        .executableTarget(
            name: "OperatorMCP",
            dependencies: ["OperatorMCPCore"],
            path: "Sources/MCPServer/Entry"
        ),
        .executableTarget(
            name: "E2ETests",
            dependencies: ["OperatorCore"],
            path: "Tests/E2E"
        ),
        .testTarget(
            name: "OperatorMCPTests",
            dependencies: ["OperatorMCPCore"],
            path: "Tests/OperatorMCPTests"
        ),
        .testTarget(
            name: "OperatorTests",
            dependencies: ["OperatorCore"],
            path: "Tests/OperatorTests"
        ),
        .executableTarget(
            name: "Benchmarks",
            dependencies: ["OperatorCore"],
            path: "Tests/Benchmarks"
        ),
    ]
)

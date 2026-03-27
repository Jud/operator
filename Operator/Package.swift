// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Operator",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.0.0"),
        .package(url: "git@github.com:Jud/harness.git", from: "0.1.0"),
        .package(url: "https://github.com/Jud/kokoro-coreml.git", from: "0.6.0"),
        .package(url: "https://github.com/Jud/vocabulary-corrector-swift.git", from: "0.1.0"),
        .package(url: "https://github.com/argmaxinc/WhisperKit.git", from: "0.9.0"),
    ],
    targets: [
        .target(
            name: "OperatorShared",
            path: "Sources/OperatorShared"
        ),
        .target(
            name: "QwenTokenizer",
            path: "Sources/Tokenizer"
        ),
        .binaryTarget(
            name: "QwenTokenizerFFI",
            path: "Frameworks/QwenTokenizerFFI.xcframework"
        ),
        .target(
            name: "QwenTokenizerRust",
            dependencies: ["QwenTokenizerFFI"],
            path: "Sources/QwenTokenizerRust"
        ),
        .target(
            name: "OperatorCore",
            dependencies: [
                "OperatorShared",
                "QwenTokenizer",
                .product(name: "KokoroCoreML", package: "kokoro-coreml"),
                .product(name: "WhisperKit", package: "WhisperKit"),
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "HarnessCore", package: "Harness"),
                .product(name: "VocabularyCorrector", package: "vocabulary-corrector-swift"),
            ],
            path: "Sources",
            exclude: [
                "App/",
                "OperatorShared/",
                "MCPServer/",
                "OperatorCLI/",
                "Tokenizer/",
                "CleanupCLI/",
                "QwenTokenizerRust/",
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
            dependencies: ["OperatorMCPCore", "OperatorShared"],
            path: "Sources/MCPServer/Entry"
        ),
        .executableTarget(
            name: "OperatorCLI",
            dependencies: ["OperatorShared"],
            path: "Sources/OperatorCLI"
        ),
        .executableTarget(
            name: "CleanupCLI",
            dependencies: ["QwenTokenizerRust"],
            path: "Sources/CleanupCLI",
            exclude: ["tokenizer_helper.py"]
        ),
        .executableTarget(
            name: "E2ETests",
            dependencies: ["OperatorCore"],
            path: "Tests/E2E"
        ),
        .testTarget(
            name: "OperatorMCPTests",
            dependencies: ["OperatorMCPCore", "OperatorShared"],
            path: "Tests/OperatorMCPTests"
        ),
        .testTarget(
            name: "OperatorTests",
            dependencies: ["OperatorCore"],
            path: "Tests/OperatorTests"
        ),
        .executableTarget(
            name: "Benchmarks",
            dependencies: ["OperatorCore", .product(name: "KokoroCoreML", package: "kokoro-coreml")],
            path: "Tests/Benchmarks"
        ),
        .executableTarget(
            name: "TranscriptionTests",
            dependencies: ["OperatorCore"],
            path: "Tests/TranscriptionTests"
        ),
    ]
)

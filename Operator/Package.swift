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
            name: "OperatorCore",
            dependencies: [
                "OperatorShared",
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

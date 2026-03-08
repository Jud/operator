// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Operator",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.0.0"),
        .package(url: "https://github.com/jud/speech-swift.git", branch: "operator-fork"),
    ],
    targets: [
        .target(
            name: "OperatorCore",
            dependencies: [
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "AudioCommon", package: "speech-swift"),
                .product(name: "Qwen3TTS", package: "speech-swift"),
                .product(name: "ParakeetASR", package: "speech-swift"),
            ],
            path: "Sources",
            exclude: ["App/OperatorApp.swift"],
            resources: [
                .process("Resources"),
            ]
        ),
        .executableTarget(
            name: "Operator",
            dependencies: ["OperatorCore"],
            path: "Sources/App"
        ),
        .executableTarget(
            name: "E2ETests",
            dependencies: ["OperatorCore"],
            path: "Tests/E2E"
        ),
        .testTarget(
            name: "OperatorTests",
            dependencies: ["OperatorCore"],
            path: "Tests/OperatorTests"
        ),
    ]
)

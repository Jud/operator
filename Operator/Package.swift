// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Operator",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.0.0"),
        .package(path: "../../harness"),
        .package(path: "../KokoroTTS"),
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
                "KokoroTTS",
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "HarnessCore", package: "Harness"),
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
            dependencies: ["OperatorCore"],
            path: "Tests/Benchmarks"
        ),
    ]
)

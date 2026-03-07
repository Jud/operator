// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Operator",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.0.0"),
    ],
    targets: [
        .target(
            name: "OperatorCore",
            dependencies: [
                .product(name: "Hummingbird", package: "hummingbird"),
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

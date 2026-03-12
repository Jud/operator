// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "KokoroTTS",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "KokoroTTS", targets: ["KokoroTTS"]),
    ],
    targets: [
        .target(
            name: "KokoroTTS",
            path: "Sources/KokoroTTS",
            resources: [
                .process("Resources"),
            ]
        ),
        .testTarget(
            name: "KokoroTTSTests",
            dependencies: ["KokoroTTS"],
            path: "Tests/KokoroTTSTests"
        ),
    ]
)

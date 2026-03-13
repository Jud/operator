// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "KokoroTTS",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "KokoroTTS", targets: ["KokoroTTS"]),
        .executable(name: "kokoro-say", targets: ["CLI"]),
    ],
    targets: [
        .target(
            name: "KokoroTTS",
            path: "Sources/KokoroTTS",
            resources: [
                .process("Resources"),
            ]
        ),
        .executableTarget(
            name: "CLI",
            dependencies: ["KokoroTTS"],
            path: "Sources/CLI"
        ),
        .testTarget(
            name: "KokoroTTSTests",
            dependencies: ["KokoroTTS"],
            path: "Tests/KokoroTTSTests"
        ),
    ]
)

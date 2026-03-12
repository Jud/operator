// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "KokoroTTS",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "KokoroTTS", targets: ["KokoroTTS"]),
    ],
    dependencies: [
        .package(url: "https://github.com/mlalma/MisakiSwift", from: "1.0.1"),
    ],
    targets: [
        .target(
            name: "KokoroTTS",
            dependencies: ["MisakiSwift"],
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

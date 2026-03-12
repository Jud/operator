// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "kokoro-two-stage",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
        .package(path: "../../KokoroTTS"),
    ],
    targets: [
        .executableTarget(
            name: "kokoro-spike",
            dependencies: [
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                "KokoroTTS",
            ]
        ),
    ]
)

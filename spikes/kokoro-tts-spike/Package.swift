// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "kokoro-tts-spike",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/jud/speech-swift.git", branch: "operator-fork"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
    ],
    targets: [
        .executableTarget(
            name: "kokoro-spike",
            dependencies: [
                .product(name: "KokoroTTS", package: "speech-swift"),
                .product(name: "AudioCommon", package: "speech-swift"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
    ]
)

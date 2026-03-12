import Foundation
import ArgumentParser
import KokoroTTS
import AudioCommon
import AVFoundation

@main
struct KokoroSpike: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "kokoro-spike",
        abstract: "Kokoro-82M TTS spike — synthesize text and play through speakers"
    )

    @Argument(help: "Text to synthesize (omit with --list-voices)")
    var text: String?

    @Option(name: .long, help: "Voice preset (default: af_heart)")
    var voice: String = "af_heart"

    @Option(name: .long, help: "Playback speed multiplier (0.5–2.0, default: 1.0)")
    var speed: Float = 1.0

    @Option(name: .shortAndLong, help: "Save WAV to path (optional)")
    var output: String?

    @Flag(name: .long, help: "List available voices and exit")
    var listVoices: Bool = false

    @Flag(name: .long, help: "Don't play audio, just synthesize")
    var noPlay: Bool = false

    @Flag(name: .long, help: "Try all voices with the same text")
    var demo: Bool = false

    @Flag(name: .long, help: "Run benchmark: warmup + multiple synthesis calls")
    var benchmark: Bool = false

    func validate() throws {
        if text == nil && !listVoices && !demo && !benchmark {
            throw ValidationError("Provide text, --list-voices, --demo, or --benchmark")
        }
        if speed < 0.5 || speed > 2.0 {
            throw ValidationError("Speed must be between 0.5 and 2.0")
        }
    }

    func run() async throws {
        print("Loading Kokoro-82M model...")
        let startLoad = CFAbsoluteTimeGetCurrent()
        let model = try await KokoroTTSModel.fromPretrained { progress, message in
            if progress < 1.0 {
                print("  \(message) (\(Int(progress * 100))%)")
            }
        }
        let loadTime = CFAbsoluteTimeGetCurrent() - startLoad
        print(String(format: "Model loaded in %.1fs", loadTime))

        let voices = model.availableVoices

        if listVoices {
            print("\nAvailable voices (\(voices.count)):")
            let grouped = Dictionary(grouping: voices) { String($0.prefix(1)) }
            let langNames = [
                "a": "American English", "b": "British English",
                "e": "Spanish", "f": "French", "h": "Hindi",
                "i": "Italian", "j": "Japanese", "k": "Korean",
                "p": "Portuguese", "z": "Chinese"
            ]
            for key in grouped.keys.sorted() {
                let lang = langNames[key] ?? key
                print("\n  \(lang):")
                for v in grouped[key]!.sorted() {
                    let gender = v.dropFirst().first == "f" ? "F" : "M"
                    let name = String(v.dropFirst(3))
                    print("    \(v)  (\(gender)) \(name)")
                }
            }
            return
        }

        if benchmark {
            try await runBenchmark(model: model)
            return
        }

        if demo {
            let demoText = text ?? "The quick brown fox jumps over the lazy dog."
            let demoVoices = voices.filter { $0.hasPrefix("a") || $0.hasPrefix("b") }.prefix(8)
            print("\nDemo: \"\(demoText)\"\n")
            for v in demoVoices {
                print("--- \(v) ---")
                let audio = try model.synthesize(text: demoText, voice: v, language: "en")
                try await playAudio(samples: audio, sampleRate: KokoroTTSModel.outputSampleRate, speed: speed)
                print()
            }
            return
        }

        guard let inputText = text else { return }

        print("\nSynthesizing: \"\(inputText)\"")
        print("  Voice: \(voice)")
        print("  Speed: \(speed)x")

        let t0 = CFAbsoluteTimeGetCurrent()
        let audio = try model.synthesize(text: inputText, voice: voice, language: "en")
        let synthTime = CFAbsoluteTimeGetCurrent() - t0
        let duration = Double(audio.count) / Double(KokoroTTSModel.outputSampleRate)

        print(String(format: "  Synth: %.1fms, Duration: %.2fs, RTFx: %.1f",
                      synthTime * 1000, duration, duration / synthTime))

        if let outputPath = output {
            let url = URL(fileURLWithPath: outputPath)
            try WAVWriter.write(samples: audio, sampleRate: KokoroTTSModel.outputSampleRate, to: url)
            print("  Saved: \(outputPath)")
        }

        if !noPlay {
            try await playAudio(samples: audio, sampleRate: KokoroTTSModel.outputSampleRate, speed: speed)
        }
    }

    // MARK: - Benchmark

    func runBenchmark(model: KokoroTTSModel) async throws {
        let testCases: [(label: String, text: String)] = [
            ("short",  "Hello world."),
            ("medium", "The deployment completed successfully. Three services updated, zero downtime."),
            ("long",   "I found three critical vulnerabilities in the authentication module. The first is an SQL injection in the login handler. The second is a missing CSRF token on the password reset form. The third is an insecure direct object reference in the user profile API. Patching all three now."),
        ]
        let testVoices = ["am_adam", "am_onyx", "af_heart", "bm_george"]

        print("\n=== BENCHMARK: Warmup Effect ===")
        print("Running 5 sequential synthesis calls to measure warmup improvement...\n")

        let warmupText = "This is a warmup sentence to initialize the CoreML model."
        for i in 1...5 {
            let t0 = CFAbsoluteTimeGetCurrent()
            let audio = try model.synthesize(text: warmupText, voice: "am_adam", language: "en")
            let elapsed = CFAbsoluteTimeGetCurrent() - t0
            let duration = Double(audio.count) / Double(KokoroTTSModel.outputSampleRate)
            print(String(format: "  Run %d: %.1fms synth, %.2fs audio, RTFx=%.1f",
                         i, elapsed * 1000, duration, duration / elapsed))
        }

        print("\n=== BENCHMARK: Text Length Impact ===")
        print("Testing short/medium/long text to measure fixed overhead vs proportional cost...\n")

        for tc in testCases {
            var times = [Double]()
            var durations = [Double]()
            for _ in 1...3 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let audio = try model.synthesize(text: tc.text, voice: "am_adam", language: "en")
                let elapsed = CFAbsoluteTimeGetCurrent() - t0
                let dur = Double(audio.count) / Double(KokoroTTSModel.outputSampleRate)
                times.append(elapsed)
                durations.append(dur)
            }
            let avgTime = times.reduce(0, +) / Double(times.count)
            let avgDur = durations.reduce(0, +) / Double(durations.count)
            let minTime = times.min()!
            print(String(format: "  %-7s: avg %.1fms (min %.1fms), %.2fs audio, RTFx=%.1f, chars=%d",
                         tc.label, avgTime * 1000, minTime * 1000, avgDur, avgDur / avgTime, tc.text.count))
        }

        print("\n=== BENCHMARK: Voice Switching Overhead ===")
        print("Testing if switching voices adds latency...\n")

        let voiceText = "Testing voice switching overhead."
        // First: same voice repeatedly
        var sameVoiceTimes = [Double]()
        for _ in 1...4 {
            let t0 = CFAbsoluteTimeGetCurrent()
            _ = try model.synthesize(text: voiceText, voice: "am_adam", language: "en")
            sameVoiceTimes.append(CFAbsoluteTimeGetCurrent() - t0)
        }
        let avgSame = sameVoiceTimes.reduce(0, +) / Double(sameVoiceTimes.count)

        // Then: different voice each time
        var diffVoiceTimes = [Double]()
        for v in testVoices {
            let t0 = CFAbsoluteTimeGetCurrent()
            _ = try model.synthesize(text: voiceText, voice: v, language: "en")
            diffVoiceTimes.append(CFAbsoluteTimeGetCurrent() - t0)
        }
        let avgDiff = diffVoiceTimes.reduce(0, +) / Double(diffVoiceTimes.count)

        print(String(format: "  Same voice (x4):      avg %.1fms", avgSame * 1000))
        print(String(format: "  Different voices (x4): avg %.1fms", avgDiff * 1000))
        print(String(format: "  Overhead: %.1fms (%.1f%%)", (avgDiff - avgSame) * 1000, ((avgDiff / avgSame) - 1) * 100))

        print("\n=== BENCHMARK: Speed Control Quality ===")
        print("Testing speed parameter effect on audio duration...\n")

        let speedText = "The quick brown fox jumps over the lazy dog."
        let speeds: [Float] = [0.8, 1.0, 1.2, 1.4, 1.6]
        for spd in speeds {
            // Speed is post-processing in our spike, so audio length won't change
            // but playback duration will
            let t0 = CFAbsoluteTimeGetCurrent()
            let audio = try model.synthesize(text: speedText, voice: "am_adam", language: "en")
            let elapsed = CFAbsoluteTimeGetCurrent() - t0
            let duration = Double(audio.count) / Double(KokoroTTSModel.outputSampleRate)
            let playDuration = duration / Double(spd)
            print(String(format: "  Speed %.1fx: synth %.1fms, audio %.2fs, play %.2fs, RTFx=%.1f",
                         spd, elapsed * 1000, duration, playDuration, duration / elapsed))
        }

        print("\n=== SUMMARY ===")
        print("Model: Kokoro-82M CoreML (speech-swift, end-to-end)")
        print("Compute units: .all (default)")
        print("Model cache: ~/Library/Caches/qwen3-speech/models/aufklarer/Kokoro-82M-CoreML/")
        print("Speed control: post-processing via AVAudioUnitTimePitch (not native duration scaling)")
        print("\nKey finding: speech-swift uses a SINGLE end-to-end CoreML model.")
        print("FluidAudio splits into Duration Model (CPU/GPU) + HAR Decoder (ANE) for 23x RTFx.")
        print("The speed gap is architecture, not hardware.")
    }

    // MARK: - Audio Playback

    func playAudio(samples: [Float], sampleRate: Int, speed: Float) async throws {
        let engine = AVAudioEngine()
        let playerNode = AVAudioPlayerNode()
        engine.attach(playerNode)

        guard let format = AVAudioFormat(
            standardFormatWithSampleRate: Double(sampleRate),
            channels: 1
        ) else {
            print("Error: Could not create audio format")
            return
        }

        let frameCount = AVAudioFrameCount(samples.count)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            print("Error: Could not create audio buffer")
            return
        }

        buffer.frameLength = frameCount
        let channelData = buffer.floatChannelData![0]
        samples.withUnsafeBufferPointer { src in
            channelData.update(from: src.baseAddress!, count: samples.count)
        }

        let timePitch = AVAudioUnitTimePitch()
        timePitch.rate = speed
        timePitch.pitch = 0
        engine.attach(timePitch)

        engine.connect(playerNode, to: timePitch, format: format)
        engine.connect(timePitch, to: engine.mainMixerNode, format: format)

        try engine.start()
        playerNode.scheduleBuffer(buffer, completionHandler: nil)
        playerNode.play()

        let playDuration = Double(samples.count) / Double(sampleRate) / Double(speed)
        print(String(format: "  Playing (%.1fs at %.1fx)...", playDuration, speed))
        try await Task.sleep(nanoseconds: UInt64((playDuration + 0.3) * 1_000_000_000))

        playerNode.stop()
        engine.stop()
    }
}

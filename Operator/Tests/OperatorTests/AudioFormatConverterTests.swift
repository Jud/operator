import AVFoundation
import Testing

@testable import OperatorCore

@Suite("AudioFormatConverter")
internal struct AudioFormatConverterTests {
    @Test("converts 48kHz mono to 16kHz mono")
    func convert48kTo16k() throws {
        let inputFormat = try #require(
            AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: 48_000,
                channels: 1,
                interleaved: false
            )
        )

        let converter = try AudioFormatConverter(inputFormat: inputFormat)

        let buffer = try #require(AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: 4_800))
        buffer.frameLength = 4_800
        let channelData = try #require(buffer.floatChannelData?[0])
        for idx in 0..<4_800 {
            channelData[idx] = sin(Float(idx) * 2 * .pi * 440 / 48_000)
        }

        var samples: [Float] = []
        let success = converter.appendConverted(buffer, to: &samples)
        #expect(success)
        #expect(samples.count > 1_200)
        #expect(samples.count < 1_800)
    }

    @Test("converts 48kHz stereo to 16kHz mono")
    func convertStereoToMono() throws {
        let inputFormat = try #require(
            AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: 48_000,
                channels: 2,
                interleaved: false
            )
        )

        let converter = try AudioFormatConverter(inputFormat: inputFormat)

        let buffer = try #require(AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: 4_800))
        buffer.frameLength = 4_800
        for ch in 0..<2 {
            let channelData = try #require(buffer.floatChannelData?[ch])
            for idx in 0..<4_800 {
                channelData[idx] = sin(Float(idx) * 2 * .pi * 440 / 48_000)
            }
        }

        var samples: [Float] = []
        let success = converter.appendConverted(buffer, to: &samples)
        #expect(success)
        #expect(samples.count > 1_200)
        #expect(samples.count < 1_800)
    }

    @Test("handles empty buffer")
    func emptyBuffer() throws {
        let inputFormat = try #require(
            AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: 48_000,
                channels: 1,
                interleaved: false
            )
        )

        let converter = try AudioFormatConverter(inputFormat: inputFormat)

        let buffer = try #require(AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: 0))
        buffer.frameLength = 0

        var samples: [Float] = []
        let success = converter.appendConverted(buffer, to: &samples)
        #expect(!success || samples.isEmpty)
    }
}

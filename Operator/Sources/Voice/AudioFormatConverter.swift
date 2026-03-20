import AVFoundation

/// Converts audio buffers from the microphone's native format (typically 48kHz)
/// to 16kHz mono Float32 as required by Whisper models.
///
/// The `AVAudioConverter` is created once and reused across buffers. An output
/// buffer is lazily allocated and reused when the input size is stable (as it
/// is for audio tap buffers). Not inherently thread-safe — the caller
/// (`WhisperKitEngine`) serializes access via its session lock.
internal final class AudioFormatConverter: @unchecked Sendable {
    enum Error: Swift.Error {
        case formatCreationFailed
        case converterCreationFailed
    }

    private let converter: AVAudioConverter
    private let outputFormat: AVAudioFormat
    private let ratio: Double
    private var reusableBuffer: AVAudioPCMBuffer?

    /// Creates a converter from the given input format to 16kHz mono Float32.
    ///
    /// - Parameter inputFormat: The microphone's capture format.
    /// - Throws: If the output format or converter cannot be created.
    init(inputFormat: AVAudioFormat) throws {
        guard
            let output = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: 16_000,
                channels: 1,
                interleaved: false
            )
        else {
            throw Error.formatCreationFailed
        }
        guard let conv = AVAudioConverter(from: inputFormat, to: output) else {
            throw Error.converterCreationFailed
        }
        self.outputFormat = output
        self.converter = conv
        self.ratio = output.sampleRate / inputFormat.sampleRate
    }

    /// Convert a captured audio buffer and append the result to the destination array.
    ///
    /// - Parameters:
    ///   - buffer: The input buffer from the audio tap.
    ///   - destination: The array to append converted samples to.
    /// - Returns: True if samples were appended, false on failure or empty input.
    func appendConverted(_ buffer: AVAudioPCMBuffer, to destination: inout [Float]) -> Bool {
        guard buffer.frameLength > 0 else {
            return false
        }

        let needed = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 1
        let outBuf: AVAudioPCMBuffer
        if let existing = reusableBuffer, existing.frameCapacity >= needed {
            outBuf = existing
        } else {
            guard let buf = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: needed) else {
                return false
            }
            reusableBuffer = buf
            outBuf = buf
        }
        outBuf.frameLength = 0
        converter.reset()

        var error: NSError?
        var consumed = false
        converter.convert(to: outBuf, error: &error) { _, outStatus in
            if consumed {
                outStatus.pointee = .endOfStream
                return nil
            }
            consumed = true
            outStatus.pointee = .haveData
            return buffer
        }

        if error != nil {
            return false
        }

        guard let channelData = outBuf.floatChannelData?[0] else {
            return false
        }
        let count = Int(outBuf.frameLength)
        guard count > 0 else {
            return false
        }
        destination.append(contentsOf: UnsafeBufferPointer(start: channelData, count: count))
        return true
    }
}

import AVFoundation
import Accelerate

/// Computes frequency band magnitudes from audio buffers using FFT.
///
/// Pre-allocates all buffers at init time. Thread-safe via NSLock.
/// Bands are logarithmically spaced from 60 Hz to 8 kHz, centering human
/// speech frequencies (~300-3000 Hz) in the middle of the output array.
internal final class SpectrumAnalyzer: @unchecked Sendable {
    /// Number of output frequency bands.
    static let bandCount = AudioLevelMonitor.bandCount

    /// Floor of the dB range mapped to 0.0.
    private static let dbFloor: Float = -40
    /// Width of the dB range mapped to 0.0-1.0.
    private static let dbRange: Float = 40

    private static let minFreq: Float = 60
    private static let maxFreq: Float = 8_000

    private let fftSetup: FFTSetup
    private let log2n: vDSP_Length
    private let halfN: Int
    private let window: [Float]
    private let bandRanges: [(start: Int, end: Int)]

    // Mutable work buffers protected by lock.
    private let lock = NSLock()
    private var windowed: [Float]
    private var realp: [Float]
    private var imagp: [Float]
    private var magnitudes: [Float]
    private var bands: [Float]

    /// Creates a spectrum analyzer for a given frame count and sample rate.
    ///
    /// - Parameters:
    ///   - frameCount: Number of samples per audio buffer (must be power of 2).
    ///   - sampleRate: Audio sample rate in Hz.
    init(frameCount: Int, sampleRate: Float) {
        let log2n = vDSP_Length(log2(Float(frameCount)))
        self.log2n = log2n
        self.halfN = frameCount / 2
        guard let setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            fatalError("vDSP_create_fftsetup failed for log2n=\(log2n)")
        }
        self.fftSetup = setup

        // Pre-compute Hann window.
        var win = [Float](repeating: 0, count: frameCount)
        vDSP_hann_window(&win, vDSP_Length(frameCount), Int32(vDSP_HANN_NORM))
        self.window = win

        // Pre-allocate work buffers.
        self.windowed = [Float](repeating: 0, count: frameCount)
        self.realp = [Float](repeating: 0, count: frameCount / 2)
        self.imagp = [Float](repeating: 0, count: frameCount / 2)
        self.magnitudes = [Float](repeating: 0, count: frameCount / 2)
        self.bands = [Float](repeating: 0, count: Self.bandCount)

        // Compute half the bands from spectrum, then mirror for symmetric output.
        // Left half = low→speech center, right half = mirror.
        let halfBands = Self.bandCount / 2
        let binWidth = sampleRate / Float(frameCount)
        var ranges: [(start: Int, end: Int)] = []
        for idx in 0..<halfBands {
            let t0 = Float(idx) / Float(halfBands)
            let t1 = Float(idx + 1) / Float(halfBands)
            let f0 = Self.minFreq * pow(Self.maxFreq / Self.minFreq, t0)
            let f1 = Self.minFreq * pow(Self.maxFreq / Self.minFreq, t1)
            let bin0 = max(1, Int(f0 / binWidth))
            let bin1 = min(self.halfN - 2, Int(f1 / binWidth))
            ranges.append((start: bin0, end: max(bin0, bin1)))
        }
        self.bandRanges = ranges
    }

    // swiftlint:disable:next type_contents_order
    deinit {
        vDSP_destroy_fftsetup(fftSetup)
    }

    // MARK: - Analysis

    /// Compute frequency band magnitudes from an audio buffer.
    ///
    /// Returns an array of `bandCount` normalized levels (0.0-1.0).
    func analyze(_ buffer: AVAudioPCMBuffer) -> [Float] {
        guard let channelData = buffer.floatChannelData else {
            return [Float](repeating: 0, count: Self.bandCount)
        }
        let frames = Int(buffer.frameLength)
        guard frames > 0 else {
            return [Float](repeating: 0, count: Self.bandCount)
        }

        lock.lock()
        defer { lock.unlock() }

        computeFFT(channelData: channelData, frames: frames)
        mapBandsFromMagnitudes()
        mirrorBands()

        return bands
    }

    // MARK: - Private

    private func computeFFT(channelData: UnsafePointer<UnsafeMutablePointer<Float>>, frames: Int) {
        let sampleCount = min(frames, windowed.count)

        // Apply Hann window.
        vDSP_vmul(channelData[0], 1, window, 1, &windowed, 1, vDSP_Length(sampleCount))

        // Zero tail when fewer frames arrive to prevent stale FFT data.
        if sampleCount < windowed.count {
            for idx in sampleCount..<windowed.count {
                windowed[idx] = 0
            }
        }

        // Pack interleaved real data into split complex format.
        windowed.withUnsafeBufferPointer { buf in
            guard let base = buf.baseAddress else {
                return
            }
            base.withMemoryRebound(to: DSPComplex.self, capacity: halfN) { complexPtr in
                var split = DSPSplitComplex(realp: &realp, imagp: &imagp)
                vDSP_ctoz(complexPtr, 2, &split, 1, vDSP_Length(halfN))
            }
        }

        // In-place FFT.
        var split = DSPSplitComplex(realp: &realp, imagp: &imagp)
        vDSP_fft_zrip(fftSetup, &split, 1, log2n, FFTDirection(kFFTDirection_Forward))

        // Squared magnitudes, scaled by 1/(N^2) to normalize the FFT output.
        vDSP_zvmags(&split, 1, &magnitudes, 1, vDSP_Length(halfN))
        var scale = 1.0 / Float(sampleCount * sampleCount)
        vDSP_vsmul(magnitudes, 1, &scale, &magnitudes, 1, vDSP_Length(halfN))
    }

    private func mapBandsFromMagnitudes() {
        let halfBands = Self.bandCount / 2
        magnitudes.withUnsafeMutableBufferPointer { magPtr in
            guard let base = magPtr.baseAddress else {
                return
            }
            for idx in 0..<halfBands {
                let range = bandRanges[idx]
                let count = range.end - range.start + 1
                if count > 0 {
                    var sum: Float = 0
                    vDSP_sve(base + range.start, 1, &sum, vDSP_Length(count))
                    let avg = sum / Float(count)
                    let db = 10 * log10(max(avg, 1e-10))
                    bands[idx] = max(0, min(1, (db - Self.dbFloor) / Self.dbRange))
                } else {
                    bands[idx] = 0
                }
            }
        }
    }

    private func mirrorBands() {
        // Mirror: low freqs at edges, high freqs (speech) at center.
        // Copy half-band values first to avoid in-place read-after-write corruption.
        let halfBands = Self.bandCount / 2
        let halfValues = Array(bands[0..<halfBands])
        for idx in 0..<halfBands {
            let specVal = halfValues[halfBands - 1 - idx]
            bands[idx] = specVal  // left side
            bands[Self.bandCount - 1 - idx] = specVal  // right side (mirror)
        }
    }
}

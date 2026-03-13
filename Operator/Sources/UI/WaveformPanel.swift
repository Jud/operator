import AppKit
import SwiftUI
import os

/// Thread-safe frequency band levels shared between the audio capture thread and the UI.
///
/// The audio tap pushes spectrum band magnitudes; the WaveformModel reads them at 30 Hz.
public final class AudioLevelMonitor: @unchecked Sendable {
    static let bandCount = 40

    private let lock = OSAllocatedUnfairLock<[Float]>(
        initialState: [Float](repeating: 0, count: bandCount)
    )

    /// Creates a new audio level monitor.
    public init() {}

    /// Replace all band levels at once.
    ///
    /// Called from the audio tap thread with spectrum data.
    public func pushBands(_ bands: [Float]) {
        lock.withLock { state in
            for i in 0..<min(bands.count, state.count) {
                state[i] = bands[i]
            }
        }
    }

    /// Read current band levels into a pre-allocated buffer.
    ///
    /// Called from the main thread.
    public func readBands(into destination: inout [Float]) {
        lock.withLockUnchecked { state in
            for i in 0..<min(state.count, destination.count) {
                destination[i] = state[i]
            }
        }
    }

    /// Reset all levels to zero.
    public func reset() {
        lock.withLock { state in
            for i in 0..<state.count { state[i] = 0 }
        }
    }
}

/// Spectrum-driven waveform: each position = a frequency band, center = speech frequencies.
///
/// Reads band levels at 30 Hz, applies exponential smoothing, and alternates
/// above/below the center line. Speech energy naturally peaks in the center
/// bands (~300-3000 Hz), giving the tallest-in-center look without an artificial envelope.
@MainActor
@Observable
public final class WaveformModel {
    static let sampleCount = AudioLevelMonitor.bandCount
    private static let zeroed = Array(repeating: CGFloat.zero, count: sampleCount)
    private static let frameInterval: TimeInterval = 1.0 / 30.0

    /// Exponential smoothing factor (0 = no change, 1 = instant).
    private static let smoothingFactor: CGFloat = 0.45

    /// Normalized amplitudes for each sample point across the waveform width.
    var samples: [CGFloat] = zeroed

    private var animationTimer: Timer?
    private var isAnimating = false
    private var levelMonitor: AudioLevelMonitor?
    private var smoothedBands = [CGFloat](repeating: 0, count: sampleCount)
    private var rawBands = [Float](repeating: 0, count: sampleCount)

    func startAnimating(levelMonitor: AudioLevelMonitor?) {
        guard !isAnimating else {
            return
        }
        self.levelMonitor = levelMonitor
        isAnimating = true

        animationTimer = Timer.scheduledTimer(
            withTimeInterval: Self.frameInterval,
            repeats: true
        ) { [weak self] _ in
            Task { @MainActor in
                self?.updateSamples()
            }
        }
    }

    func stopAnimating() {
        isAnimating = false
        animationTimer?.invalidate()
        animationTimer = nil
        levelMonitor = nil
        smoothedBands = [CGFloat](repeating: 0, count: Self.sampleCount)
        samples = Self.zeroed
    }

    private func updateSamples() {
        guard isAnimating else {
            return
        }

        let count = Self.sampleCount

        if let monitor = levelMonitor {
            monitor.readBands(into: &rawBands)
        } else {
            for i in 0..<count { rawBands[i] = 0 }
        }

        for i in 0..<count {
            let raw = CGFloat(rawBands[i])
            smoothedBands[i] += (raw - smoothedBands[i]) * Self.smoothingFactor

            // Alternate above/below center line for waveform look.
            let sign: CGFloat = i.isMultiple(of: 2) ? 1 : -1
            samples[i] = smoothedBands[i] * sign
        }
    }
}

/// SwiftUI view that renders a thin animated waveform line indicator.
///
/// Driven by frequency band levels during recording. Rendered as a smooth
/// cubic Bezier path with round line caps for a refined appearance.
public struct WaveformView: View {
    var model: WaveformModel

    private let waveformWidth: CGFloat = 40
    private let waveformHeight: CGFloat = 18

    /// The waveform view rendering animated sample points.
    public var body: some View {
        Canvas { context, size in
            let midY = size.height / 2
            let amplitude = waveformHeight / 2
            let samples = model.samples
            guard samples.count >= 2 else {
                return
            }

            let step = size.width / CGFloat(samples.count - 1)
            var path = Path()
            var prevX = CGFloat.zero
            var prevY = midY - samples[0] * amplitude
            path.move(to: CGPoint(x: prevX, y: prevY))

            for i in 1..<samples.count {
                let x = CGFloat(i) * step
                let y = midY - samples[i] * amplitude
                let cpX = (prevX + x) / 2
                path.addCurve(
                    to: CGPoint(x: x, y: y),
                    control1: CGPoint(x: cpX, y: prevY),
                    control2: CGPoint(x: cpX, y: y)
                )
                prevX = x
                prevY = y
            }

            context.stroke(
                path,
                with: .color(.white),
                style: StrokeStyle(lineWidth: 1.5, lineCap: .round, lineJoin: .round)
            )
        }
        .frame(width: waveformWidth, height: waveformHeight)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(
            Capsule()
                .fill(Color.black.opacity(0.55))
        )
    }
}

/// Borderless, non-activating floating panel that displays an animated waveform
/// during push-to-talk voice capture.
///
/// - NSPanel with [.borderless, .nonactivatingPanel] style mask
/// - .floating level so it appears above all normal windows
/// - .canJoinAllSpaces + .fullScreenAuxiliary for visibility across Spaces and fullscreen
/// - Transparent background (isOpaque = false, backgroundColor = .clear)
/// - Positioned at top-center of screen, 50pt below top (below menu bar)
/// - Does not steal focus or activate when shown
@MainActor
public final class WaveformPanel: NSPanel {
    private static let logger = Log.logger(for: "WaveformPanel")
    private static let panelWidth: CGFloat = 56
    private static let panelHeight: CGFloat = 26

    private let waveformModel = WaveformModel()
    private let levelMonitor: AudioLevelMonitor?
    private var fadeOutWorkItem: DispatchWorkItem?

    /// Prevent the panel from ever becoming the key window.
    override public var canBecomeKey: Bool { false }

    /// Prevent the panel from ever becoming the main window.
    override public var canBecomeMain: Bool { false }

    /// Creates a new waveform panel positioned at top-center of the screen.
    public init(levelMonitor: AudioLevelMonitor? = nil) {
        self.levelMonitor = levelMonitor
        super.init(
            contentRect: NSRect(x: 0, y: 0, width: Self.panelWidth, height: Self.panelHeight),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )

        self.level = .floating
        self.isOpaque = false
        self.backgroundColor = .clear
        self.hidesOnDeactivate = false
        self.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        self.isMovableByWindowBackground = false

        let hostingView = NSHostingView(rootView: WaveformView(model: waveformModel))
        self.contentView = hostingView

        positionAtTopCenter()

        Self.logger.debug("WaveformPanel initialized")
    }

    /// Show the waveform panel and begin animation.
    ///
    /// Called when push-to-talk activates (trigger start).
    public func show() {
        cancelPendingFadeOut()

        self.alphaValue = 1.0
        positionAtTopCenter()
        waveformModel.startAnimating(levelMonitor: levelMonitor)
        orderFront(nil)

        Self.logger.debug("WaveformPanel shown")
    }

    /// Fade out the waveform panel over approximately 1 second.
    ///
    /// Called when the state machine returns to IDLE.
    public func fadeOut() {
        cancelPendingFadeOut()

        waveformModel.stopAnimating()

        let workItem = DispatchWorkItem { [weak self] in
            guard let self else {
                return
            }
            NSAnimationContext.runAnimationGroup(
                { context in
                    context.duration = 0.5
                    self.animator().alphaValue = 0.0
                },
                completionHandler: {
                    Task { @MainActor [weak self] in
                        self?.orderOut(nil)
                        self?.alphaValue = 1.0
                    }
                }
            )
        }
        fadeOutWorkItem = workItem
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5, execute: workItem)

        Self.logger.debug("WaveformPanel fade-out scheduled")
    }

    // MARK: - Private

    private func cancelPendingFadeOut() {
        fadeOutWorkItem?.cancel()
        fadeOutWorkItem = nil
    }

    private func positionAtTopCenter() {
        guard let screen = NSScreen.main else {
            return
        }

        let x = (screen.frame.width - Self.panelWidth) / 2
        let y = screen.frame.height - Self.panelHeight - 50

        self.setFrameOrigin(NSPoint(x: x, y: y))
    }
}

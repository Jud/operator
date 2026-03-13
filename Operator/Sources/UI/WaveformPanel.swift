import AppKit
import SwiftUI
import os

/// Thread-safe audio level ring buffer shared between the audio capture thread and the UI.
///
/// Written from the audio tap callback, read from the WaveformModel's display timer.
public final class AudioLevelMonitor: @unchecked Sendable {
    struct RingState: Sendable {
        private var buffer = [Float](repeating: 0, count: AudioLevelMonitor.bufferSize)
        private var writeIndex = 0

        mutating func push(_ value: Float) {
            buffer[writeIndex % buffer.count] = value
            writeIndex += 1
        }

        func levels() -> [Float] {
            let count = buffer.count
            var result = [Float](repeating: 0, count: count)
            for i in 0..<count {
                result[i] = buffer[(writeIndex + i) % count]
            }
            return result
        }

        mutating func reset() {
            buffer = [Float](repeating: 0, count: buffer.count)
            writeIndex = 0
        }
    }

    static let bufferSize = 40

    private let lock = OSAllocatedUnfairLock<RingState>(
        initialState: RingState()
    )

    /// Creates a new audio level monitor.
    public init() {}

    /// Push a new normalized audio level (0.0-1.0).
    ///
    /// Called from the audio tap thread.
    public func push(_ level: Float) {
        lock.withLock { $0.push(level) }
    }

    /// Read all levels in chronological order (oldest first).
    ///
    /// Called from the main thread.
    public func levels() -> [Float] {
        lock.withLock { $0.levels() }
    }

    /// Reset all levels to zero.
    public func reset() {
        lock.withLock { $0.reset() }
    }
}

/// Animated waveform data driven by real audio levels from an AudioLevelMonitor.
///
/// Reads audio levels on a 30 Hz timer, applies exponential smoothing for fluid
/// transitions, and modulates a traveling wave with the smoothed amplitude.
@MainActor
@Observable
public final class WaveformModel {
    static let sampleCount = 40
    private static let zeroed = Array(repeating: CGFloat.zero, count: sampleCount)
    private static let frameInterval: TimeInterval = 1.0 / 30.0

    /// Normalized amplitudes for each sample point across the waveform width.
    var samples: [CGFloat] = zeroed

    private var animationTimer: Timer?
    private var isAnimating = false
    private var startTime: CFTimeInterval = 0
    private var levelMonitor: AudioLevelMonitor?
    private var smoothedLevels = [CGFloat](repeating: 0, count: sampleCount)

    func startAnimating(levelMonitor: AudioLevelMonitor?) {
        guard !isAnimating else {
            return
        }
        self.levelMonitor = levelMonitor
        isAnimating = true
        startTime = CACurrentMediaTime()

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
        samples = Self.zeroed
        smoothedLevels = [CGFloat](repeating: 0, count: Self.sampleCount)
    }

    private func updateSamples() {
        guard isAnimating else {
            return
        }

        let elapsed = CACurrentMediaTime() - startTime
        let count = Self.sampleCount
        let rawLevels = levelMonitor?.levels() ?? [Float](repeating: 0, count: count)

        for i in 0..<count {
            let x = Double(i) / Double(count - 1)

            // Map ring buffer position to this sample.
            let levelIndex = min(i, rawLevels.count - 1)
            let rawLevel = CGFloat(rawLevels[max(0, levelIndex)])
            let gatedLevel = rawLevel < 0.03 ? CGFloat(0) : rawLevel

            // Exponential smoothing for fluid transitions.
            smoothedLevels[i] += (gatedLevel - smoothedLevels[i]) * 0.25
            let level = smoothedLevels[i]

            // Taper amplitude toward edges.
            let envelope = sin(x * .pi)

            // Traveling wave provides organic oscillation.
            let wave = sin(elapsed * 2.0 + x * 2.5 * .pi)

            // Gentle idle ripple when no audio is present.
            let idleRipple = 0.04 * sin(elapsed * 1.2 + x * .pi)

            let amplitude = level > 0.02 ? Double(level) : idleRipple
            samples[i] = CGFloat(amplitude * wave * envelope)
        }
    }
}

/// SwiftUI view that renders a thin animated waveform line indicator.
///
/// Driven by real audio levels during recording, with a gentle idle ripple
/// during processing states. Rendered as a smooth cubic Bezier path with
/// round line caps for a refined appearance.
public struct WaveformView: View {
    var model: WaveformModel

    private let waveformWidth: CGFloat = 36
    private let waveformHeight: CGFloat = 12

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
    private static let panelWidth: CGFloat = 52
    private static let panelHeight: CGFloat = 20

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

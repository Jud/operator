import AppKit
import SwiftUI

/// Animated waveform line data driven by a timer during the listening state.
///
/// Generates sample points that combine multiple sine waves at staggered
/// frequencies to produce a natural, flowing waveform line.
@MainActor
@Observable
public final class WaveformModel {
    static let sampleCount = 40
    private static let zeroed = Array(repeating: CGFloat.zero, count: sampleCount)
    private static let frameInterval: TimeInterval = 1.0 / 30.0
    private static let sampleStep = 1.0 / Double(sampleCount - 1)

    /// Normalized amplitudes (0…1) for each sample point across the waveform width.
    var samples: [CGFloat] = zeroed

    private var animationTimer: Timer?
    private var isAnimating = false
    private var startTime: CFTimeInterval = 0

    func startAnimating() {
        guard !isAnimating else {
            return
        }
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
        samples = Self.zeroed
    }

    private func updateSamples() {
        guard isAnimating else {
            return
        }

        let elapsed = CACurrentMediaTime() - startTime
        let count = Self.sampleCount

        for i in 0..<count {
            let x = Double(i) * Self.sampleStep

            // Blend several travelling sine waves for organic movement
            let wave1 = sin(elapsed * 2.6 + x * 4.0 * .pi)
            let wave2 = 0.6 * sin(elapsed * 3.4 + x * 6.0 * .pi + 1.0)
            let wave3 = 0.3 * sin(elapsed * 1.8 + x * 2.0 * .pi + 2.5)

            // Envelope: taper amplitude toward the edges
            let envelope = sin(x * .pi)

            let combined = (wave1 + wave2 + wave3) / 1.9  // normalize roughly to -1…1
            samples[i] = CGFloat(combined * envelope)
        }
    }
}

/// SwiftUI view that renders an animated line waveform indicator.
///
/// The waveform animates during the listening state to provide visual feedback
/// that Operator is capturing audio. Rendered with a translucent dark
/// background capsule for visibility against any screen content.
public struct WaveformView: View {
    var model: WaveformModel

    private let waveformWidth: CGFloat = 32
    private let waveformHeight: CGFloat = 22

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

            context.stroke(path, with: .color(.white), lineWidth: 2)
        }
        .frame(width: waveformWidth, height: waveformHeight)
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(
            Capsule()
                .fill(Color.black.opacity(0.6))
        )
    }
}

/// Borderless, non-activating floating panel that displays an animated waveform
/// during push-to-talk voice capture.
///
/// Configured per technical-spec.md Component 8:
/// - NSPanel with [.borderless, .nonactivatingPanel] style mask
/// - .floating level so it appears above all normal windows
/// - .canJoinAllSpaces + .fullScreenAuxiliary for visibility across Spaces and fullscreen
/// - Transparent background (isOpaque = false, backgroundColor = .clear)
/// - Positioned at top-center of screen, 50pt below top (below menu bar)
/// - Does not steal focus or activate when shown
///
/// The panel appears when push-to-talk starts and fades out approximately
/// 1 second after returning to IDLE state.
@MainActor
public final class WaveformPanel: NSPanel {
    private static let logger = Log.logger(for: "WaveformPanel")
    private static let panelWidth: CGFloat = 60
    private static let panelHeight: CGFloat = 30

    private let waveformModel = WaveformModel()
    private var fadeOutWorkItem: DispatchWorkItem?

    /// Prevent the panel from ever becoming the key window.
    override public var canBecomeKey: Bool { false }

    /// Prevent the panel from ever becoming the main window.
    override public var canBecomeMain: Bool { false }

    /// Creates a new waveform panel positioned at bottom-center of the screen.
    public init() {
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

    /// Show the waveform panel and begin bar animation.
    ///
    /// Called when push-to-talk activates (trigger start).
    public func show() {
        cancelPendingFadeOut()

        self.alphaValue = 1.0
        positionAtTopCenter()
        waveformModel.startAnimating()
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

import AppKit
import SwiftUI

/// Animated waveform bar data driven by a timer during the listening state.
///
/// Each bar oscillates at a slightly different frequency to produce a natural waveform effect.
@MainActor
public final class WaveformModel: ObservableObject {
    @Published var barHeights: [CGFloat] = Array(repeating: 0.15, count: 5)

    private var animationTimer: Timer?
    private var isAnimating = false
    private var startTime: CFTimeInterval = 0

    /// Base oscillation frequencies (Hz) per bar -- staggered for organic movement.
    private let frequencies: [Double] = [1.3, 1.7, 2.1, 1.5, 1.9]

    func startAnimating() {
        guard !isAnimating else {
            return
        }
        isAnimating = true
        startTime = CACurrentMediaTime()

        animationTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 30.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.updateBars()
            }
        }
    }

    func stopAnimating() {
        isAnimating = false
        animationTimer?.invalidate()
        animationTimer = nil
        barHeights = Array(repeating: 0.15, count: 5)
    }

    private func updateBars() {
        guard isAnimating else {
            return
        }

        let elapsed = CACurrentMediaTime() - startTime

        for i in 0..<barHeights.count {
            let wave = sin(elapsed * frequencies[i] * 2.0 * .pi)
            barHeights[i] = 0.15 + 0.85 * CGFloat((wave + 1.0) / 2.0)
        }
    }
}

/// SwiftUI view that renders animated vertical bars as a waveform indicator.
///
/// The bars animate during the listening state to provide visual feedback
/// that Operator is capturing audio. Rendered with a translucent dark
/// background capsule for visibility against any screen content.
public struct WaveformView: View {
    @ObservedObject var model: WaveformModel

    private let barWidth: CGFloat = 4
    private let barSpacing: CGFloat = 3
    private let maxBarHeight: CGFloat = 22

    /// The view body rendering animated waveform bars.
    public var body: some View {
        HStack(spacing: barSpacing) {
            ForEach(0..<model.barHeights.count, id: \.self) { index in
                RoundedRectangle(cornerRadius: barWidth / 2)
                    .fill(Color.white)
                    .frame(
                        width: barWidth,
                        height: max(barWidth, maxBarHeight * model.barHeights[index])
                    )
                    .animation(.easeInOut(duration: 0.08), value: model.barHeights[index])
            }
        }
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
/// - Positioned at bottom-center of screen, 80pt above bottom (above dock)
/// - Does not steal focus or activate when shown
///
/// The panel appears when push-to-talk starts and fades out approximately
/// 1 second after returning to IDLE state.
@MainActor
public final class WaveformPanel: NSPanel {
    private static let logger = Log.logger(for: "WaveformPanel")
    private let waveformModel = WaveformModel()
    private let panelWidth: CGFloat = 60
    private let panelHeight: CGFloat = 30
    private var fadeOutTask: Task<Void, Never>?

    /// Prevent the panel from ever becoming the key window.
    override public var canBecomeKey: Bool { false }

    /// Prevent the panel from ever becoming the main window.
    override public var canBecomeMain: Bool { false }

    /// Creates a new waveform panel positioned at bottom-center of the screen.
    public init() {
        super.init(
            contentRect: NSRect(x: 0, y: 0, width: 60, height: 30),
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

        positionAtBottomCenter()

        Self.logger.debug("WaveformPanel initialized")
    }

    /// Show the waveform panel and begin bar animation.
    ///
    /// Called when push-to-talk activates (trigger start).
    public func show() {
        cancelPendingFadeOut()

        self.alphaValue = 1.0
        positionAtBottomCenter()
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

        fadeOutTask = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .milliseconds(500))
            guard !Task.isCancelled, let self else {
                return
            }

            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                NSAnimationContext.runAnimationGroup(
                    { context in
                        context.duration = 0.5
                        self.animator().alphaValue = 0.0
                    },
                    completionHandler: {
                        continuation.resume()
                    }
                )
            }

            guard !Task.isCancelled else {
                return
            }
            self.orderOut(nil)
            self.alphaValue = 1.0
        }

        Self.logger.debug("WaveformPanel fade-out scheduled")
    }

    // MARK: - Private

    private func cancelPendingFadeOut() {
        fadeOutTask?.cancel()
        fadeOutTask = nil
    }

    private func positionAtBottomCenter() {
        guard let screen = NSScreen.main else {
            return
        }

        let x = (screen.frame.width - panelWidth) / 2
        let y: CGFloat = 80

        self.setFrameOrigin(NSPoint(x: x, y: y))
    }
}

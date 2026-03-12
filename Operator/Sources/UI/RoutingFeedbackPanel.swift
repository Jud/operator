import AppKit
import SwiftUI

/// Observable model backing the routing feedback HUD.
@MainActor
@Observable
public final class RoutingFeedbackModel {
    /// The routing trace currently being displayed.
    var trace: RoutingTrace?

    /// Session names available for correction.
    var sessionNames: [String] = []

    /// Whether the correction picker is showing.
    var showingCorrection = false

    /// Callback invoked when the user provides feedback.
    var onFeedback: ((UUID, RoutingTrace.Annotation) -> Void)?
}

/// SwiftUI view for the routing feedback HUD content.
private struct RoutingFeedbackView: View {
    @Bindable var model: RoutingFeedbackModel

    private var panelShape: RoundedRectangle { RoundedRectangle(cornerRadius: 10) }

    var body: some View {
        if let trace = model.trace {
            traceContent(trace)
        }
    }

    @ViewBuilder
    private func traceContent(_ trace: RoutingTrace) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            headerRow(trace)
            transcribedTextRow(trace)
            resultRow(trace)
            scoresRow(trace)
            correctionPicker(trace)
        }
        .padding(10)
        .frame(width: 280)
        .background(
            VisualEffectBlur(material: .hudWindow, blendingMode: .behindWindow)
        )
        .clipShape(panelShape)
        .overlay(
            panelShape.stroke(Color.primary.opacity(0.1), lineWidth: 0.5)
        )
    }

    @ViewBuilder
    private func headerRow(_ trace: RoutingTrace) -> some View {
        HStack(spacing: 6) {
            stepBadge(trace.resolvedBy)
            Spacer()
            if !model.showingCorrection {
                Button(
                    action: {
                        submitFeedback(trace, correct: true)
                    },
                    label: {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(.green)
                    }
                )
                .buttonStyle(.plain)
                Button(
                    action: {
                        model.showingCorrection = true
                    },
                    label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundColor(.red)
                    }
                )
                .buttonStyle(.plain)
            }
        }
    }

    @ViewBuilder
    private func transcribedTextRow(_ trace: RoutingTrace) -> some View {
        Text("\"\(trace.transcribedText)\"")
            .font(.system(size: 11, design: .monospaced))
            .foregroundColor(.primary)
            .lineLimit(2)
    }

    @ViewBuilder
    private func resultRow(_ trace: RoutingTrace) -> some View {
        HStack(spacing: 4) {
            Image(systemName: "arrow.right.circle")
                .font(.system(size: 10))
            Text(resultLabel(trace.result))
                .font(.system(size: 11, weight: .medium))
        }
        .foregroundColor(.secondary)
    }

    @ViewBuilder
    private func scoresRow(_ trace: RoutingTrace) -> some View {
        if !trace.heuristicScores.isEmpty {
            let sortedScores = trace.heuristicScores.sorted { $0.score > $1.score }
            let routedSession = routedSessionName(from: trace)
            HStack(spacing: 8) {
                ForEach(sortedScores, id: \.session) { score in
                    Text("\(score.session):\(score.score)")
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundColor(
                            score.session == routedSession ? .primary : .secondary
                        )
                }
            }
        }
    }

    @ViewBuilder
    private func correctionPicker(_ trace: RoutingTrace) -> some View {
        if model.showingCorrection {
            VStack(alignment: .leading, spacing: 4) {
                Text("Should have been:")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundColor(.secondary)
                ForEach(model.sessionNames, id: \.self) { name in
                    Button(
                        action: {
                            submitFeedback(trace, correct: false, correctedSession: name)
                        },
                        label: {
                            Text(name)
                                .font(.system(size: 11, design: .monospaced))
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(.vertical, 2)
                                .padding(.horizontal, 6)
                                .background(Color.accentColor.opacity(0.1))
                                .clipShape(RoundedRectangle(cornerRadius: 4))
                        }
                    )
                    .buttonStyle(.plain)
                }
            }
        }
    }

    private func submitFeedback(_ trace: RoutingTrace, correct: Bool, correctedSession: String? = nil) {
        let annotation = RoutingTrace.Annotation(correct: correct, correctedSession: correctedSession)
        model.onFeedback?(trace.id, annotation)
        model.trace = nil
        model.showingCorrection = false
    }

    @ViewBuilder
    private func stepBadge(_ step: RoutingTrace.RoutingStep) -> some View {
        let (label, color) = stepStyle(step)
        Text(label)
            .font(.system(size: 10, weight: .semibold, design: .monospaced))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.2))
            .foregroundColor(color)
            .clipShape(RoundedRectangle(cornerRadius: 4))
    }

    private func stepStyle(_ step: RoutingTrace.RoutingStep) -> (label: String, color: Color) {
        switch step {
        case .operatorCommand: ("CMD", .purple)
        case .keywordExtraction: ("KEYWORD", .blue)
        case .singleSessionBypass: ("SINGLE", .green)
        case .sessionAffinity: ("AFFINITY", .orange)
        case .heuristicScoring: ("HEURISTIC", .teal)
        case .engineRouting: ("ENGINE", .indigo)
        case .clarification: ("CLARIFY", .yellow)
        case .noSessions: ("NONE", .red)
        case .emptyTranscription: ("EMPTY", .gray)
        case .error: ("ERROR", .red)
        }
    }

    private func resultLabel(_ result: RoutingTrace.RoutingTraceResult) -> String {
        switch result {
        case .routed(let session, _): session
        case .clarify(let candidates): "clarify: \(candidates.joined(separator: ", "))"
        case .operatorCommand(let cmd): "cmd: \(cmd)"
        case .noSessions: "no sessions"
        case .notConfident: "not confident"
        case .cliNotFound: "CLI not found"
        case .error(let state, let seconds): "timeout: \(state) (\(Int(seconds))s)"
        }
    }

    private func routedSessionName(from trace: RoutingTrace) -> String? {
        if case .routed(let session, _) = trace.result {
            return session
        }
        return nil
    }
}

/// NSVisualEffectView wrapper for SwiftUI.
private struct VisualEffectBlur: NSViewRepresentable {
    let material: NSVisualEffectView.Material
    let blendingMode: NSVisualEffectView.BlendingMode

    func makeNSView(context: Context) -> NSVisualEffectView {
        let view = NSVisualEffectView()
        view.material = material
        view.blendingMode = blendingMode
        view.state = .active
        return view
    }

    func updateNSView(_ nsView: NSVisualEffectView, context: Context) {
        nsView.material = material
        nsView.blendingMode = blendingMode
    }
}

/// Floating, non-activating panel that shows routing decisions with feedback buttons.
///
/// Positioned at top-left of the screen. Appears over iTerm (even with secure keyboard
/// entry) because it uses `.screenSaver` window level. Auto-dismisses after 5 seconds
/// unless the user interacts with it.
@MainActor
public final class RoutingFeedbackPanel: NSPanel {
    private static let logger = Log.logger(for: "RoutingFeedbackPanel")
    private let feedbackModel = RoutingFeedbackModel()
    private var autoDismissWorkItem: DispatchWorkItem?

    override public var canBecomeKey: Bool { false }
    override public var canBecomeMain: Bool { false }

    /// Creates a new routing feedback panel backed by the given trace store.
    public init(traceStore: RoutingTraceStore) {
        super.init(
            contentRect: NSRect(x: 0, y: 0, width: 280, height: 120),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )

        self.level = .screenSaver
        self.isOpaque = false
        self.backgroundColor = .clear
        self.hidesOnDeactivate = false
        self.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        self.isMovableByWindowBackground = true
        self.hasShadow = true

        let hostingView = NSHostingView(rootView: RoutingFeedbackView(model: feedbackModel))
        self.contentView = hostingView

        feedbackModel.onFeedback = { [weak self] traceId, annotation in
            Task {
                await traceStore.annotate(traceId: traceId, annotation: annotation)
            }
            self?.dismiss()
        }

        positionTopLeft()

        Self.logger.debug("RoutingFeedbackPanel initialized")
    }

    /// Show the panel with a new routing trace.
    public func show(trace: RoutingTrace, sessionNames: [String]) {
        cancelAutoDismiss()

        feedbackModel.trace = trace
        feedbackModel.sessionNames = sessionNames
        feedbackModel.showingCorrection = false

        // Use a fixed width; SwiftUI sizes vertically to fit.
        self.setContentSize(NSSize(width: 280, height: 120))
        positionTopLeft()
        self.alphaValue = 1.0
        orderFront(nil)

        // Re-fit after SwiftUI has laid out the new content
        DispatchQueue.main.async { [weak self] in
            guard let self, let contentView = self.contentView else {
                return
            }
            let fitting = contentView.fittingSize
            if fitting.width > 0, fitting.height > 0 {
                self.setContentSize(fitting)
                self.positionTopLeft()
            }
        }

        // Auto-dismiss after 5 seconds
        let workItem = DispatchWorkItem { [weak self] in
            self?.fadeOut()
        }
        autoDismissWorkItem = workItem
        DispatchQueue.main.asyncAfter(deadline: .now() + 5, execute: workItem)

        Self.logger.debug("Showing routing feedback for trace \(trace.id)")
    }

    /// Dismiss the panel immediately.
    public func dismiss() {
        cancelAutoDismiss()
        feedbackModel.trace = nil
        orderOut(nil)
    }

    private func fadeOut() {
        NSAnimationContext.runAnimationGroup(
            { context in
                context.duration = 0.3
                self.animator().alphaValue = 0.0
            },
            completionHandler: {
                Task { @MainActor [weak self] in
                    self?.orderOut(nil)
                    self?.feedbackModel.trace = nil
                }
            }
        )
    }

    private func cancelAutoDismiss() {
        autoDismissWorkItem?.cancel()
        autoDismissWorkItem = nil
    }

    private func positionTopLeft() {
        guard let screen = NSScreen.main else {
            return
        }
        let screenFrame = screen.visibleFrame
        let x = screenFrame.minX + 20
        let y = screenFrame.maxY - self.frame.height - 10
        self.setFrameOrigin(NSPoint(x: x, y: y))
    }
}

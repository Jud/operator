import AppKit
import CoreGraphics

/// Push-to-talk trigger using the FN/Globe key via CGEventTap.
///
/// Supports toggle mode (quick tap, <300ms), push-to-talk mode (long press, >=300ms),
/// double-tap (second tap within 300ms of toggle-mode entry), and Escape to cancel.
/// FN generates kCGEventFlagsChanged events; detection uses the maskSecondaryFn flag.
/// A configurable secondary hotkey supports external keyboards without a usable FN key.
public final class FNKeyTrigger: TriggerSource {
    private static let logger = Log.logger(for: "FNKeyTrigger")

    /// Escape key code on macOS.
    private static let escapeKeyCode: CGKeyCode = 53

    /// Callback invoked when the push-to-talk key is pressed.
    public var onStart: (@Sendable @MainActor () -> Void)?
    /// Callback invoked when the push-to-talk key is released.
    public var onStop: (@Sendable @MainActor () -> Void)?
    /// Callback invoked when Escape is pressed to cancel recording.
    public var onCancel: (@Sendable @MainActor () -> Void)?
    /// Callback invoked when the user double-taps the push-to-talk key within 300ms.
    public var onDoubleTap: (@Sendable @MainActor () -> Void)?

    /// Duration threshold distinguishing toggle mode from push-to-talk mode.
    private let toggleThreshold: TimeInterval

    /// Tracks whether FN is currently held down to detect edges (press/release).
    private var fnDown = false

    /// Timestamp of the most recent FN key-down event for duration measurement.
    private var fnKeyDownTime: Date?

    /// Whether we are in toggle-listening mode (first quick tap started listening).
    private var isToggleListening = false

    /// Secondary hotkey key code (nil disables secondary hotkey).
    private var secondaryKeyCode: CGKeyCode?
    /// Required modifier flags for the secondary hotkey.
    private var secondaryModifiers: CGEventFlags
    /// Tracks whether the secondary hotkey is currently held down.
    private var secondaryDown = false
    /// Timestamp of the most recent secondary key-down event.
    private var secondaryKeyDownTime: Date?
    /// Whether we are in toggle-listening mode via the secondary hotkey.
    private var isSecondaryToggleListening = false

    /// Timer for the 300ms double-tap detection window after toggle-mode entry.
    private var doubleTapTimer: Timer?

    /// Retained references to prevent the tap and run loop source from being deallocated.
    private var eventTap: CFMachPort?
    private var runLoopSource: CFRunLoopSource?
    private var tapWatchdogTimer: Timer?

    /// Whether either trigger source is currently in an active listening state.
    private var isListening: Bool {
        fnDown || isToggleListening || secondaryDown || isSecondaryToggleListening
    }

    /// Creates a new FNKeyTrigger.
    ///
    /// - Parameters:
    ///   - secondaryKeyCode: Optional CGKeyCode for a secondary global hotkey.
    ///   - secondaryModifiers: Modifier flags required for the secondary hotkey. Defaults to none.
    ///   - toggleThreshold: Duration in seconds below which a press is treated as a toggle tap.
    public init(
        secondaryKeyCode: CGKeyCode? = nil,
        secondaryModifiers: CGEventFlags = [],
        toggleThreshold: TimeInterval = 0.3
    ) {
        self.secondaryKeyCode = secondaryKeyCode
        self.secondaryModifiers = secondaryModifiers
        self.toggleThreshold = toggleThreshold
    }

    /// Set up the CGEventTap for FN key monitoring.
    ///
    /// This must be called after the app has launched and the run loop is running.
    public func setupEventTap() {
        guard AXIsProcessTrusted() else {
            Self.logger.error("Accessibility permission not granted, cannot create CGEventTap")
            promptForAccessibilityPermission()
            return
        }

        // Always listen for flagsChanged (FN key) and key events (Escape + secondary hotkey).
        var mask: CGEventMask = (1 << CGEventType.flagsChanged.rawValue)
        mask |= (1 << CGEventType.keyDown.rawValue)
        mask |= (1 << CGEventType.keyUp.rawValue)

        guard
            let tap = CGEvent.tapCreate(
                tap: .cgSessionEventTap,
                place: .headInsertEventTap,
                options: .listenOnly,
                eventsOfInterest: mask,
                callback: { _, type, event, refcon -> Unmanaged<CGEvent>? in
                    guard let refcon else {
                        return Unmanaged.passUnretained(event)
                    }
                    let trigger = Unmanaged<FNKeyTrigger>.fromOpaque(refcon).takeUnretainedValue()
                    trigger.handleEvent(type: type, event: event)
                    return Unmanaged.passUnretained(event)
                },
                userInfo: Unmanaged.passUnretained(self).toOpaque()
            )
        else {
            Self.logger.error("CGEvent.tapCreate failed — Accessibility permission may not be granted")
            promptForAccessibilityPermission()
            return
        }

        self.eventTap = tap

        let source = CFMachPortCreateRunLoopSource(nil, tap, 0)

        self.runLoopSource = source
        CFRunLoopAddSource(CFRunLoopGetMain(), source, .commonModes)
        CGEvent.tapEnable(tap: tap, enable: true)
        startTapWatchdog()

        Self.logger.info("CGEventTap installed for FN key monitoring")
        if let keyCode = secondaryKeyCode {
            Self.logger.info("Secondary hotkey configured: keyCode=\(keyCode)")
        }
    }

    /// Periodically re-enable the event tap in case macOS disables it due to timeout.
    private func startTapWatchdog() {
        tapWatchdogTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { [weak self] _ in
            guard let self, let tap = self.eventTap else {
                return
            }
            if !CGEvent.tapIsEnabled(tap: tap) {
                CGEvent.tapEnable(tap: tap, enable: true)
                Self.logger.warning("Re-enabled event tap (was disabled by system)")
            }
        }
    }

    /// Dispatch a trigger callback on the main queue.
    private func dispatch(_ callback: (@Sendable @MainActor () -> Void)?) {
        DispatchQueue.main.async { callback?() }
    }

    deinit {
        doubleTapTimer?.invalidate()
        tapWatchdogTimer?.invalidate()
        if let source = runLoopSource {
            CFRunLoopRemoveSource(CFRunLoopGetMain(), source, .commonModes)
        }
        if let tap = eventTap {
            CGEvent.tapEnable(tap: tap, enable: false)
        }
    }
}

// MARK: - Event Handling

extension FNKeyTrigger {
    /// Process a single CGEvent from the event tap callback.
    private func handleEvent(type: CGEventType, event: CGEvent) {
        switch type {
        case .flagsChanged:
            handleFlagsChanged(event: event)

        case .keyDown:
            let keyCode = CGKeyCode(event.getIntegerValueField(.keyboardEventKeycode))
            if keyCode == Self.escapeKeyCode {
                handleEscapeKey()
            } else {
                handleSecondaryKeyDown(event: event)
            }

        case .keyUp:
            handleSecondaryKeyUp(event: event)

        case .tapDisabledByTimeout:
            if let tap = eventTap {
                CGEvent.tapEnable(tap: tap, enable: true)
                Self.logger.warning("Event tap was disabled by timeout, re-enabled")
            }

        default:
            break
        }
    }

    /// Handle Escape key press to cancel recording.
    private func handleEscapeKey() {
        guard isListening else {
            return
        }

        Self.logger.debug("Escape pressed while listening — cancelling")

        doubleTapTimer?.invalidate()
        doubleTapTimer = nil
        isToggleListening = false
        isSecondaryToggleListening = false
        fnKeyDownTime = nil
        secondaryKeyDownTime = nil
        dispatch(onCancel)
    }
}

// MARK: - FN Key Handling

extension FNKeyTrigger {
    /// Whether toggle mode is enabled via user preferences.
    private var isToggleModeEnabled: Bool {
        UserDefaults.standard.bool(forKey: "toggleModeEnabled")
    }

    /// Handle kCGEventFlagsChanged for FN/Globe key detection.
    private func handleFlagsChanged(event: CGEvent) {
        let flags = event.flags
        let fnPressed = flags.contains(.maskSecondaryFn)

        if fnPressed && !fnDown {
            handleFnKeyDown()
        } else if !fnPressed && fnDown {
            handleFnKeyUp()
        }
    }

    /// Process the FN key-down edge: toggle off, double-tap, or start listening.
    private func handleFnKeyDown() {
        fnDown = true

        if isToggleListening, doubleTapTimer != nil {
            Self.logger.debug("FN key pressed (double-tap detected)")
            doubleTapTimer?.invalidate()
            doubleTapTimer = nil
            isToggleListening = false
            fnKeyDownTime = nil
            dispatch(onDoubleTap)
        } else if isToggleListening {
            Self.logger.debug("FN key pressed (toggle off)")
            isToggleListening = false
            fnKeyDownTime = nil
            dispatch(onStop)
        } else {
            Self.logger.debug("FN key pressed")
            fnKeyDownTime = Date()
            dispatch(onStart)
        }
    }

    /// Process the FN key-up edge: enter toggle mode or complete push-to-talk.
    private func handleFnKeyUp() {
        fnDown = false

        let pressDuration: TimeInterval
        if let downTime = fnKeyDownTime {
            pressDuration = Date().timeIntervalSince(downTime)
        } else {
            pressDuration = .infinity
        }
        fnKeyDownTime = nil

        if isToggleListening {
            Self.logger.debug("FN key released (toggle stop press — ignored)")
        } else if isToggleModeEnabled && pressDuration < toggleThreshold {
            Self.logger.debug("FN key released (quick tap — toggle mode)")
            isToggleListening = true
            startDoubleTapTimer()
        } else {
            Self.logger.debug("FN key released (push-to-talk)")
            dispatch(onStop)
        }
    }

    /// Start the 300ms double-tap detection window after entering toggle-listening mode.
    private func startDoubleTapTimer() {
        doubleTapTimer?.invalidate()
        doubleTapTimer = Timer.scheduledTimer(
            withTimeInterval: toggleThreshold,
            repeats: false
        ) { [weak self] _ in
            self?.doubleTapTimer = nil
        }
    }
}

// MARK: - Secondary Hotkey Handling

extension FNKeyTrigger {
    /// Handle kCGEventKeyDown for secondary hotkey press detection.
    private func handleSecondaryKeyDown(event: CGEvent) {
        guard let keyCode = secondaryKeyCode else {
            return
        }

        let eventKeyCode = CGKeyCode(event.getIntegerValueField(.keyboardEventKeycode))
        guard eventKeyCode == keyCode else {
            return
        }

        if !secondaryModifiers.isEmpty {
            let currentModifiers = event.flags.intersection(
                [.maskShift, .maskControl, .maskAlternate, .maskCommand]
            )
            let requiredModifiers = secondaryModifiers.intersection(
                [.maskShift, .maskControl, .maskAlternate, .maskCommand]
            )
            guard currentModifiers == requiredModifiers else {
                return
            }
        }

        if isSecondaryToggleListening {
            if !secondaryDown {
                secondaryDown = true
                Self.logger.debug("Secondary hotkey pressed (toggle off): keyCode=\(keyCode)")
                isSecondaryToggleListening = false
                secondaryKeyDownTime = nil
                dispatch(onStop)
            }
        } else if !secondaryDown {
            secondaryDown = true
            Self.logger.debug("Secondary hotkey pressed: keyCode=\(keyCode)")
            secondaryKeyDownTime = Date()
            dispatch(onStart)
        }
    }

    /// Handle kCGEventKeyUp for secondary hotkey release detection.
    private func handleSecondaryKeyUp(event: CGEvent) {
        guard let keyCode = secondaryKeyCode else {
            return
        }

        let eventKeyCode = CGKeyCode(event.getIntegerValueField(.keyboardEventKeycode))
        guard eventKeyCode == keyCode else {
            return
        }

        if secondaryDown {
            secondaryDown = false

            let pressDuration: TimeInterval
            if let downTime = secondaryKeyDownTime {
                pressDuration = Date().timeIntervalSince(downTime)
            } else {
                pressDuration = .infinity
            }
            secondaryKeyDownTime = nil

            if isSecondaryToggleListening {
                Self.logger.debug(
                    "Secondary hotkey released (toggle stop press — ignored): keyCode=\(keyCode)"
                )
            } else if isToggleModeEnabled && pressDuration < toggleThreshold {
                Self.logger.debug("Secondary hotkey released (quick tap — toggle mode): keyCode=\(keyCode)")
                isSecondaryToggleListening = true
            } else {
                Self.logger.debug("Secondary hotkey released (push-to-talk): keyCode=\(keyCode)")
                dispatch(onStop)
            }
        }
    }
}

// MARK: - Permissions

extension FNKeyTrigger {
    /// Prompt the user for Accessibility permission with a spoken error and open System Settings.
    private func promptForAccessibilityPermission() {
        Self.logger.info("Opening System Settings for Accessibility permission")
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility") {
            NSWorkspace.shared.open(url)
        }
    }
}

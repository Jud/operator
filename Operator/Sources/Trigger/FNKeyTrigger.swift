import AppKit
import CoreGraphics

/// Push-to-talk trigger using the FN/Globe key via CGEventTap.
///
/// Supports two trigger modes based on press duration:
/// - **Toggle mode** (quick tap, <300ms): First tap starts listening, second tap stops.
/// - **Push-to-talk mode** (long press, >=300ms): Hold to listen, release to stop.
///
/// Pressing Escape while listening cancels the recording without processing.
///
/// FN is a modifier key on macOS -- it generates kCGEventFlagsChanged events,
/// not kCGEventKeyDown/kCGEventKeyUp. Detection uses the maskSecondaryFn flag.
/// The event tap requires Accessibility permission; if AXIsProcessTrusted()
/// returns false, the user is prompted with a spoken error and System Settings
/// opens to the Accessibility pane.
///
/// A configurable secondary hotkey is also supported for external keyboards
/// that may not have a usable FN key. The secondary hotkey is specified as a
/// CGKeyCode and optional modifier flags, and uses kCGEventKeyDown/kCGEventKeyUp.
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

    /// Duration threshold distinguishing toggle mode from push-to-talk mode.
    private let toggleThreshold: TimeInterval

    /// Tracks whether FN is currently held down to detect edges (press/release).
    private var fnDown = false

    /// Timestamp of the most recent FN key-down event for duration measurement.
    private var fnKeyDownTime: Date?

    /// Whether we are in toggle-listening mode (first quick tap started listening).
    private var isToggleListening = false

    /// Secondary hotkey key code (e.g., CGKeyCode for right Option).
    ///
    /// Nil disables secondary hotkey.
    private var secondaryKeyCode: CGKeyCode?

    /// Required modifier flags for the secondary hotkey (e.g., [.maskControl, .maskShift]).
    private var secondaryModifiers: CGEventFlags

    /// Tracks whether the secondary hotkey is currently held down.
    private var secondaryDown = false

    /// Timestamp of the most recent secondary key-down event for duration measurement.
    private var secondaryKeyDownTime: Date?

    /// Whether we are in toggle-listening mode via the secondary hotkey.
    private var isSecondaryToggleListening = false

    /// Retained references to prevent the tap and run loop source from being deallocated.
    private var eventTap: CFMachPort?
    private var runLoopSource: CFRunLoopSource?

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
                options: .defaultTap,
                eventsOfInterest: mask,
                callback: { _, type, event, refcon -> Unmanaged<CGEvent>? in
                    guard let refcon else {
                        return Unmanaged.passRetained(event)
                    }
                    let trigger = Unmanaged<FNKeyTrigger>.fromOpaque(refcon).takeUnretainedValue()
                    trigger.handleEvent(type: type, event: event)
                    return Unmanaged.passRetained(event)
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

        Self.logger.info("CGEventTap installed for FN key monitoring")
        if let keyCode = secondaryKeyCode {
            Self.logger.info("Secondary hotkey configured: keyCode=\(keyCode)")
        }
    }

    deinit {
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

        isToggleListening = false
        isSecondaryToggleListening = false
        fnKeyDownTime = nil
        secondaryKeyDownTime = nil

        let callback = onCancel

        DispatchQueue.main.async {
            callback?()
        }
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

    /// Process the FN key-down edge: toggle off or start listening.
    private func handleFnKeyDown() {
        fnDown = true

        if isToggleListening {
            Self.logger.debug("FN key pressed (toggle off)")
            isToggleListening = false
            fnKeyDownTime = nil

            let callback = onStop

            DispatchQueue.main.async {
                callback?()
            }
        } else {
            Self.logger.debug("FN key pressed")
            fnKeyDownTime = Date()

            let callback = onStart

            DispatchQueue.main.async {
                callback?()
            }
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
        } else {
            Self.logger.debug("FN key released (push-to-talk)")

            let callback = onStop

            DispatchQueue.main.async {
                callback?()
            }
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

                let callback = onStop

                DispatchQueue.main.async {
                    callback?()
                }
            }
        } else if !secondaryDown {
            secondaryDown = true
            Self.logger.debug("Secondary hotkey pressed: keyCode=\(keyCode)")
            secondaryKeyDownTime = Date()

            let callback = onStart

            DispatchQueue.main.async {
                callback?()
            }
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

                let callback = onStop

                DispatchQueue.main.async {
                    callback?()
                }
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

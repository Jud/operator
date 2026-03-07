import AppKit
import CoreGraphics
import OSLog

/// Push-to-talk trigger using the FN/Globe key via CGEventTap.
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
    private static let logger = Logger(subsystem: "com.operator.app", category: "FNKeyTrigger")

    public var onStart: (@Sendable @MainActor () -> Void)?
    public var onStop: (@Sendable @MainActor () -> Void)?

    /// Tracks whether FN is currently held down to detect edges (press/release).
    private var fnDown = false

    /// Secondary hotkey key code (e.g., CGKeyCode for right Option). Nil disables secondary hotkey.
    private var secondaryKeyCode: CGKeyCode?

    /// Required modifier flags for the secondary hotkey (e.g., [.maskControl, .maskShift]).
    private var secondaryModifiers: CGEventFlags

    /// Tracks whether the secondary hotkey is currently held down.
    private var secondaryDown = false

    /// Retained references to prevent the tap and run loop source from being deallocated.
    private var eventTap: CFMachPort?
    private var runLoopSource: CFRunLoopSource?

    /// Creates a new FNKeyTrigger.
    ///
    /// - Parameters:
    ///   - secondaryKeyCode: Optional CGKeyCode for a secondary global hotkey.
    ///   - secondaryModifiers: Modifier flags required for the secondary hotkey. Defaults to none.
    public init(secondaryKeyCode: CGKeyCode? = nil, secondaryModifiers: CGEventFlags = []) {
        self.secondaryKeyCode = secondaryKeyCode
        self.secondaryModifiers = secondaryModifiers
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

        var mask: CGEventMask = (1 << CGEventType.flagsChanged.rawValue)
        if secondaryKeyCode != nil {
            mask |= (1 << CGEventType.keyDown.rawValue)
            mask |= (1 << CGEventType.keyUp.rawValue)
        }

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

    /// Process a single CGEvent from the event tap callback.
    private func handleEvent(type: CGEventType, event: CGEvent) {
        switch type {
        case .flagsChanged:
            handleFlagsChanged(event: event)

        case .keyDown:
            handleSecondaryKeyDown(event: event)

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

    /// Handle kCGEventFlagsChanged for FN/Globe key detection.
    private func handleFlagsChanged(event: CGEvent) {
        let flags = event.flags
        let fnPressed = flags.contains(.maskSecondaryFn)

        if fnPressed && !fnDown {
            fnDown = true
            Self.logger.debug("FN key pressed")

            let callback = onStart

            DispatchQueue.main.async {
                callback?()
            }
        } else if !fnPressed && fnDown {
            fnDown = false
            Self.logger.debug("FN key released")

            let callback = onStop

            DispatchQueue.main.async {
                callback?()
            }
        }
    }

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

        if !secondaryDown {
            secondaryDown = true
            Self.logger.debug("Secondary hotkey pressed: keyCode=\(keyCode)")

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
            Self.logger.debug("Secondary hotkey released: keyCode=\(keyCode)")

            let callback = onStop

            DispatchQueue.main.async {
                callback?()
            }
        }
    }

    /// Prompt the user for Accessibility permission with a spoken error and open System Settings.
    private func promptForAccessibilityPermission() {
        Self.logger.info("Opening System Settings for Accessibility permission")
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility") {
            NSWorkspace.shared.open(url)
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

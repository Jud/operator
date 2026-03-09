import AppKit
import ApplicationServices

/// Result of querying the current accessibility context.
///
/// Captures the frontmost application, whether it is a supported terminal,
/// whether the focused UI element is a text input, and whether accessibility
/// permission is granted. All fields are determined at query time.
public struct AccessibilityContext: Sendable {
    /// The bundle identifier of the frontmost application, if available.
    public let frontmostBundleID: String?
    /// Whether the frontmost app is a supported terminal (iTerm2, Ghostty).
    public let isTerminal: Bool
    /// Whether the focused UI element is a text input field.
    public let isTextField: Bool
    /// Whether accessibility permission is granted for UI element queries.
    public let hasPermission: Bool

    /// Creates a new accessibility context with the given values.
    public init(
        frontmostBundleID: String?,
        isTerminal: Bool,
        isTextField: Bool,
        hasPermission: Bool
    ) {
        self.frontmostBundleID = frontmostBundleID
        self.isTerminal = isTerminal
        self.isTextField = isTextField
        self.hasPermission = hasPermission
    }
}

/// Protocol for accessibility context queries.
///
/// Abstracts frontmost app detection and focused element inspection so
/// the bimodal decision engine can be tested with mock implementations.
public protocol AccessibilityQuerying: Sendable {
    /// Query the current accessibility context: frontmost app, terminal status,
    /// text field status, and permission state.
    func queryContext() -> AccessibilityContext
}

/// Queries macOS accessibility state for frontmost app detection,
/// terminal identification by bundle ID, and text field detection via AXRole.
///
/// Uses `NSWorkspace.shared.frontmostApplication` for the frontmost app and
/// `AXUIElementCreateSystemWide()` with `kAXFocusedUIElementAttribute` for
/// focused element inspection. All queries are synchronous C API calls that
/// complete within single-digit milliseconds.
public struct AccessibilityQueryService: AccessibilityQuerying {
    private static let logger = Log.logger(for: "AccessibilityQueryService")

    /// Bundle identifiers recognized as supported terminals.
    static let terminalBundleIDs: Set<String> = [
        "com.googlecode.iterm2",
        "com.mitchellh.ghostty"
    ]

    /// AXRole values recognized as text input fields.
    static let textFieldRoles: Set<String> = [
        "AXTextField",
        "AXTextArea",
        "AXWebArea",
        "AXComboBox",
        "AXSearchField"
    ]

    /// Creates a new accessibility query service.
    public init() {}

    /// Query the current accessibility context.
    ///
    /// Determines the frontmost application, whether it is a supported terminal,
    /// whether the focused UI element is a text input, and whether accessibility
    /// permission is granted.
    public func queryContext() -> AccessibilityContext {
        let bundleID = NSWorkspace.shared.frontmostApplication?.bundleIdentifier
        let isTerminal = bundleID.map { Self.terminalBundleIDs.contains($0) } ?? false

        let (isTextField, hasPermission) = queryFocusedElement()

        let bid = bundleID ?? "nil"
        Self.logger.debug(
            "Context: bundle=\(bid) terminal=\(isTerminal) textField=\(isTextField) perm=\(hasPermission)"
        )

        return AccessibilityContext(
            frontmostBundleID: bundleID,
            isTerminal: isTerminal,
            isTextField: isTextField,
            hasPermission: hasPermission
        )
    }

    /// Query the focused UI element's AXRole to determine if it is a text field.
    ///
    /// - Returns: A tuple of (isTextField, hasPermission). If accessibility permission
    ///   is not granted, returns (false, false).
    private func queryFocusedElement() -> (isTextField: Bool, hasPermission: Bool) {
        let systemWide = AXUIElementCreateSystemWide()
        var focusedValue: CFTypeRef?
        let result = AXUIElementCopyAttributeValue(
            systemWide,
            kAXFocusedUIElementAttribute as CFString,
            &focusedValue
        )

        if result == .apiDisabled {
            Self.logger.warning("Accessibility permission not granted")
            return (isTextField: false, hasPermission: false)
        }

        guard result == .success, let focused = focusedValue else {
            return (isTextField: false, hasPermission: true)
        }

        let axElement: AXUIElement = unsafeDowncast(focused, to: AXUIElement.self)
        var roleValue: CFTypeRef?
        let roleResult = AXUIElementCopyAttributeValue(
            axElement,
            kAXRoleAttribute as CFString,
            &roleValue
        )

        guard roleResult == .success, let role = roleValue as? String else {
            return (isTextField: false, hasPermission: true)
        }

        return (isTextField: Self.textFieldRoles.contains(role), hasPermission: true)
    }
}

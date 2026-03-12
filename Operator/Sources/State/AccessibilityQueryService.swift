import AppKit
import ApplicationServices
@preconcurrency import HarnessCore

/// Result of querying the current accessibility context.
///
/// Captures the frontmost application, whether it is a supported terminal,
/// whether the focused UI element is a text input, and whether accessibility
/// permission is granted. All fields are determined at query time.
///
/// When HarnessCore is available, also includes cursor context from the focused
/// element: selected text, text before/after cursor, and the element's AX role.
public struct AccessibilityContext: Sendable {
    /// The bundle identifier of the frontmost application, if available.
    public let frontmostBundleID: String?
    /// Whether the frontmost app is a supported terminal (iTerm2, Ghostty).
    public let isTerminal: Bool
    /// Whether the focused UI element is a text input field.
    public let isTextField: Bool
    /// Whether accessibility permission is granted for UI element queries.
    public let hasPermission: Bool
    /// The AX role of the focused element (e.g. "AXTextField", "AXTextArea").
    public let focusedRole: String?
    /// Text currently selected in the focused element, if any.
    public let selectedText: String?
    /// Text before the cursor/insertion point in the focused element.
    public let textBeforeCursor: String?
    /// Text after the cursor/insertion point in the focused element.
    public let textAfterCursor: String?

    /// Creates a new accessibility context with the given values.
    public init(
        frontmostBundleID: String?,
        isTerminal: Bool,
        isTextField: Bool,
        hasPermission: Bool,
        focusedRole: String? = nil,
        selectedText: String? = nil,
        textBeforeCursor: String? = nil,
        textAfterCursor: String? = nil
    ) {
        self.frontmostBundleID = frontmostBundleID
        self.isTerminal = isTerminal
        self.isTextField = isTextField
        self.hasPermission = hasPermission
        self.focusedRole = focusedRole
        self.selectedText = selectedText
        self.textBeforeCursor = textBeforeCursor
        self.textAfterCursor = textAfterCursor
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
/// Uses HarnessCore's `focusedElement()` for rich cursor context (selected text,
/// text before/after cursor, settable attributes) and falls back to raw AX C API
/// calls if the harness query fails.
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

    nonisolated(unsafe) private let harness: HarnessSession

    /// Creates a new accessibility query service.
    public init(harness: HarnessSession = HarnessSession(enableBackgroundPerception: false)) {
        self.harness = harness
    }

    /// Query the current accessibility context.
    ///
    /// Uses HarnessCore to query the focused element for rich cursor context.
    /// Falls back to raw AX calls if the harness query fails.
    public func queryContext() -> AccessibilityContext {
        let bundleID = NSWorkspace.shared.frontmostApplication?.bundleIdentifier
        let isTerminal = bundleID.map(Self.terminalBundleIDs.contains) ?? false
        let bid = bundleID ?? "nil"

        // Try HarnessCore for rich context
        do {
            let focused = try harness.focusedElement()
            let isTextField = Self.textFieldRoles.contains(focused.role)

            Self.logger.debug(
                "Context: bundle=\(bid) terminal=\(isTerminal) textField=\(isTextField) role=\(focused.role)"
            )

            return AccessibilityContext(
                frontmostBundleID: bundleID,
                isTerminal: isTerminal,
                isTextField: isTextField,
                hasPermission: true,
                focusedRole: focused.role,
                selectedText: focused.selectedText,
                textBeforeCursor: focused.textBeforeCursor,
                textAfterCursor: focused.textAfterCursor
            )
        } catch {
            Self.logger.debug("HarnessCore query failed, falling back to raw AX: \(error.localizedDescription)")
        }

        // Fallback: raw AX C API
        let (isTextField, hasPermission) = queryFocusedElementFallback()

        Self.logger.debug(
            "Context (fallback): bundle=\(bid) terminal=\(isTerminal) textField=\(isTextField) perm=\(hasPermission)"
        )

        return AccessibilityContext(
            frontmostBundleID: bundleID,
            isTerminal: isTerminal,
            isTextField: isTextField,
            hasPermission: hasPermission
        )
    }

    /// Fallback: query the focused UI element's AXRole via raw C API.
    private func queryFocusedElementFallback() -> (isTextField: Bool, hasPermission: Bool) {
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

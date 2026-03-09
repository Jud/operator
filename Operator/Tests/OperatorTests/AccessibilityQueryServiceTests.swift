import Foundation
import Testing

@testable import OperatorCore

/// Mock AccessibilityQuerying that returns preconfigured context for testing.
internal struct MockAccessibilityQuery: AccessibilityQuerying {
    var context: AccessibilityContext

    init(
        bundleID: String? = "com.example.app",
        isTerminal: Bool = false,
        isTextField: Bool = true,
        hasPermission: Bool = true
    ) {
        self.context = AccessibilityContext(
            frontmostBundleID: bundleID,
            isTerminal: isTerminal,
            isTextField: isTextField,
            hasPermission: hasPermission
        )
    }

    func queryContext() -> AccessibilityContext {
        context
    }
}

/// Namespace for AccessibilityQueryService tests.
private enum AccessibilityQueryServiceTests {}

@Suite("AccessibilityQueryService - Terminal Detection")
internal struct AccessibilityTerminalDetectionTests {
    @Test("iTerm2 bundle ID is recognized as terminal")
    func itermRecognized() {
        #expect(AccessibilityQueryService.terminalBundleIDs.contains("com.googlecode.iterm2"))
    }

    @Test("Ghostty bundle ID is recognized as terminal")
    func ghosttyRecognized() {
        #expect(AccessibilityQueryService.terminalBundleIDs.contains("com.mitchellh.ghostty"))
    }

    @Test("unknown bundle ID is not recognized as terminal")
    func unknownNotRecognized() {
        #expect(!AccessibilityQueryService.terminalBundleIDs.contains("com.apple.Safari"))
        #expect(!AccessibilityQueryService.terminalBundleIDs.contains("com.microsoft.VSCode"))
        #expect(!AccessibilityQueryService.terminalBundleIDs.contains("com.apple.Terminal"))
    }
}

@Suite("AccessibilityQueryService - Text Field Detection")
internal struct AccessibilityTextFieldDetectionTests {
    @Test("AXTextField is recognized as text field")
    func textFieldRecognized() {
        #expect(AccessibilityQueryService.textFieldRoles.contains("AXTextField"))
    }

    @Test("AXTextArea is recognized as text field")
    func textAreaRecognized() {
        #expect(AccessibilityQueryService.textFieldRoles.contains("AXTextArea"))
    }

    @Test("AXWebArea is recognized as text field")
    func webAreaRecognized() {
        #expect(AccessibilityQueryService.textFieldRoles.contains("AXWebArea"))
    }

    @Test("AXComboBox is recognized as text field")
    func comboBoxRecognized() {
        #expect(AccessibilityQueryService.textFieldRoles.contains("AXComboBox"))
    }

    @Test("AXSearchField is recognized as text field")
    func searchFieldRecognized() {
        #expect(AccessibilityQueryService.textFieldRoles.contains("AXSearchField"))
    }

    @Test("non-text AXRoles are not recognized as text fields")
    func nonTextRolesNotRecognized() {
        #expect(!AccessibilityQueryService.textFieldRoles.contains("AXButton"))
        #expect(!AccessibilityQueryService.textFieldRoles.contains("AXImage"))
        #expect(!AccessibilityQueryService.textFieldRoles.contains("AXGroup"))
        #expect(!AccessibilityQueryService.textFieldRoles.contains("AXWindow"))
        #expect(!AccessibilityQueryService.textFieldRoles.contains("AXStaticText"))
    }
}

@Suite("AccessibilityQuerying - Mock Protocol Isolation")
internal struct AccessibilityMockProtocolTests {
    @Test("mock returns configured terminal context")
    func mockTerminalContext() {
        let mock = MockAccessibilityQuery(
            bundleID: "com.googlecode.iterm2",
            isTerminal: true,
            isTextField: false
        )
        let ctx = mock.queryContext()
        #expect(ctx.frontmostBundleID == "com.googlecode.iterm2")
        #expect(ctx.isTerminal == true)
        #expect(ctx.isTextField == false)
        #expect(ctx.hasPermission == true)
    }

    @Test("mock returns configured non-terminal text field context")
    func mockNonTerminalTextField() {
        let mock = MockAccessibilityQuery(
            bundleID: "com.apple.TextEdit",
            isTerminal: false,
            isTextField: true
        )
        let ctx = mock.queryContext()
        #expect(ctx.frontmostBundleID == "com.apple.TextEdit")
        #expect(ctx.isTerminal == false)
        #expect(ctx.isTextField == true)
    }

    @Test("mock returns permission denied context")
    func mockPermissionDenied() {
        let mock = MockAccessibilityQuery(hasPermission: false)
        let ctx = mock.queryContext()
        #expect(ctx.hasPermission == false)
    }

    @Test("mock returns no text field context")
    func mockNoTextField() {
        let mock = MockAccessibilityQuery(
            bundleID: "com.apple.finder",
            isTerminal: false,
            isTextField: false
        )
        let ctx = mock.queryContext()
        #expect(ctx.isTextField == false)
        #expect(ctx.isTerminal == false)
    }
}

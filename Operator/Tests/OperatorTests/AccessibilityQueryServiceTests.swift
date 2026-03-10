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

internal enum AccessibilityQueryServiceTests {
    @Suite("AccessibilityQueryService - Terminal Detection")
    internal struct TerminalDetectionTests {
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
    internal struct TextFieldDetectionTests {
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
}

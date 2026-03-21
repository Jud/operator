import AppKit
import CoreGraphics

/// Result of a dictation delivery attempt.
public enum DictationResult: Sendable {
    /// Text was successfully inserted at the cursor.
    case success
    /// No text field is focused in the frontmost application.
    case noTextField
    /// No previous dictation exists for replay.
    case noLastDictation
    /// The synthetic paste failed.
    case pasteFailed(String)
}

/// Protocol for dictation text delivery via pasteboard insertion.
///
/// Implementations insert text at the current cursor position using a
/// pasteboard save/replace/paste/restore cycle. The last successfully
/// dictated text is stored for replay via double-tap.
@MainActor
public protocol DictationDelivering: AnyObject, Sendable {
    /// The most recently dictated text, available for replay.
    var lastDictation: String? { get }

    /// Insert text at the current cursor position.
    ///
    /// Performs a pasteboard save/replace/paste/restore cycle:
    /// 1. Save all current pasteboard contents (types and raw data).
    /// 2. Replace pasteboard with the dictated text.
    /// 3. Send synthetic Cmd+V to paste.
    /// 4. Brief delay for paste to register.
    /// 5. Restore original pasteboard contents (best-effort).
    ///
    /// On success, stores the text as `lastDictation` for replay.
    ///
    /// - Parameter text: The transcribed text to insert.
    /// - Returns: The result of the delivery attempt.
    func deliver(_ text: String) async -> DictationResult

    /// Replay the last successfully dictated text at the cursor.
    ///
    /// Uses the same pasteboard cycle as `deliver(_:)` with the stored
    /// `lastDictation` text. Returns `.noLastDictation` if no previous
    /// dictation exists.
    ///
    /// - Returns: The result of the replay attempt.
    func replayLast() async -> DictationResult
}

/// Delivers transcribed text at the cursor via pasteboard-based insertion.
///
/// Uses a save/replace/paste/restore cycle on `NSPasteboard.general` to insert
/// text without permanently altering the user's clipboard. The restore step is
/// best-effort: if it fails, the delivery is still considered successful.
///
/// Synthetic Cmd+V is sent via CGEvent, which requires Accessibility permission
/// (already granted for the FN key event tap).
@MainActor
public final class DictationDelivery: DictationDelivering {
    private static let logger = Log.logger(for: "DictationDelivery")

    /// Virtual key code for the V key on macOS (kVK_ANSI_V).
    private static let vKeyCode: CGKeyCode = 9

    /// Delay after synthetic paste before restoring the pasteboard.
    private static let pasteDelayNanoseconds: UInt64 = 50_000_000  // 50ms

    /// The most recently dictated text, available for replay.
    public private(set) var lastDictation: String?

    /// Creates a new dictation delivery instance.
    public init() {}

    /// Insert text at the current cursor position via pasteboard cycle.
    public func deliver(_ text: String) async -> DictationResult {
        Self.logger.info("Delivering dictation (\(text.count) chars)")

        // Always store as lastDictation so double-tap replay works
        // even if paste fails (e.g., noTextField). The transcription
        // was successful — delivery failure is a separate concern.
        lastDictation = text

        let result = await performPasteboardCycle(text: text)
        if case .success = result {
            Self.logger.info("Dictation delivered successfully")
        }
        return result
    }

    /// Replay the last dictation at the cursor.
    public func replayLast() async -> DictationResult {
        guard let text = lastDictation else {
            Self.logger.warning("No lastDictation available for replay")
            return .noLastDictation
        }

        Self.logger.info("Replaying last dictation (\(text.count) chars)")
        return await performPasteboardCycle(text: text)
    }
}

// MARK: - Pasteboard Cycle

extension DictationDelivery {
    /// Saved representation of a single pasteboard item's type/data pairs.
    private struct SavedPasteboardItem {
        let entries: [(type: NSPasteboard.PasteboardType, data: Data)]
    }

    /// Perform the full pasteboard save/replace/paste/restore cycle.
    private func performPasteboardCycle(text: String) async -> DictationResult {
        let pasteboard = NSPasteboard.general

        // 1. Save current pasteboard contents.
        let savedItems = savePasteboardContents(pasteboard)

        // 2. Replace pasteboard with dictation text.
        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)

        // 3. Send synthetic Cmd+V.
        guard sendSyntheticPaste() else {
            // Restore pasteboard before returning failure.
            restorePasteboardContents(savedItems, to: pasteboard)
            return .pasteFailed("Failed to create or post synthetic Cmd+V event")
        }

        // 4. Brief delay for paste to register.
        try? await Task.sleep(nanoseconds: Self.pasteDelayNanoseconds)

        // 5. Restore original pasteboard contents (best-effort).
        restorePasteboardContents(savedItems, to: pasteboard)

        return .success
    }

    /// Read all types and data from every item on the pasteboard.
    private func savePasteboardContents(_ pasteboard: NSPasteboard) -> [SavedPasteboardItem] {
        guard let items = pasteboard.pasteboardItems else {
            Self.logger.debug("No pasteboard items to save")
            return []
        }

        var saved: [SavedPasteboardItem] = []
        for item in items {
            var entries: [(type: NSPasteboard.PasteboardType, data: Data)] = []
            for type in item.types {
                if let data = item.data(forType: type) {
                    entries.append((type: type, data: data))
                }
            }
            if !entries.isEmpty {
                saved.append(SavedPasteboardItem(entries: entries))
            }
        }

        Self.logger.debug("Saved \(saved.count) pasteboard item(s)")
        return saved
    }

    /// Write previously saved items back to the pasteboard.
    ///
    /// Best-effort: errors are logged but do not cause delivery failure.
    private func restorePasteboardContents(_ savedItems: [SavedPasteboardItem], to pasteboard: NSPasteboard) {
        guard !savedItems.isEmpty else {
            Self.logger.debug("No pasteboard items to restore (was empty)")
            return
        }

        pasteboard.clearContents()

        var pasteboardItems: [NSPasteboardItem] = []
        for saved in savedItems {
            let item = NSPasteboardItem()
            for entry in saved.entries {
                item.setData(entry.data, forType: entry.type)
            }
            pasteboardItems.append(item)
        }

        let success = pasteboard.writeObjects(pasteboardItems)
        if success {
            Self.logger.debug("Restored \(pasteboardItems.count) pasteboard item(s)")
        } else {
            Self.logger.warning("Failed to restore pasteboard contents (best-effort, continuing)")
        }
    }
}

// MARK: - Synthetic Keystroke

extension DictationDelivery {
    /// Send a synthetic Cmd+V keystroke via CGEvent.
    ///
    /// - Returns: `true` if the events were created and posted successfully.
    private func sendSyntheticPaste() -> Bool {
        guard
            let keyDown = CGEvent(
                keyboardEventSource: nil,
                virtualKey: Self.vKeyCode,
                keyDown: true
            )
        else {
            Self.logger.error("Failed to create CGEvent for Cmd+V key-down")
            return false
        }

        guard
            let keyUp = CGEvent(
                keyboardEventSource: nil,
                virtualKey: Self.vKeyCode,
                keyDown: false
            )
        else {
            Self.logger.error("Failed to create CGEvent for Cmd+V key-up")
            return false
        }

        keyDown.flags = .maskCommand
        keyUp.flags = .maskCommand

        keyDown.post(tap: .cghidEventTap)
        keyUp.post(tap: .cghidEventTap)

        Self.logger.debug("Posted synthetic Cmd+V keystroke")
        return true
    }
}

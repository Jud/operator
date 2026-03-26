import Foundation
import Testing

@testable import OperatorCore

/// Mock DictationDelivering that records calls and returns preconfigured results.
///
/// Used by StateMachine integration tests and BimodalDecisionEngine tests
/// to verify dictation delivery is called correctly without depending on
/// NSPasteboard or CGEvent.
@MainActor
internal final class MockDictationDelivery: DictationDelivering {
    var lastDictation: String?
    var deliverResult: DictationResult = .success
    var deliverCallCount = 0
    var replayCallCount = 0
    var lastDeliveredText: String?

    func deliver(_ text: String) async -> DictationResult {
        deliverCallCount += 1
        lastDeliveredText = text
        if case .success = deliverResult {
            lastDictation = text
        }
        return deliverResult
    }

    var storeForReplayCallCount = 0

    func storeForReplay(_ text: String) {
        storeForReplayCallCount += 1
        lastDictation = text
    }

    func replayLast() async -> DictationResult {
        replayCallCount += 1
        guard lastDictation != nil else {
            return .noLastDictation
        }
        return deliverResult
    }
}

internal enum DictationDeliveryTests {
    @Suite("DictationDelivery - Contract Behavior")
    @MainActor
    internal struct ContractTests {
        @Test("replayLast returns noLastDictation when no previous dictation exists")
        func replayWithNoPriorDictation() async {
            let delivery = DictationDelivery()
            let result = await delivery.replayLast()
            guard case .noLastDictation = result else {
                Issue.record("Expected .noLastDictation, got \(result)")
                return
            }
        }

        @Test("lastDictation is nil initially")
        func initialLastDictationIsNil() {
            let delivery = DictationDelivery()
            #expect(delivery.lastDictation == nil)
        }

        @Test("storeForReplay sets lastDictation without delivering")
        func storeForReplaySetsLastDictation() {
            let delivery = DictationDelivery()
            delivery.storeForReplay("hello world")
            #expect(delivery.lastDictation == "hello world")
        }
    }
}

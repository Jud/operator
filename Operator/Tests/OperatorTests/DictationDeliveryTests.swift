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

    func replayLast() async -> DictationResult {
        replayCallCount += 1
        guard lastDictation != nil else {
            return .noLastDictation
        }
        return deliverResult
    }
}

/// Namespace for DictationDelivery tests.
private enum DictationDeliveryTests {}

@Suite("DictationDelivery - Contract Behavior")
@MainActor
internal struct DictationDeliveryContractTests {
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
}

@Suite("MockDictationDelivery - Protocol Conformance")
@MainActor
internal struct MockDictationDeliveryTests {
    @Test("deliver stores lastDictation on success")
    func deliverStoresLastDictation() async {
        let mock = MockDictationDelivery()
        mock.deliverResult = .success

        let result = await mock.deliver("hello world")
        guard case .success = result else {
            Issue.record("Expected .success, got \(result)")
            return
        }
        #expect(mock.lastDictation == "hello world")
        #expect(mock.deliverCallCount == 1)
    }

    @Test("deliver does not store lastDictation on failure")
    func deliverDoesNotStoreOnFailure() async {
        let mock = MockDictationDelivery()
        mock.deliverResult = .pasteFailed("test error")

        _ = await mock.deliver("hello world")
        #expect(mock.lastDictation == nil)
    }

    @Test("replayLast returns noLastDictation when empty")
    func replayLastWhenEmpty() async {
        let mock = MockDictationDelivery()
        let result = await mock.replayLast()
        guard case .noLastDictation = result else {
            Issue.record("Expected .noLastDictation, got \(result)")
            return
        }
        #expect(mock.replayCallCount == 1)
    }

    @Test("replayLast uses stored lastDictation text")
    func replayLastUsesStoredText() async {
        let mock = MockDictationDelivery()
        _ = await mock.deliver("meeting at 3pm")
        let result = await mock.replayLast()
        guard case .success = result else {
            Issue.record("Expected .success, got \(result)")
            return
        }
        #expect(mock.lastDictation == "meeting at 3pm")
    }

    @Test("sequential deliver then replay preserves text")
    func sequentialDeliverThenReplay() async {
        let mock = MockDictationDelivery()

        _ = await mock.deliver("first message")
        #expect(mock.lastDictation == "first message")

        _ = await mock.deliver("second message")
        #expect(mock.lastDictation == "second message")

        let result = await mock.replayLast()
        guard case .success = result else {
            Issue.record("Expected .success, got \(result)")
            return
        }
        #expect(mock.lastDictation == "second message")
    }
}

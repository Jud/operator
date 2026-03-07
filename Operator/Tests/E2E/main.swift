import Foundation
import OperatorCore

/// E2E test runner entry point.
///
/// Creates a real iTerm window with 2 tabs, boots the Operator daemon with
/// mock voice I/O and real routing/delivery, runs 10 test scenarios, prints
/// a pass/fail summary, and exits with code 0 if all pass or 1 if any fail.
///
/// Signal handling ensures the test iTerm window is closed on ctrl-C.

@MainActor
internal func runE2ETests() async -> Int32 {
    let harness = E2EHarness()

    // Install signal handler to ensure teardown on ctrl-C
    let signalSource = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
    signal(SIGINT, SIG_IGN)  // Ignore default handler so DispatchSource receives it
    signalSource.setEventHandler {
        print("\n[e2e] Caught SIGINT, cleaning up...")
        harness.tearDown()
        exit(1)
    }
    signalSource.resume()

    // Setup
    do {
        try await harness.setUp()
    } catch {
        print("[e2e] SETUP FAILED: \(error)")
        harness.tearDown()
        return 1
    }

    // Run all scenarios
    print("\n========== E2E Test Run ==========\n")
    let results = await harness.runAllScenarios()

    // Print summary
    print("\n========== Results ==========\n")
    var passCount = 0
    var failCount = 0

    for result in results {
        let icon = result.passed ? "PASS" : "FAIL"
        print("  [\(icon)] \(result.name)")
        if !result.passed {
            print("         \(result.detail)")
        }
        if result.passed { passCount += 1 } else { failCount += 1 }
    }

    print("\n  \(passCount) passed, \(failCount) failed out of \(results.count) scenarios")
    print("\n==================================\n")

    // Teardown
    harness.tearDown()

    signalSource.cancel()

    return failCount == 0 ? 0 : 1
}

// Entry point -- run on MainActor via async main

internal let exitCode: Int32 = await runE2ETests()

exit(exitCode)

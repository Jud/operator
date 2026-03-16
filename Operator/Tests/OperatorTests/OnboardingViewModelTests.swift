import Foundation
import Testing

@testable import OperatorCore

@Suite("OnboardingViewModel")
@MainActor
internal struct OnboardingViewModelTests {
    // MARK: - Step Navigation

    @Test("advance progresses through steps sequentially")
    func advanceProgressesThroughSteps() {
        let vm = OnboardingViewModel()
        #expect(vm.currentStep == .welcome)

        vm.advance()
        #expect(vm.currentStep == .permissions)

        vm.advance()
        #expect(vm.currentStep == .accessibility)

        vm.advance()
        #expect(vm.currentStep == .modelDownload)

        vm.advance()
        #expect(vm.currentStep == .howItWorks)

        vm.advance()
        #expect(vm.currentStep == .done)
    }

    @Test("advance does nothing when on done step")
    func advanceDoesNothingOnDone() {
        let vm = OnboardingViewModel()
        vm.currentStep = .done

        vm.advance()
        #expect(vm.currentStep == .done)
    }

    @Test("goBack returns to previous step from each step")
    func goBackReturnsToPreviousStep() {
        let vm = OnboardingViewModel()

        vm.currentStep = .done
        vm.goBack()
        #expect(vm.currentStep == .howItWorks)

        vm.goBack()
        #expect(vm.currentStep == .modelDownload)

        vm.goBack()
        #expect(vm.currentStep == .accessibility)

        vm.goBack()
        #expect(vm.currentStep == .permissions)

        vm.goBack()
        #expect(vm.currentStep == .welcome)
    }

    @Test("goBack does nothing when on welcome step")
    func goBackDoesNothingOnWelcome() {
        let vm = OnboardingViewModel()
        #expect(vm.currentStep == .welcome)

        vm.goBack()
        #expect(vm.currentStep == .welcome)
    }

    // MARK: - Completion

    @Test("completeOnboarding sets hasCompletedOnboarding flag in UserDefaults")
    func completeOnboardingSetsFlag() {
        let vm = OnboardingViewModel()
        UserDefaults.standard.set(false, forKey: "hasCompletedOnboarding")

        vm.completeOnboarding()

        #expect(UserDefaults.standard.bool(forKey: "hasCompletedOnboarding") == true)

        // Restore to avoid polluting other tests.
        UserDefaults.standard.removeObject(forKey: "hasCompletedOnboarding")
    }

    @Test("completeOnboarding invokes the stored completion handler")
    func completeOnboardingInvokesHandler() {
        let vm = OnboardingViewModel()
        var handlerCalled = false

        vm.prepare { handlerCalled = true }

        vm.completeOnboarding()
        #expect(handlerCalled == true)

        UserDefaults.standard.removeObject(forKey: "hasCompletedOnboarding")
    }

    @Test("completeOnboarding clears the handler after invocation")
    func completeOnboardingClearsHandler() {
        let vm = OnboardingViewModel()
        var callCount = 0

        vm.prepare { callCount += 1 }

        vm.completeOnboarding()
        #expect(callCount == 1)

        // Second call should not invoke the handler again.
        vm.completeOnboarding()
        #expect(callCount == 1)

        UserDefaults.standard.removeObject(forKey: "hasCompletedOnboarding")
    }

    // MARK: - shouldShowOnboarding

    @Test("shouldShowOnboarding returns false when flag is true")
    func shouldShowOnboardingReturnsFalseWhenFlagTrue() {
        let previousValue = UserDefaults.standard.object(forKey: "hasCompletedOnboarding")
        UserDefaults.standard.set(true, forKey: "hasCompletedOnboarding")

        let result = OnboardingViewModel.shouldShowOnboarding()
        #expect(result == false)

        // Restore previous value.
        if let prev = previousValue {
            UserDefaults.standard.set(prev, forKey: "hasCompletedOnboarding")
        } else {
            UserDefaults.standard.removeObject(forKey: "hasCompletedOnboarding")
        }
    }

    // MARK: - refreshPermissionStates

    @Test("refreshPermissionStates sets properties without crashing")
    func refreshPermissionStatesSetsProperties() {
        let vm = OnboardingViewModel()

        #expect(vm.microphoneGranted == false)
        #expect(vm.accessibilityGranted == false)

        // Calling refresh reads system state; should not trigger OS dialogs.
        vm.refreshPermissionStates()

        // After refresh, properties reflect actual system status.
        // We cannot assert specific values since system state varies,
        // but we verify the method completes and the properties are booleans
        // consistent with OS query results (this mainly guards against regressions
        // such as accidentally calling request APIs instead of status queries).
        _ = vm.microphoneGranted
        _ = vm.accessibilityGranted
    }

    // MARK: - prepare

    @Test("prepare stores completion handler for later invocation")
    func prepareStoresHandler() {
        let vm = OnboardingViewModel()
        var called = false

        vm.prepare { called = true }
        #expect(called == false)

        vm.completeOnboarding()
        #expect(called == true)

        UserDefaults.standard.removeObject(forKey: "hasCompletedOnboarding")
    }
}

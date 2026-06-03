// WelcomeViewModel drives the first-run wizard. The interesting behaviors
// are validation gates (storage + api-key) feeding `lastError`, the
// intro → setup → complete state, and the Start Server validation path.

import XCTest
@testable import oMLX

@MainActor
final class WelcomeViewModelTests: XCTestCase {

    // AppServices uses a weak reference to its services on WelcomeViewModel,
    // so the test must keep a strong reference for the lifetime of each case.
    private var services: AppServices!

    private func makeVM(basePath: String = "/Users/Fido/.omlx",
                        modelDir: String  = "/Users/Fido/.omlx/models",
                        port: Int = 8000,
                        apiKey: String? = nil) -> WelcomeViewModel {
        let cfg = AppConfig(
            bindAddress: "127.0.0.1",
            port: port,
            apiKey: apiKey,
            basePath: basePath,
            modelDir: modelDir,
            hfEndpoint: ""
        )
        services = AppServices(config: cfg, server: nil)
        return WelcomeViewModel(services: services, server: nil)
    }

    // MARK: - flow

    func testStartsOnIntroStep() {
        let vm = makeVM()
        XCTAssertEqual(vm.step, .intro)
    }

    func testBeginSetupAdvancesToSetupAndClearsError() {
        let vm = makeVM()
        vm.apiKey = "abc"
        XCTAssertFalse(vm.validateApiKey())
        XCTAssertNotNil(vm.lastError)
        vm.beginSetup()
        XCTAssertEqual(vm.step, .setup)
        XCTAssertNil(vm.lastError)
    }

    func testDefaultPortIs8000() {
        let vm = makeVM()
        XCTAssertEqual(vm.portText, "8000")
    }

    // MARK: - validateSetup

    func testValidateSetupHappyPath() {
        let vm = makeVM()
        vm.apiKey = "secret-key"
        XCTAssertTrue(vm.validateSetup())
        XCTAssertNil(vm.lastError)
    }

    func testValidateSetupFailsOnEmptyBase() {
        let vm = makeVM()
        vm.basePath = "   "
        vm.apiKey = "secret-key"
        XCTAssertFalse(vm.validateSetup())
        XCTAssertEqual(vm.lastError, "Base directory is required.")
    }

    func testValidateSetupFailsOnInvalidPort() {
        let vm = makeVM()
        vm.apiKey = "secret-key"
        vm.portText = "0"
        XCTAssertFalse(vm.validateSetup())
        XCTAssertEqual(vm.lastError, "Port must be a number between 1 and 65535.")
    }

    func testValidateSetupFailsOnPortNonNumeric() {
        let vm = makeVM()
        vm.apiKey = "secret-key"
        vm.portText = "abc"
        XCTAssertFalse(vm.validateSetup())
        XCTAssertEqual(vm.lastError, "Port must be a number between 1 and 65535.")
    }

    func testValidateSetupFailsOnShortApiKey() {
        let vm = makeVM()
        vm.apiKey = "abc"
        XCTAssertFalse(vm.validateSetup())
        XCTAssertEqual(vm.lastError, "API key must be at least 4 characters.")
    }

    func testValidateSetupFailsOnApiKeyWhitespace() {
        let vm = makeVM()
        // 4+ chars but a space inside — server-side validator rejects.
        vm.apiKey = "ab cd"
        XCTAssertFalse(vm.validateSetup())
        XCTAssertEqual(vm.lastError, "API key must not contain whitespace.")
    }

    func testValidateSetupFailsOnApiKeyNonPrintable() {
        let vm = makeVM()
        vm.apiKey = "abcd\u{007F}"   // DEL char, outside printable ASCII
        XCTAssertFalse(vm.validateSetup())
        XCTAssertEqual(vm.lastError, "API key must contain only printable ASCII.")
    }
}

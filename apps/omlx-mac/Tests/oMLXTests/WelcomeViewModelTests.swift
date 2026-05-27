// WelcomeViewModel drives the first-run wizard. The interesting behaviors
// are validation gates (storage + api-key) feeding `lastError`, the
// "Generate" button keeping `apiKey` and `apiKeyConfirm` in lockstep via
// the shared APIKeyGenerator, and `skipSnapshot()` which persists Storage
// values without an unvalidated API key on early close.

import XCTest
@testable import oMLX

@MainActor
final class WelcomeViewModelTests: XCTestCase {

    // AppServices uses a weak reference to its services on WelcomeViewModel,
    // so the test must keep a strong reference for the lifetime of each case
    // — otherwise skipSnapshot() short-circuits to AppConfig.default.
    private var services: AppServices!

    private func makeVM(basePath: String = "/Users/Fido/.omlx",
                        modelDir: String  = "/Users/Fido/.omlx/models",
                        port: Int = 8080,
                        apiKey: String? = nil) -> WelcomeViewModel {
        let cfg = AppConfig(
            host: "127.0.0.1",
            port: port,
            apiKey: apiKey,
            basePath: basePath,
            modelDir: modelDir,
            hfEndpoint: ""
        )
        services = AppServices(config: cfg, server: nil)
        return WelcomeViewModel(services: services, server: nil)
    }

    // MARK: - generateApiKey

    func testGenerateApiKeyPopulatesBothFields() {
        let vm = makeVM()
        XCTAssertEqual(vm.apiKey, "")
        XCTAssertEqual(vm.apiKeyConfirm, "")
        vm.generateApiKey()
        XCTAssertFalse(vm.apiKey.isEmpty)
        XCTAssertEqual(vm.apiKey, vm.apiKeyConfirm,
                       "Confirm field must mirror the generated key.")
    }

    func testGenerateApiKeyMatchesSharedGeneratorShape() {
        // Welcome must produce keys indistinguishable from the Security
        // screen's regenerate flow — both go through APIKeyGenerator.
        let vm = makeVM()
        vm.generateApiKey()
        XCTAssertTrue(vm.apiKey.hasPrefix(APIKeyGenerator.prefix))
        XCTAssertEqual(vm.apiKey.count,
                       APIKeyGenerator.prefix.count + APIKeyGenerator.bodyLength)
    }

    func testGenerateApiKeyClearsPriorError() {
        let vm = makeVM()
        vm.apiKey = "abc"
        XCTAssertFalse(vm.validateApiKey())
        XCTAssertNotNil(vm.lastError)
        vm.generateApiKey()
        XCTAssertNil(vm.lastError,
                     "Generate should clear any prior validation error.")
    }

    // MARK: - validateSetup

    func testValidateSetupHappyPath() {
        let vm = makeVM()
        vm.generateApiKey()
        XCTAssertTrue(vm.validateSetup())
        XCTAssertNil(vm.lastError)
    }

    func testValidateSetupFailsOnEmptyBase() {
        let vm = makeVM()
        vm.basePath = "   "
        vm.generateApiKey()
        XCTAssertFalse(vm.validateSetup())
        XCTAssertEqual(vm.lastError, "Base directory is required.")
    }

    func testValidateSetupFailsOnInvalidPort() {
        let vm = makeVM()
        vm.generateApiKey()
        vm.portText = "0"
        XCTAssertFalse(vm.validateSetup())
        XCTAssertEqual(vm.lastError, "Port must be a number between 1 and 65535.")
    }

    func testValidateSetupFailsOnPortNonNumeric() {
        let vm = makeVM()
        vm.generateApiKey()
        vm.portText = "abc"
        XCTAssertFalse(vm.validateSetup())
        XCTAssertEqual(vm.lastError, "Port must be a number between 1 and 65535.")
    }

    func testValidateSetupFailsOnShortApiKey() {
        let vm = makeVM()
        vm.apiKey = "abc"
        vm.apiKeyConfirm = "abc"
        XCTAssertFalse(vm.validateSetup())
        XCTAssertEqual(vm.lastError, "API key must be at least 4 characters.")
    }

    func testValidateSetupFailsOnApiKeyWhitespace() {
        let vm = makeVM()
        // 4+ chars but a space inside — server-side validator rejects.
        vm.apiKey = "ab cd"
        vm.apiKeyConfirm = "ab cd"
        XCTAssertFalse(vm.validateSetup())
        XCTAssertEqual(vm.lastError, "API key must not contain whitespace.")
    }

    func testValidateSetupFailsOnApiKeyNonPrintable() {
        let vm = makeVM()
        vm.apiKey = "abcd\u{007F}"   // DEL char, outside printable ASCII
        vm.apiKeyConfirm = "abcd\u{007F}"
        XCTAssertFalse(vm.validateSetup())
        XCTAssertEqual(vm.lastError, "API key must contain only printable ASCII.")
    }

    func testValidateSetupFailsOnConfirmMismatch() {
        let vm = makeVM()
        vm.apiKey = "sk-omlx-AAAA"
        vm.apiKeyConfirm = "sk-omlx-BBBB"
        XCTAssertFalse(vm.validateSetup())
        XCTAssertEqual(vm.lastError, "API keys do not match.")
    }

    // MARK: - skipSnapshot

    func testSkipSnapshotPreservesStorageOnValidInputs() {
        let vm = makeVM()
        vm.basePath = "/tmp/custom-base"
        vm.modelDir = "/tmp/custom-models"
        vm.portText = "9090"

        let snapshot = vm.skipSnapshot()
        XCTAssertEqual(snapshot.basePath, "/tmp/custom-base")
        XCTAssertEqual(snapshot.modelDir, "/tmp/custom-models")
        XCTAssertEqual(snapshot.port,     9090)
    }

    func testSkipSnapshotDropsUnvalidatedApiKey() {
        let vm = makeVM(apiKey: "previously-saved")
        vm.apiKey = "ab"               // too short
        vm.apiKeyConfirm = "ab"
        let snapshot = vm.skipSnapshot()
        XCTAssertEqual(snapshot.apiKey, "",
                       "An invalid in-progress key should be dropped on skip, " +
                       "not persisted over the saved value.")
    }

    func testSkipSnapshotKeepsValidatedApiKey() {
        let vm = makeVM()
        vm.generateApiKey()
        let generated = vm.apiKey
        let snapshot = vm.skipSnapshot()
        XCTAssertEqual(snapshot.apiKey, generated,
                       "A fully-validated key should make it into the snapshot.")
    }

    func testSkipSnapshotDefaultsModelDirToBaseWhenBlank() {
        let vm = makeVM(basePath: "/tmp/custom-base",
                        modelDir: "/tmp/custom-models")
        vm.modelDir = ""    // user hit Reset
        let snapshot = vm.skipSnapshot()
        XCTAssertEqual(snapshot.modelDir,
                       AppConfig.defaultModelDir(forBasePath: snapshot.basePath))
    }

    func testSkipSnapshotIgnoresInvalidPort() {
        let vm = makeVM(port: 8080)
        vm.portText = "999999"   // out of range
        let snapshot = vm.skipSnapshot()
        XCTAssertEqual(snapshot.port, 8080,
                       "Out-of-range port text should leave the existing port intact.")
    }
}

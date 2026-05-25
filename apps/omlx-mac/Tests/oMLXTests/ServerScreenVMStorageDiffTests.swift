// ServerScreenVM.storageDiff is what gates the Apply button and the actual
// migration. The risk is false-positive diffs (claiming a change when there
// isn't one) that would bounce the server for an idempotent click — or
// false-negative diffs that silently swallow user edits.
//
// All cases below feed the diff text fields that look textually different
// from `services.config` but are semantically equivalent — and assert no
// diff is reported.

import XCTest
@testable import oMLX

@MainActor
final class ServerScreenVMStorageDiffTests: XCTestCase {

    private func makeServices(basePath: String, modelDir: String) -> AppServices {
        let cfg = AppConfig(
            host: "127.0.0.1",
            port: 8080,
            apiKey: nil,
            basePath: basePath,
            modelDir: modelDir,
            hfEndpoint: ""
        )
        return AppServices(config: cfg, server: nil)
    }

    func testNoChanges() {
        let services = makeServices(basePath: "/Users/Fido/.omlx",
                                    modelDir: "/Users/Fido/.omlx/models")
        let vm = ServerScreenVM()
        vm.basePathText = "/Users/Fido/.omlx"
        vm.modelDirText = "/Users/Fido/.omlx/models"

        let diff = vm.storageDiff(services: services)
        XCTAssertFalse(diff.baseChanged)
        XCTAssertFalse(diff.dirChanged)
        XCTAssertFalse(diff.hasChanges)
    }

    func testBaseChangedOnly() {
        let services = makeServices(basePath: "/Users/Fido/.omlx",
                                    modelDir: "/Users/Fido/.omlx/models")
        let vm = ServerScreenVM()
        vm.basePathText = "/Users/Fido/.omlx-other"
        vm.modelDirText = "/Users/Fido/.omlx/models"

        let diff = vm.storageDiff(services: services)
        XCTAssertTrue(diff.baseChanged)
        XCTAssertFalse(diff.dirChanged)
        XCTAssertEqual(diff.normalizedBase, "/Users/Fido/.omlx-other")
    }

    func testModelDirChangedOnly() {
        let services = makeServices(basePath: "/Users/Fido/.omlx",
                                    modelDir: "/Users/Fido/.omlx/models")
        let vm = ServerScreenVM()
        vm.basePathText = "/Users/Fido/.omlx"
        vm.modelDirText = "/Volumes/SSD/models"

        let diff = vm.storageDiff(services: services)
        XCTAssertFalse(diff.baseChanged)
        XCTAssertTrue(diff.dirChanged)
        XCTAssertEqual(diff.normalizedModelDir, "/Volumes/SSD/models")
    }

    func testBothChanged() {
        let services = makeServices(basePath: "/Users/Fido/.omlx",
                                    modelDir: "/Users/Fido/.omlx/models")
        let vm = ServerScreenVM()
        vm.basePathText = "/Users/Fido/.omlx-other"
        vm.modelDirText = "/Volumes/SSD/models"

        XCTAssertTrue(vm.storageDiff(services: services).hasChanges)
    }

    func testTrailingSlashNormalizesToNoDiff() {
        // standardizingPath strips the trailing slash. Typing it in the
        // field must not flip the Apply button into "pending".
        let services = makeServices(basePath: "/Users/Fido/.omlx",
                                    modelDir: "/Users/Fido/.omlx/models")
        let vm = ServerScreenVM()
        vm.basePathText = "/Users/Fido/.omlx/"
        vm.modelDirText = "/Users/Fido/.omlx/models/"

        let diff = vm.storageDiff(services: services)
        XCTAssertFalse(diff.baseChanged)
        XCTAssertFalse(diff.dirChanged)
    }

    func testWhitespaceNormalizesToNoDiff() {
        let services = makeServices(basePath: "/Users/Fido/.omlx",
                                    modelDir: "/Users/Fido/.omlx/models")
        let vm = ServerScreenVM()
        vm.basePathText = "  /Users/Fido/.omlx  "
        vm.modelDirText = "\n/Users/Fido/.omlx/models\t"

        let diff = vm.storageDiff(services: services)
        XCTAssertFalse(diff.baseChanged)
        XCTAssertFalse(diff.dirChanged)
    }

    func testTildeExpansion() {
        let home = NSHomeDirectory()
        let services = makeServices(basePath: "\(home)/.omlx",
                                    modelDir: "\(home)/.omlx/models")
        let vm = ServerScreenVM()
        vm.basePathText = "~/.omlx"
        vm.modelDirText = "~/.omlx/models"

        let diff = vm.storageDiff(services: services)
        XCTAssertFalse(diff.baseChanged,
                       "tilde must expand before comparing to the home-absolute config value")
        XCTAssertFalse(diff.dirChanged)
    }

    func testEmptyTextDoesNotTriggerChange() {
        // If the user clears the field, we currently treat that as "no
        // change" rather than "set to empty". applyStorage() separately
        // refuses to submit empty values. Belt-and-suspenders.
        let services = makeServices(basePath: "/Users/Fido/.omlx",
                                    modelDir: "/Users/Fido/.omlx/models")
        let vm = ServerScreenVM()
        vm.basePathText = ""
        vm.modelDirText = ""

        let diff = vm.storageDiff(services: services)
        XCTAssertFalse(diff.baseChanged)
        XCTAssertFalse(diff.dirChanged)
    }
}

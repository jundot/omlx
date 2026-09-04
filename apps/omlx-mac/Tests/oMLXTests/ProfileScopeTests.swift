// The Profiles tab splits templates into Preset / Global scopes purely by
// the server's `is_builtin` flag. These tests pin that mapping so a
// rename of the wire field, or a future "default to Preset" decision,
// reaches the build instead of silently flipping every user template
// into the read-only group.

import XCTest
@testable import oMLX

final class ProfileScopeTests: XCTestCase {

    private func template(isBuiltin: Bool?) -> ProfileDTO {
        ProfileDTO(
            name: "x", displayName: "X",
            description: nil, createdAt: nil, updatedAt: nil,
            sourceTemplate: nil, isBuiltin: isBuiltin,
            exposeAsModel: nil, modelId: nil, hasEngineFields: nil,
            settings: nil
        )
    }

    func testBuiltinTrueResolvesToPreset() {
        XCTAssertEqual(template(isBuiltin: true).templateScope, .preset)
    }

    func testBuiltinFalseResolvesToGlobal() {
        XCTAssertEqual(template(isBuiltin: false).templateScope, .global)
    }

    func testMissingBuiltinDefaultsToGlobal() {
        // Legacy / partial server responses where the field is absent —
        // the server is the only source of truth for builtin status, so
        // "the server didn't claim built-in" means user-managed.
        XCTAssertEqual(template(isBuiltin: nil).templateScope, .global)
    }
}

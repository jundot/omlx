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

extension ProfileScopeTests {
    private func profile(_ json: String) throws -> ProfileDTO {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(ProfileDTO.self, from: Data(json.utf8))
    }

    func testIndependentModelWithSameNameRemainsModelScoped() throws {
        let model = try profile(#"{"name":"coding","display_name":"Coding","settings":{"temperature":0.1}}"#)
        let global = try profile(#"{"name":"coding","display_name":"Coding","settings":{"temperature":0.9}}"#)
        let active = resolveActiveProfileDisplay(activeName: "coding", modelProfiles: [model], templates: [global])
        XCTAssertEqual(active?.scope, .model)
    }

    func testCopyResolvesBySourceInsteadOfItsName() throws {
        let model = try profile(#"{"name":"copy-2","display_name":"Coding","source_template":"global-id","settings":{"temperature":0.9}}"#)
        let global = try profile(#"{"name":"global-id","display_name":"Coding","settings":{"temperature":0.9}}"#)
        let active = resolveActiveProfileDisplay(activeName: "copy-2", modelProfiles: [model], templates: [global])
        XCTAssertEqual(active?.scope, .global)
        XCTAssertEqual(active?.name, "global-id")
    }

    func testOldCopyDoesNotClaimLatestTemplateIsActive() throws {
        let model = try profile(#"{"name":"copy","display_name":"Coding","source_template":"global-id","settings":{"temperature":0.2}}"#)
        let global = try profile(#"{"name":"global-id","display_name":"Coding","settings":{"temperature":0.9}}"#)
        let active = resolveActiveProfileDisplay(activeName: "copy", modelProfiles: [model], templates: [global])
        XCTAssertEqual(active?.scope, .model)
        XCTAssertNil(model.matchingTemplate(in: [global]))
    }

    func testOrphanRemainsAnAccessibleModelProfile() throws {
        let model = try profile(#"{"name":"copy","display_name":"Coding","source_template":"deleted"}"#)
        XCTAssertNil(model.matchingTemplate(in: []))
        XCTAssertEqual(resolveActiveProfileDisplay(activeName: "copy", modelProfiles: [model], templates: [])?.scope, .model)
    }

    @MainActor
    func testDisplayNamesDoNotChangeActionReferences() throws {
        let vm = ModelSettingsScreenVM()
        vm.templates = [try profile(#"{"name":"t-123","display_name":"My Coding Profile"}"#)]
        vm.activeProfileScope = .global
        vm.activeProfileName = "t-123"
        XCTAssertEqual(vm.activeProfileState, .named(scope: .global, name: "t-123"))
        XCTAssertEqual(vm.displayProfileState, .named(scope: .global, name: "My Coding Profile"))
        vm.profileDirty = true
        XCTAssertEqual(vm.displayProfileState, .working(basedOn: .init(scope: .global, name: "My Coding Profile")))
    }
}

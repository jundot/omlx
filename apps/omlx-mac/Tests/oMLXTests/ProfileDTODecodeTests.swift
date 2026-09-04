// The ProfileDTO/TemplateListResponse pair is the contract with the
// Python admin API. `/admin/api/profile-templates` now returns
// user-created templates only — the shipped built-ins were retired in
// favor of the client-side preset bundle. The DTO still tolerates
// `is_builtin: true` decode for back-compat with stale disk state and
// older server builds, and `created_at`/`updated_at` remain nullable
// because user templates may persist without stamps in legacy state.

import XCTest
@testable import oMLX

final class ProfileDTODecodeTests: XCTestCase {

    private let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return d
    }()

    func testTemplateWithNullTimestampsDecodes() throws {
        // Templates with null `created_at`/`updated_at` (legacy disk
        // state from before timestamps were required) must still
        // decode — non-optional fields here would silently empty the
        // templates list via `try?` in ModelSettingsScreen.
        let json = """
        {
            "name": "shared-coding",
            "display_name": "Shared · Coding",
            "description": "Legacy entry without stamps.",
            "created_at": null,
            "updated_at": null,
            "settings": {
                "max_context_window": 131072,
                "temperature": 0.6,
                "enable_thinking": false
            },
            "is_builtin": false
        }
        """.data(using: .utf8)!

        let dto = try decoder.decode(ProfileDTO.self, from: json)
        XCTAssertEqual(dto.name, "shared-coding")
        XCTAssertEqual(dto.displayName, "Shared · Coding")
        XCTAssertNil(dto.createdAt)
        XCTAssertNil(dto.updatedAt)
        XCTAssertEqual(dto.isBuiltin, false)
        XCTAssertNotNil(dto.settings)
    }

    func testUserTemplateWithStampsDecodes() throws {
        let json = """
        {
            "name": "my-tuned",
            "display_name": "My Tuned",
            "description": null,
            "created_at": "2026-05-12T10:00:00+00:00",
            "updated_at": "2026-05-12T10:00:00+00:00",
            "settings": {"temperature": 0.3},
            "is_builtin": false
        }
        """.data(using: .utf8)!

        let dto = try decoder.decode(ProfileDTO.self, from: json)
        XCTAssertEqual(dto.createdAt, "2026-05-12T10:00:00+00:00")
        XCTAssertEqual(dto.updatedAt, "2026-05-12T10:00:00+00:00")
        XCTAssertEqual(dto.isBuiltin, false)
    }

    func testTemplateListDecodes() throws {
        // After the builtin-template retirement, the templates list is
        // user-created entries only. The DTO still tolerates the legacy
        // `is_builtin: true` shape from stale state — included here as
        // a back-compat smoke test, not a current API behavior.
        let json = """
        {
            "templates": [
                {
                    "name": "legacy-shipped",
                    "display_name": "Legacy Shipped",
                    "description": "decode back-compat only",
                    "created_at": null,
                    "updated_at": null,
                    "settings": {"temperature": 1.0},
                    "is_builtin": true
                },
                {
                    "name": "my-tuned",
                    "display_name": "My Tuned",
                    "description": null,
                    "created_at": "2026-05-12T10:00:00+00:00",
                    "updated_at": "2026-05-12T10:00:00+00:00",
                    "settings": {"temperature": 0.3},
                    "is_builtin": false
                }
            ]
        }
        """.data(using: .utf8)!

        let resp = try decoder.decode(TemplateListResponse.self, from: json)
        XCTAssertEqual(resp.templates.count, 2)
        XCTAssertEqual(resp.templates[0].isBuiltin, true)
        XCTAssertEqual(resp.templates[1].isBuiltin, false)
        XCTAssertEqual(resp.templates[1].createdAt, "2026-05-12T10:00:00+00:00")
    }

    func testModelProfileWithExposeAsModelDecodes() throws {
        // /api/models/{id}/profiles carries `expose_as_model` plus the
        // server-derived `model_id` and `has_engine_fields` — the server
        // owns the engine-field classification, the client never mirrors it.
        let json = """
        {
            "name": "thinking",
            "display_name": "Thinking",
            "description": null,
            "created_at": "2026-06-10T10:00:00+00:00",
            "updated_at": "2026-06-10T10:00:00+00:00",
            "settings": {"enable_thinking": true, "dflash_enabled": true},
            "expose_as_model": true,
            "model_id": "qwen-base:thinking",
            "has_engine_fields": true
        }
        """.data(using: .utf8)!

        let dto = try decoder.decode(ProfileDTO.self, from: json)
        XCTAssertEqual(dto.exposeAsModel, true)
        XCTAssertEqual(dto.modelId, "qwen-base:thinking")
        XCTAssertEqual(dto.hasEngineFields, true)
    }

    func testUpdateRequestOmitsExposeAsModelWhenNil() throws {
        // The server merges only the fields present in the body, so a
        // settings-only update must NOT carry `expose_as_model` — sending
        // it would clobber exposure state set from the web dashboard.
        let encoder = JSONEncoder()

        let settingsOnly = try JSONSerialization.jsonObject(
            with: encoder.encode(UpdateProfileRequest(settings: [:]))
        ) as! [String: Any]
        XCTAssertNil(settingsOnly["expose_as_model"])

        let exposeToggle = try JSONSerialization.jsonObject(
            with: encoder.encode(UpdateProfileRequest(exposeAsModel: true))
        ) as! [String: Any]
        XCTAssertEqual(exposeToggle["expose_as_model"] as? Bool, true)
    }

    func testLegacyResponseMissingIsBuiltinStillDecodes() throws {
        // The server is the new contract carrier and always sets is_builtin,
        // but the field is declared optional on the Swift side so an older
        // server (or a /api/models/{id}/profiles response that doesn't set
        // it) doesn't break decoding.
        let json = """
        {
            "name": "legacy",
            "display_name": "Legacy",
            "description": null,
            "created_at": "2026-05-01T00:00:00+00:00",
            "updated_at": "2026-05-01T00:00:00+00:00",
            "settings": {}
        }
        """.data(using: .utf8)!

        let dto = try decoder.decode(ProfileDTO.self, from: json)
        XCTAssertNil(dto.isBuiltin)
    }
}

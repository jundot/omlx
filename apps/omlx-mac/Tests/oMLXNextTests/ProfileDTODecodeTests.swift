// The ProfileDTO/TemplateListResponse pair is the contract with the
// Python admin API. When the server started returning null timestamps for
// shipped built-in templates, ProfileDTO's non-optional `createdAt`/`updatedAt`
// caused JSONDecoder to throw, and `try?` in ModelSettingsScreen swallowed it
// to an empty list — the user saw "no templates" silently.
//
// These tests pin the decode shape so any future server-side nullability
// change breaks the build instead of the UI.

import XCTest
@testable import oMLX_next

final class ProfileDTODecodeTests: XCTestCase {

    private let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return d
    }()

    func testBuiltinTemplateWithNullTimestampsDecodes() throws {
        let json = """
        {
            "name": "qwen36-thinking-general",
            "display_name": "Qwen3.6 Thinking · General",
            "description": "Built-in default.",
            "created_at": null,
            "updated_at": null,
            "settings": {
                "max_context_window": 131072,
                "temperature": 1.0,
                "enable_thinking": true
            },
            "is_builtin": true
        }
        """.data(using: .utf8)!

        let dto = try decoder.decode(ProfileDTO.self, from: json)
        XCTAssertEqual(dto.name, "qwen36-thinking-general")
        XCTAssertEqual(dto.displayName, "Qwen3.6 Thinking · General")
        XCTAssertNil(dto.createdAt)
        XCTAssertNil(dto.updatedAt)
        XCTAssertEqual(dto.isBuiltin, true)
        XCTAssertNotNil(dto.settings)
    }

    func testUserTemplateWithStampsDecodes() throws {
        // The other half of the contract: user-created templates carry
        // ISO timestamps and is_builtin=false.
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

    func testTemplateListWithMixedBuiltinAndUserDecodes() throws {
        // Mirrors the real /admin/api/profile-templates response: built-ins
        // first (null stamps, is_builtin: true), then user entries.
        let json = """
        {
            "templates": [
                {
                    "name": "qwen36-thinking-general",
                    "display_name": "Qwen3.6 Thinking · General",
                    "description": "ship default",
                    "created_at": null,
                    "updated_at": null,
                    "settings": {"temperature": 1.0},
                    "is_builtin": true
                },
                {
                    "name": "qwen36-instruct-general",
                    "display_name": "Qwen3.6 Instruct · General",
                    "description": "ship default",
                    "created_at": null,
                    "updated_at": null,
                    "settings": {"temperature": 0.7},
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
        XCTAssertEqual(resp.templates.count, 3)
        XCTAssertEqual(resp.templates[0].isBuiltin, true)
        XCTAssertEqual(resp.templates[2].isBuiltin, false)
        XCTAssertEqual(resp.templates[2].createdAt, "2026-05-12T10:00:00+00:00")
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

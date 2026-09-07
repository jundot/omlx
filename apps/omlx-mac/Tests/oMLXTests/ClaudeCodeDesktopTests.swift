// Claude Desktop tier-alias toggle tests.
// These tests pin the read shape (`claude_code.desktop_enabled`, optional
// so older servers that omit the key still decode) and the flat patch
// shape (`claude_code_desktop_enabled`) so a future rename on either edge
// breaks the build instead of silently dropping the toggle state.

import XCTest
@testable import oMLX

final class ClaudeCodeDesktopTests: XCTestCase {

    private let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return d
    }()

    private let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.keyEncodingStrategy = .convertToSnakeCase
        e.outputFormatting = [.sortedKeys]
        return e
    }()

    // MARK: - Decode

    func testDesktopEnabledDecodesWhenPresent() throws {
        // Mirrors `ClaudeCodeSettings.to_dict()` — the read shape is nested
        // under `claude_code`, separate from the flat
        // `claude_code_desktop_enabled` key on the patch body.
        let json = """
        {
            "server": {
                "host": "127.0.0.1",
                "port": 8080,
                "log_level": "info",
                "server_aliases": []
            },
            "claude_code": {
                "mode": "local",
                "opus_model": "model-a",
                "sonnet_model": "model-b",
                "haiku_model": "model-c",
                "desktop_enabled": true
            }
        }
        """.data(using: .utf8)!

        let dto = try decoder.decode(GlobalSettingsDTO.self, from: json)
        XCTAssertEqual(dto.claudeCode?.desktopEnabled, true)
        // Neighboring tier fields must be untouched.
        XCTAssertEqual(dto.claudeCode?.opusModel, "model-a")
        XCTAssertEqual(dto.claudeCode?.sonnetModel, "model-b")
        XCTAssertEqual(dto.claudeCode?.haikuModel, "model-c")
    }

    func testDesktopEnabledIsNilWhenAbsent() throws {
        // Older servers omit the key entirely. Decode must succeed with
        // nil so the VM can fall back to `false`.
        let json = """
        {
            "server": {
                "host": "127.0.0.1",
                "port": 8080,
                "log_level": "info",
                "server_aliases": []
            },
            "claude_code": {
                "mode": "local",
                "opus_model": "model-a"
            }
        }
        """.data(using: .utf8)!

        let dto = try decoder.decode(GlobalSettingsDTO.self, from: json)
        XCTAssertNil(dto.claudeCode?.desktopEnabled)
    }

    // MARK: - Patch encode

    func testPatchEncodesDesktopEnabledAsSnakeCaseFlatKey() throws {
        // The Python `GlobalSettingsRequest` accepts the toggle as the flat
        // `claude_code_desktop_enabled` key (omlx/admin/routes.py). The
        // .convertToSnakeCase strategy on Swift's encoder must produce
        // exactly that wire shape.
        var patch = GlobalSettingsPatch()
        patch.claudeCodeDesktopEnabled = true

        let data = try encoder.encode(patch)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]

        XCTAssertEqual(json["claude_code_desktop_enabled"] as? Bool, true)
    }

    func testPatchOmitsDesktopEnabledWhenNil() throws {
        // `encodeIfPresent` for Optionals means a nil toggle is skipped —
        // a patch that only touches a tier model must not also overwrite
        // the desktop toggle.
        var patch = GlobalSettingsPatch()
        patch.claudeCodeOpusModel = "model-a"

        let data = try encoder.encode(patch)
        let str = String(data: data, encoding: .utf8) ?? ""

        XCTAssertTrue(str.contains("\"claude_code_opus_model\""))
        XCTAssertFalse(str.contains("claude_code_desktop_enabled"))
    }
}

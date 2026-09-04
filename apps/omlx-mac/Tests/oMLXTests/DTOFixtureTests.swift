// Round-trip JSON fixtures captured from a live oMLX server through each
// DTO's Codable decoder. The signal we want is "server JSON changed shape
// in a way the Swift app can't decode" — the fixture lives in git, so a
// passing test means the wire contract is unchanged.
//
// Fixtures were captured via curl against the running server (see
// docs/Fixtures/README inline note below) and sanitized: API keys
// redacted, real paths replaced with `/Users/test/...`, LAN IPs swapped
// for the RFC 5737 documentation range (192.0.2.x).
//
// To re-capture (e.g., after intentional server-side wire changes):
//   PORT=<port> KEY=<api-key> COOKIES=$(mktemp)
//   curl -s -c "$COOKIES" -X POST "http://127.0.0.1:$PORT/admin/api/login" \
//        -H "Content-Type: application/json" \
//        -d "{\"api_key\":\"$KEY\",\"remember\":true}"
//   curl -s -b "$COOKIES" "http://127.0.0.1:$PORT/admin/api/<endpoint>" \
//        | python3 -m json.tool > Fixtures/<name>.json
// Then re-sanitize before committing.

import XCTest
@testable import oMLX

final class DTOFixtureTests: XCTestCase {

    // Matches the JSONDecoder config in OMLXClient: snake_case keys → camelCase
    // Codable members. Any DTO that the real client decodes must round-trip
    // here too.
    private static func makeDecoder() -> JSONDecoder {
        let dec = JSONDecoder()
        dec.keyDecodingStrategy = .convertFromSnakeCase
        return dec
    }

    private func fixture(_ name: String) throws -> Data {
        let dir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures")
        let url = dir.appendingPathComponent("\(name).json")
        return try Data(contentsOf: url)
    }

    // MARK: - oQ quantization

    func testOQStartRequestEncodesEnhancedOptions() throws {
        let request = OQStartRequest(
            modelPath: "/Users/test/models/model",
            oqLevel: 4,
            groupSize: 64,
            sensitivityModelPath: "",
            textOnly: false,
            dtype: "bfloat16",
            preserveMtp: false,
            enhanced: true,
            imatrixCachePath: "/Users/test/cache/imatrix.npz",
            imatrixReuseCache: true,
            imatrixStrict: true
        )
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let body = try JSONSerialization.jsonObject(with: encoder.encode(request)) as? [String: Any]

        XCTAssertEqual(body?["enhanced"] as? Bool, true)
        XCTAssertEqual(body?["imatrix_cache_path"] as? String, "/Users/test/cache/imatrix.npz")
        XCTAssertEqual(body?["imatrix_reuse_cache"] as? Bool, true)
        XCTAssertEqual(body?["imatrix_strict"] as? Bool, true)
    }

    // MARK: - Stats

    func testStatsSessionFixtureDecodes() throws {
        let data = try fixture("stats-session")
        let stats = try Self.makeDecoder().decode(StatsDTO.self, from: data)

        // The four-field tuple we surface across the menubar + Status screen.
        XCTAssertNotNil(stats.host)
        XCTAssertNotNil(stats.port)
        XCTAssertNotNil(stats.cliPrefix,
                        "Stats must carry cli_prefix so Integrations can render `omlx launch …` commands.")
        XCTAssertNotNil(stats.apiKey,
                        "Stats must surface api_key (empty string allowed) so the Welcome-skip path can recover it.")

        // active_models is the structure Status renders; nested fields can
        // be nil but the wrapper must always decode.
        XCTAssertNotNil(stats.activeModels)
    }

    // MARK: - Server info

    func testServerInfoFixtureDecodes() throws {
        let data = try fixture("server-info")
        let info = try Self.makeDecoder().decode(ServerInfoDTO.self, from: data)

        XCTAssertFalse(info.host.isEmpty,
                       "ServerInfo.host must be present — drives Settings → Listen Address.")
        XCTAssertGreaterThan(info.port, 0)
    }

    // MARK: - Global settings

    func testGlobalSettingsFixtureDecodes() throws {
        let data = try fixture("global-settings")
        let settings = try Self.makeDecoder().decode(GlobalSettingsDTO.self, from: data)

        // Sub-structures Server / Status / Integrations screens depend on.
        XCTAssertNotNil(settings.server,        "server block missing")
        XCTAssertNotNil(settings.model,         "model block missing")
        XCTAssertNotNil(settings.auth,          "auth block missing")
        XCTAssertNotNil(settings.claudeCode,    "claude_code block missing")
        XCTAssertNotNil(settings.integrations,  "integrations block missing")
        XCTAssertEqual(settings.scheduler?.embeddingBatchSize, 32)
        XCTAssertEqual(settings.huggingface?.hfCacheEnabled, true)
    }

    // MARK: - Models list

    func testModelsListFixtureDecodes() throws {
        let data = try fixture("models")
        let list = try Self.makeDecoder().decode(ListModelsResponse.self, from: data)

        // The fixture was captured with at least one model in the library.
        // Future re-captures could be empty, so just assert the array
        // structure decoded — not that it has entries.
        XCTAssertNotNil(list.models)
        // Sanity-check the first entry's shape if present.
        if let first = list.models.first {
            XCTAssertFalse(first.id.isEmpty, "ModelDTO.id must be non-empty.")
            XCTAssertEqual(first.displayName, "deepsweet/Qwen3.6-27B-UD-MLX-4bit")
        }
    }

    // MARK: - Profile list (per-model)

    func testModelProfilesFixtureDecodes() throws {
        let data = try fixture("model-profiles")
        let resp = try Self.makeDecoder().decode(ProfileListResponse.self, from: data)
        XCTAssertNotNil(resp.profiles,
                        "Profiles array must be present even when empty.")
    }

    // MARK: - Profile templates

    func testProfileTemplatesFixtureDecodes() throws {
        let data = try fixture("profile-templates")
        let resp = try Self.makeDecoder().decode(TemplateListResponse.self, from: data)
        // Templates array is empty in the captured fixture (no templates
        // configured on the dev server). Just exercise the decoder so a
        // server-side rename of `templates` → `items` would fail loudly.
        XCTAssertNotNil(resp.templates)
    }
}

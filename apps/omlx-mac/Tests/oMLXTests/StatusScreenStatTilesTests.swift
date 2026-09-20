// Regression guards for the Generated Tokens stat tile on the Status
// screen's Serving Stats row (mirrors the web dashboard's fourth card).
//
// The tile renders StatsDTO.totalCompletionTokens (already decoded from
// `total_completion_tokens` in /admin/api/stats) next to the prefill,
// cached, and cache-efficiency tiles. These guards lock:
//   • the wire contract: the fixture's total_completion_tokens decodes
//     into totalCompletionTokens,
//   • the catalog: status.tile.generated resolves to its English value,
//   • the call site: StatTilesRow actually binds the tile.

import XCTest
@testable import oMLX

final class StatusScreenStatTilesTests: XCTestCase {

    private func repoChild(_ components: [String]) -> URL {
        components.reduce(
            URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        ) { $0.appendingPathComponent($1) }
    }

    private func statsDecoder() -> JSONDecoder {
        // Matches the JSONDecoder config in OMLXClient.
        let dec = JSONDecoder()
        dec.keyDecodingStrategy = .convertFromSnakeCase
        return dec
    }

    // MARK: - Wire contract

    func testCompletionTokensDecodeFromServerShape() throws {
        let json = """
        {
          "total_tokens_served": 130,
          "total_cached_tokens": 60,
          "cache_efficiency": 60.0,
          "total_prompt_tokens": 100,
          "total_completion_tokens": 30,
          "total_requests": 2,
          "avg_prefill_tps": 100.0,
          "avg_generation_tps": 30.0,
          "uptime_seconds": 10.0,
          "active_models": { "models": [] }
        }
        """.data(using: .utf8)!

        let stats = try statsDecoder().decode(StatsDTO.self, from: json)
        XCTAssertEqual(stats.totalCompletionTokens, 30,
                       "Generated Tokens tile reads totalCompletionTokens; "
                           + "a decode miss here silently zeroes the tile.")
    }

    func testStatsSessionFixtureDecodesCompletionTokens() throws {
        let url = repoChild(["Fixtures", "stats-session.json"])
        let data = try Data(contentsOf: url)
        let stats = try statsDecoder().decode(StatsDTO.self, from: data)
        XCTAssertEqual(stats.totalCompletionTokens, 0,
                       "Fixture pins total_completion_tokens = 0")
    }

    // MARK: - Localization catalog

    // Every language the xcstrings catalog carries must translate the tile,
    // not just the en/ru/zh-Hans subset the neighbouring tiles happen to have.
    // Values mirror the web dashboard's status.stat.generated_tokens so the
    // two surfaces read identically.
    private static let generatedTokensExpected: [String: String] = [
        "en": "Generated Tokens",
        "ja": "生成トークン数",
        "ko": "생성된 토큰 수",
        "ru": "Сгенерированные токены",
        "zh-Hans": "生成 Token",
        "zh-Hant": "生成 Token 數",
    ]

    func testGeneratedTokensTileKeyInCatalog() throws {
        let url = repoChild(["..", "..", "Resources", "Localizable.xcstrings"])
            .standardizedFileURL
        let data = try Data(contentsOf: url)
        let root = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let strings = root?["strings"] as? [String: Any]

        guard let entry = strings?["status.tile.generated"] as? [String: Any],
              let localizations = entry["localizations"] as? [String: Any]
        else {
            XCTFail("status.tile.generated missing from Localizable.xcstrings")
            return
        }

        for (lang, expected) in Self.generatedTokensExpected {
            let value = (localizations[lang] as? [String: Any])?["stringUnit"]
                as? [String: Any]
            let resolved = (value?["value"]) as? String
            XCTAssertEqual(resolved, expected,
                           "status.tile.generated[\(lang)] should be \(expected)")
        }
    }

    // MARK: - Call site

    func testStatTilesRowBindsGeneratedTokensTile() throws {
        let url = repoChild(["..", "..", "Sources", "AppView", "Screens",
                             "StatusScreen.swift"]).standardizedFileURL
        let source = try String(contentsOf: url, encoding: .utf8)

        let row = try XCTUnwrap(
            source.components(separatedBy: "private struct StatTilesRow").last?
                .components(separatedBy: "private struct StatTile:").first,
            "StatTilesRow not found in StatusScreen.swift"
        )

        XCTAssertTrue(row.contains("\"status.tile.generated\""),
                      "StatTilesRow must mount the generated-tokens tile")
        XCTAssertTrue(row.contains("fmtNum($0.totalCompletionTokens)"),
                      "Generated Tokens tile must format totalCompletionTokens")

        // Tile order matches the web dashboard: prefill, cached, efficiency,
        // generated.
        let total = try XCTUnwrap(row.range(of: "\"status.tile.total\""))
        let cached = try XCTUnwrap(row.range(of: "\"status.tile.cached\""))
        let efficiency = try XCTUnwrap(row.range(of: "\"status.tile.cache_efficiency\""))
        let generated = try XCTUnwrap(row.range(of: "\"status.tile.generated\""))
        XCTAssertLessThan(total.lowerBound, cached.lowerBound)
        XCTAssertLessThan(cached.lowerBound, efficiency.lowerBound)
        XCTAssertLessThan(efficiency.lowerBound, generated.lowerBound)
    }
}

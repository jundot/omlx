import XCTest
@testable import oMLX

final class SettingsSearchTests: XCTestCase {
    private let english = SettingsSearch.englishBundle

    func testAllWordsMustMatchAcrossLabelPageAndAliases() throws {
        let port = try XCTUnwrap(SettingsSearch.results(for: "TCP", bundle: english).first)
        XCTAssertEqual(port.labelKey, "server.row.port")
        XCTAssertEqual(port.section, .server)

        let matches = SettingsSearch.results(for: "  PERFORMANCE\nRAM\tSIZE ", bundle: english)
        XCTAssertEqual(matches.map(\.labelKey), ["performance.cache.hot_size"])
        XCTAssertTrue(SettingsSearch.results(for: "cache nonexistent-term", bundle: english).isEmpty)
    }

    func testLocalizedAndEnglishNamesBothMatch() throws {
        let german = try languageBundle("de")
        let localized = SettingsSearch.results(for: "HOT CACHE GROßE", bundle: german)
        XCTAssertEqual(localized.map(\.labelKey), ["performance.cache.hot_size"])
        let english = try XCTUnwrap(SettingsSearch.results(for: "Hot Cache Size", bundle: german).first)
        XCTAssertEqual(english.id, localized.first?.id)
    }

    func testCatalogIdentifiersAndLocalizationKeysAreValid() {
        XCTAssertEqual(Set(SettingsSearch.entries.map(\.id)).count, SettingsSearch.entries.count)
        for entry in SettingsSearch.entries {
            for key in entry.pathKeys + [entry.labelKey] {
                let value = english.localizedString(forKey: key, value: "__missing__", table: nil)
                XCTAssertNotEqual(value, "__missing__", key)
                XCTAssertFalse(value.isEmpty, key)
                XCTAssertFalse(value.contains("%@"), "Search must not use dynamic values: \(key)")
            }
            if entry.askAIPage != nil { XCTAssertEqual(entry.section, .askAI) }
        }
    }

    func testWebSearchFieldsAndProviderAliasesOpenIntegrations() throws {
        for (query, key) in [
            ("web search Brave credential", "brave_api_key"),
            ("web search SearXNG endpoint", "searxng_url"),
            ("web search mojeek", "ddgs_backends"),
            ("web search full page", "content_mode"),
            ("web search characters", "content_max_chars")
        ] {
            let result = try XCTUnwrap(SettingsSearch.results(for: query, bundle: english).first)
            XCTAssertEqual(result.section, .integrations)
            XCTAssertEqual(result.labelKey, "integrations.websearch.\(key)")
        }
        let german = try languageBundle("de")
        let matches = SettingsSearch.results(for: "Suchanbieter", bundle: german)
        XCTAssertTrue(matches.contains { $0.labelKey == "integrations.websearch.provider" })
    }

    private func languageBundle(_ language: String) throws -> Bundle {
        let url = try XCTUnwrap(Bundle.main.url(forResource: language, withExtension: "lproj"))
        return try XCTUnwrap(Bundle(url: url))
    }
}

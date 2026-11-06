import XCTest
@testable import oMLX

final class SettingsSearchTests: XCTestCase {
    private let english = SettingsSearch.englishBundle

    func testBlankAndUnmatchedQueriesHaveNoResults() {
        for query in ["", " \t\n ", "no-such-setting-934782"] {
            XCTAssertTrue(SettingsSearch.results(for: query).isEmpty)
        }
    }

    func testAliasesFindTheSettingAndItsPage() throws {
        let port = try XCTUnwrap(SettingsSearch.results(for: "TCP", bundle: english).first)
        XCTAssertEqual(port.labelKey, "server.row.port")
        XCTAssertEqual(port.section, .server)
        XCTAssertNil(port.askAIPage)

        let memory = SettingsSearch.results(for: "RAM cache", bundle: english)
        XCTAssertTrue(memory.contains { $0.labelKey == "performance.cache.hot_size" })
        XCTAssertFalse(memory.contains { $0.labelKey == "performance.cache.ssd_size" })
    }

    func testAllWordsMustMatchAcrossLabelPageAndAliases() {
        let matches = SettingsSearch.results(for: "  PERFORMANCE\nRAM\tSIZE ", bundle: english)
        XCTAssertEqual(matches.map(\.labelKey), ["performance.cache.hot_size"])
        XCTAssertTrue(SettingsSearch.results(for: "cache nonexistent-term", bundle: english).isEmpty)
    }

    func testCurrentLanguageAndEnglishNamesBothMatch() throws {
        let german = try languageBundle("de")
        let localized = try XCTUnwrap(SettingsSearch.results(for: "Hot-Cache-Größe", bundle: german).first)
        let english = try XCTUnwrap(SettingsSearch.results(for: "Hot Cache Size", bundle: german).first)
        XCTAssertEqual(localized.id, english.id)
        XCTAssertEqual(localized.title(bundle: german), "Hot-Cache-Größe")
        XCTAssertEqual(localized.title(bundle: self.english), "Hot Cache Size")
    }

    func testMatchingIgnoresCaseAndAccents() throws {
        let matches = SettingsSearch.results(for: "HOT CACHE GROßE", bundle: try languageBundle("de"))
        XCTAssertEqual(matches.map(\.labelKey), ["performance.cache.hot_size"])
    }

    func testMissingTranslationFallsBackToEnglish() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("lproj")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try "\"performance.cache.hot_size\" = \"Cache privé\";".write(
            to: directory.appendingPathComponent("Localizable.strings"), atomically: true, encoding: .utf8
        )
        let bundle = try XCTUnwrap(Bundle(url: directory))
        XCTAssertEqual(SettingsSearch.localized("server.row.port", bundle: bundle), "Port")
        let entry = try XCTUnwrap(SettingsSearch.results(for: "prive", bundle: bundle).first)
        XCTAssertEqual(entry.title(bundle: bundle), "Cache privé")
        XCTAssertEqual(entry.breadcrumb(bundle: bundle), "Performance › Cache")
        XCTAssertTrue(SettingsSearch.results(for: "Hot Cache Size", bundle: bundle).contains { $0.id == entry.id })
    }

    func testResultsPreserveCatalogOrder() {
        let results = SettingsSearch.results(for: "cache", bundle: english)
        XCTAssertGreaterThan(results.count, 1)
        let positions = results.compactMap { entry in SettingsSearch.entries.firstIndex { $0.id == entry.id } }
        XCTAssertEqual(positions, positions.sorted())
        XCTAssertEqual(results.map(\.id), SettingsSearch.results(for: "cache", bundle: english).map(\.id))
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

    func testGlobalDestinationsAndPerModelExclusions() {
        XCTAssertEqual(Set(SettingsSearch.entries.map(\.section)),
                       [.status, .server, .network, .performance, .appearance, .askAI,
                        .downloads, .integrations, .security])
        let keys = Set(SettingsSearch.entries.map(\.labelKey))
        XCTAssertTrue(keys.contains("status.updates.channel"))
        XCTAssertTrue(keys.contains("downloads.format.label"))
        XCTAssertFalse(keys.contains("server.profile.min_p"))
        XCTAssertFalse(keys.contains("server.profile.enable_thinking"))
        XCTAssertFalse(keys.contains("server.profile.pin_in_memory"))
    }

    func testRepeatedLabelsLeadToTheCorrectAskAIPage() throws {
        let text = try XCTUnwrap(SettingsSearch.results(for: "text advanced thinking", bundle: english).first)
        XCTAssertEqual(text.askAIPage, .textAdvanced)
        let preset = try XCTUnwrap(SettingsSearch.results(for: "presets thinking", bundle: english).first)
        XCTAssertEqual(preset.askAIPage, .presets)
        let voice = try XCTUnwrap(SettingsSearch.results(for: "voice general prompt", bundle: english).first)
        XCTAssertEqual(voice.askAIPage, .voiceAdvanced)
        XCTAssertEqual(voice.breadcrumb(bundle: english), "Ask AI › Ask via Voice › Advanced Settings")
        let main = try XCTUnwrap(SettingsSearch.results(for: "voice hotkey", bundle: english).first)
        XCTAssertNil(main.askAIPage)
        XCTAssertEqual(main.section, .askAI)
    }

    func testConditionalSettingsRemainDiscoverable() {
        XCTAssertTrue(SettingsSearch.results(for: "custom ceiling", bundle: english)
            .contains { $0.labelKey == "performance.memory.custom_ceiling" })
        XCTAssertTrue(SettingsSearch.results(for: "voice screen capture", bundle: english)
            .contains { $0.askAIPage == .voiceAdvanced })
    }

    private func languageBundle(_ language: String) throws -> Bundle {
        let url = try XCTUnwrap(Bundle.main.url(forResource: language, withExtension: "lproj"))
        return try XCTUnwrap(Bundle(url: url))
    }
}

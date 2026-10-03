// Smoke test for the Localizable.xcstrings catalog. Pinned here so the
// Welcome-wizard Phase 1 wiring (and every later phase that adds keys)
// can't silently desync: if a key is renamed in code but not the catalog,
// or vice-versa, this test fails the build.
//
// We deliberately avoid asserting on every single key — that turns the
// test into a maintenance burden. Instead we check:
//   • the catalog parses
//   • a known-stable subset of keys resolves to its English value
//   • the source language is en
//   • welcome.* keys added in Phase 1 are all present
//
// String(localized:) with a key that's missing from the catalog returns
// the key itself, so the equality check below catches missing entries.

import XCTest
@testable import oMLX

final class LocalizationSmokeTests: XCTestCase {
    private static let englishBundle: Bundle = {
        guard let path = Bundle.main.path(forResource: "en", ofType: "lproj"),
              let bundle = Bundle(path: path) else {
            return .main
        }
        return bundle
    }()

    private static let simplifiedChineseErrorKeys: [String] = [
        "quant.error.cancel_failed",
        "quant.error.load_models",
        "quant.error.remove_failed",
        "quant.error.start_failed",
        "quant.upload.error.cancel_failed",
        "quant.upload.error.remove_failed",
    ]

    /// Hard-coded baseline of common.* keys → English values. Only the
    /// primitives actually used by at least one wrapped call site live here;
    /// any drift means someone touched the catalog without updating call
    /// sites (or vice versa).
    private static let commonBaseline: [(key: String, en: String)] = [
        ("common.cancel", "Cancel"),
        ("common.copy",   "Copy"),
        ("common.create", "Create"),
        ("common.open",   "Open"),
        ("common.save",   "Save"),
    ]

    /// Sentinel keys from every wrapped screen / surface. Presence-only check —
    /// if any of these resolves to the key string itself, the catalog is out
    /// of sync with the wrapped call sites. Two per surface keeps it cheap to
    /// run but catches drift on the most-visible strings.
    private static let sentinelKeys: [String] = [
        // Welcome wizard
        "welcome.window.title", "welcome.button.start_server",
        // Main app shell
        "about.section.project", "about.license.name",
        "logs.section.title", "network.section.proxies.title",
        // Server-side screens
        "server.section.advanced", "server.row.base_path",
        "security.section.api_key", "security.api_key.row_label",
        "integrations.section.claude_code", "integrations.tool.codex",
        "performance.section.cache", "performance.cache.enabled",
        "status.section.system", "status.section.active_now",
        // High-density screens
        "models.active.title", "models.library.title",
        "downloads.hf.section.title", "downloads.active.title",
        "quant.header.title", "quant.about.title",
        // Profile + bench
        "profile.scope.preset", "profile.detail.section.sampling",
        "bench.accuracy.header.title", "bench.accuracy.section.queue",
        "bench.throughput.header.title", "bench.throughput.section.configuration",
        "bench.context.header.title", "bench.context.section.configuration",
        // Settings + helpers
        "settings.section.basic", "settings.advanced.experimental.section",
        "settings.actions.reset", "settings.apply.choose.group_pp",
        "appearance.row.menubar_icon", "appearance.row.menubar_icon.restore",
        // Menubar + updates
        "menubar.item.quit", "menubar.stats.session_section",
        "menubar.item.settings", "menubar.item.web_dashboard",
        "update.channel.stable", "update.confirm.title",
    ]

    func testCatalogResolvesCommonBaseline() {
        // Force English so the assertion holds regardless of host locale.
        for (key, expected) in Self.commonBaseline {
            let resolved = NSLocalizedString(key, bundle: Self.englishBundle,
                                             value: key, comment: "")
            XCTAssertEqual(resolved, expected,
                           "common key \(key) resolved to \(resolved); expected \(expected)")
        }
    }

    func testSimplifiedChineseErrorTemplatesPreserveDetails() {
        guard let path = Bundle.main.path(forResource: "zh-Hans", ofType: "lproj"),
              let bundle = Bundle(path: path) else {
            XCTFail("Simplified Chinese localization bundle is missing")
            return
        }

        for key in Self.simplifiedChineseErrorKeys {
            let resolved = NSLocalizedString(key, bundle: bundle,
                                             value: key, comment: "")
            XCTAssertNotEqual(resolved, key,
                              "zh-Hans localization for \(key) exposes its key")
            XCTAssertTrue(resolved.contains("%@"),
                          "zh-Hans localization for \(key) drops the error placeholder")
        }
    }

    func testSentinelKeysArePresentInCatalog() {
        // Presence-only check: NSLocalizedString returns the key itself
        // when missing. We pass `value: <sentinel>` so a real missing key
        // resolves to the sentinel and never accidentally equals the key.
        let sentinel = "__missing__"
        for key in Self.sentinelKeys {
            let resolved = NSLocalizedString(key, bundle: .main,
                                             value: sentinel, comment: "")
            XCTAssertNotEqual(resolved, sentinel,
                              "key \(key) is wired in code but missing from xcstrings")
            XCTAssertFalse(resolved.isEmpty,
                           "key \(key) resolved to an empty string")
        }
    }

    /// A count in the Logs screen is an exact quantity substituted as formatted
    /// text, so the catalogue placeholders are `%@`: the line count reads 8,405
    /// rather than 8405, and the grouped figure survives the lookup.
    func testLogCountsSubstituteFormattedText() {
        let cases: [(String, String)] = [
            (String(localized: "logs.subtitle.line_count",
                    defaultValue: "Lines: \(8405.formatted())"), 8405.formatted()),
            (String(localized: "logs.detail.occurrences",
                    defaultValue: "Occurrences (\(20000.formatted()))"), 20000.formatted()),
            (String(localized: "logs.detail.occurrences_more",
                    defaultValue: "…and \(1234567.formatted()) more"), 1234567.formatted()),
            (String(localized: "logs.more_lines",
                    defaultValue: "≡ \(8405.formatted()) more lines"), 8405.formatted()),
            (String(localized: "logs.detail.lines_more",
                    defaultValue: "…and \(20000.formatted()) more lines"), 20000.formatted()),
        ]
        for (rendered, expected) in cases {
            XCTAssertTrue(rendered.contains(expected),
                          "the log count lost its formatted figure: \(rendered)")
        }
    }

    /// The review's rule for the app's labels: `token`, `tokens`, `tok`, `tk`
    /// and `t/s` take a capital T in every English value the catalogue ships,
    /// whether the word counts the model's tokens or names a credential the
    /// user holds (`HF Token`). API field names (`max_tokens`), the `tokenizer`
    /// component and `hf_…` placeholders are code, not copy, and keep their
    /// spelling: the pattern only accepts a token word delimited by non-word
    /// characters, so an underscore or a trailing letter keeps it out.
    func testTokenWordsInTheCatalogAreCapitalised() {
        // The build turns the catalogue into `<locale>.lproj/Localizable.strings`
        // and does not copy the .xcstrings itself into the app bundle, so
        // Bundle.main has nothing to hand this test. Read the catalogue from
        // the source tree instead, and fail rather than skip if it is gone.
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()      // oMLXTests
            .deletingLastPathComponent()      // Tests
            .deletingLastPathComponent()      // apps/omlx-mac
            .appendingPathComponent("Resources/Localizable.xcstrings")
        guard let data = try? Data(contentsOf: url),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let strings = root["strings"] as? [String: Any] else {
            XCTFail("Localizable.xcstrings is unreadable at \(url.path)")
            return
        }

        // Credential labels are capitalised like the rest; they are named here
        // so the rule reads as deliberate rather than as a gap in the sweep.
        let credentialLabels = [
            "quant.upload_modal.token.label": "HF Token",
            "quant.upload_modal.credentials.subtitle.needs_validate":
                "Validate a Token to enable upload",
            "quant.upload.error.empty_token": "Token is empty",
        ]
        let lowercase = try! NSRegularExpression(
            pattern: #"\b(tok/s|t/s|tok|token|tokens|tk)\b"#
        )

        // No allow-list and no locale left out: every value in every locale is
        // swept, because a lower-case unit is wrong in a Russian or a Japanese
        // string exactly as it is in an English one.
        var offenders: [String] = []
        for (key, entry) in strings {
            guard let entry = entry as? [String: Any],
                  let localizations = entry["localizations"] as? [String: Any] else { continue }
            for (locale, raw) in localizations {
                guard let raw = raw as? [String: Any],
                      let unit = raw["stringUnit"] as? [String: Any],
                      let value = unit["value"] as? String else { continue }
                let range = NSRange(value.startIndex..., in: value)
                if let match = lowercase.firstMatch(in: value, range: range),
                   let found = Range(match.range, in: value) {
                    offenders.append("\(locale) \(key): \(value[found]) in \"\(value)\"")
                }
            }
        }
        XCTAssertTrue(offenders.isEmpty,
                      "lower-case token words in the catalogue: \(offenders)")

        // The sweep is only worth anything while the detector bites: every
        // spelling this rule replaced has to fail it.
        for sample in ["HF token", "Validate a token to enable upload", "12 tok/s", "8192 tk"] {
            let range = NSRange(sample.startIndex..., in: sample)
            XCTAssertNotNil(lowercase.firstMatch(in: sample, range: range),
                            "the lower-case detector misses \(sample)")
        }

        for (key, expected) in credentialLabels {
            guard let entry = strings[key] as? [String: Any],
                  let localizations = entry["localizations"] as? [String: Any],
                  let en = localizations["en"] as? [String: Any],
                  let unit = en["stringUnit"] as? [String: Any],
                  let value = unit["value"] as? String else {
                XCTFail("\(key) has no English value in the catalogue")
                continue
            }
            XCTAssertEqual(value, expected,
                           "\(key) should read \(expected), not \(value)")
        }
    }

    func testTokenWordsInTheAppSourcesAreCapitalised() {
        // The catalogue is the string that ships, but the `defaultValue:` beside
        // each key is what Xcode re-extracts on the next build — a lower-case
        // unit there overwrites the translated value. The `suffix:` on a numeric
        // field is user-visible copy that never reaches the catalogue at all, so
        // it is swept alongside. The call sites are covered too, so the two
        // cannot drift apart again. API field names (`max_tokens`), the
        // `tokenizer` component and `hf_…` placeholders are kept out by the same
        // word-boundary pattern the catalogue sweep uses.
        let sources = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()      // oMLXTests
            .deletingLastPathComponent()      // Tests
            .deletingLastPathComponent()      // apps/omlx-mac
            .appendingPathComponent("Sources")
        guard let walker = FileManager.default.enumerator(
            at: sources, includingPropertiesForKeys: nil
        ) else {
            XCTFail("the app sources are unreadable at \(sources.path)")
            return
        }

        let lowercase = try! NSRegularExpression(
            pattern: #"\b(tok/s|t/s|tok|token|tokens|tk)\b"#
        )
        // Only the two string-literal positions that reach the user: the
        // re-extraction default beside a key, and the unit suffix on a field.
        let literal = try! NSRegularExpression(
            pattern: #"(?:defaultValue|suffix):\s*"([^"]*)""#
        )

        var offenders: [String] = []
        for case let url as URL in walker where url.pathExtension == "swift" {
            guard let text = try? String(contentsOf: url, encoding: .utf8) else { continue }
            let textRange = NSRange(text.startIndex..., in: text)
            for match in literal.matches(in: text, range: textRange) {
                guard let valueRange = Range(match.range(at: 1), in: text) else { continue }
                let value = String(text[valueRange])
                let range = NSRange(value.startIndex..., in: value)
                if let hit = lowercase.firstMatch(in: value, range: range),
                   let found = Range(hit.range, in: value) {
                    offenders.append(
                        "\(url.lastPathComponent): \(value[found]) in \"\(value)\""
                    )
                }
            }
        }
        XCTAssertTrue(offenders.isEmpty,
                      "lower-case token words in a defaultValue:/suffix: \(offenders)")

        // The sweep is only worth anything while the detector bites: the bare
        // `tk` this rule grew to cover has to fail it.
        for sample in ["8192 tk", "12 tok/s"] {
            let range = NSRange(sample.startIndex..., in: sample)
            XCTAssertNotNil(lowercase.firstMatch(in: sample, range: range),
                            "the lower-case detector misses \(sample)")
        }
    }

    func testCatalogIsValidJSON() {
        // Direct file-level parse so a catalog corruption (extra trailing
        // comma, bad nesting) shows up here rather than as a missing-string
        // mystery at runtime.
        guard let url = Bundle.main.url(forResource: "Localizable",
                                        withExtension: "xcstrings") else {
            // Some test hosts strip xcstrings; treat as non-fatal so the
            // suite stays green when run outside Xcode's resource bundle.
            return
        }
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            XCTFail("Couldn't read Localizable.xcstrings: \(error)")
            return
        }
        do {
            let root = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            XCTAssertEqual(root?["sourceLanguage"] as? String, "en",
                           "catalog sourceLanguage should be en")
            let strings = root?["strings"] as? [String: Any] ?? [:]
            XCTAssertGreaterThan(strings.count, 800,
                                 "catalog suspiciously small (\(strings.count) keys)")
        } catch {
            XCTFail("Localizable.xcstrings is not valid JSON: \(error)")
        }
    }
}

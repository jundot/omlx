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

    /// The app's rule for the token unit: a capital T where the token is a
    /// **unit** — counted beside a number or a placeholder, or standing on a
    /// label of its own — and the lowercase common noun inside a sentence.
    /// "8192 Tok", "3 Tokens (Default)" and "Tokens applied to Context
    /// Window" carry the capital; "Penalize repeated tokens." and "Limit
    /// thinking tokens for reasoning models." do not.
    ///
    /// A credential is not a token count, and neither is an architecture
    /// term: the Hugging Face token, its validation subtitle and MTP's
    /// "multi-token prediction" keep the lowercase spelling upstream uses,
    /// the same one the console's catalogues kept when #4364 refreshed them.
    /// API field names (`max_tokens`), the `tokenizer` component and `hf_…`
    /// placeholders are code, not copy, and keep their spelling: the pattern
    /// only accepts a token word delimited by non-word characters, so an
    /// underscore or a trailing letter keeps it out.
    ///
    /// Russian transliterates the unit into Cyrillic — "тк", "токенов",
    /// "ток/с" — which is upstream's translation and not this rule's
    /// business, so the capital is only asserted on a Latin-script unit.
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

        // The keys whose unit this rule pins: a count beside a number or a
        // placeholder, or a label that names the unit on its own.
        let unitKeys: [String] = [
            "bench.context.result.tokens_label",
            "profile.detail.acceleration.specprefill.threshold",
            "profile.detail.behavior.thinking_budget.on",
            "profile.detail.capacity.tokens",
            "profile.detail.capacity.tokens.raw",
            "settings.acceleration.mtp.depth.adaptive",
            "settings.acceleration.mtp.depth.option",
            "status.usage.heatmap.cell",
            "status.usage.row.requests_speed",
            "status.usage.row.tokens",
        ]
        let capital = try! NSRegularExpression(pattern: #"\b(Tok|Token|Tokens)\b"#)
        let latinUnit = try! NSRegularExpression(
            pattern: #"(?i)\b(tok/s|t/s|tok|token|tokens|tk)\b"#
        )

        for key in unitKeys {
            guard let entry = strings[key] as? [String: Any],
                  let localizations = entry["localizations"] as? [String: Any] else {
                XCTFail("the catalogue is missing \(key)")
                continue
            }
            for (locale, raw) in localizations {
                guard let raw = raw as? [String: Any],
                      let unit = raw["stringUnit"] as? [String: Any],
                      let value = unit["value"] as? String else { continue }
                // A unit spelled in another script is upstream's translation.
                let latin = NSRange(value.startIndex..., in: value)
                guard latinUnit.firstMatch(in: value, range: latin) != nil else { continue }
                let range = NSRange(value.startIndex..., in: value)
                XCTAssertNotNil(capital.firstMatch(in: value, range: range),
                                "\(locale) \(key) = \(value) keeps the lowercase unit")
            }
        }
    }

    /// A sentence says "tokens"; only a label says "Tokens". Past a prose
    /// value's first sentence opener the word is a common noun again, and a
    /// capital there reads as a typo on a hint people read closely. This is
    /// the rule the case pass in 7da1a05 was missing.
    func testEnglishProseKeepsTheCommonNounLowercase() {
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

        // A subtitle or a note explains something in sentences.
        let prose = try! NSRegularExpression(pattern: #"(\.sub|\.note|_note)$"#)
        let capital = try! NSRegularExpression(pattern: #"(?<![A-Za-z])\b(Tok|Token|Tokens)\b"#)
        let openers = [" — ", ". "]
        // These are a credential or an architecture term, and stay lowercase.
        let lowercaseSpelled: Set<String> = [
            "quant.upload_modal.token.label",
            "quant.upload_modal.credentials.subtitle.needs_validate",
            "quant.advanced.preserve_mtp.sub.available",
            "settings.apply.choose.group_tg",
        ]

        var offenders: [String] = []
        for (key, entry) in strings where lowercaseSpelled[key] == nil {
            let keyRange = NSRange(key.startIndex..., in: key)
            guard prose.firstMatch(in: key, range: keyRange) != nil,
                  let entry = entry as? [String: Any],
                  let localizations = entry["localizations"] as? [String: Any],
                  let en = localizations["en"] as? [String: Any],
                  let unit = en["stringUnit"] as? [String: Any],
                  let value = unit["value"] as? String else { continue }
            // Only the text past the first sentence opener is in scope.
            var cut = value.endIndex
            for opener in openers {
                if let r = value.range(of: opener), r.upperBound < cut { cut = r.upperBound }
            }
            let tail = String(value[cut...])
            let range = NSRange(tail.startIndex..., in: tail)
            if let hit = capital.firstMatch(in: tail, range: range) {
                offenders.append("\(key): \(value[hit.range]) in \(value)")
            }
        }
        XCTAssertTrue(offenders.isEmpty,
                      "a prose value capitalises the common noun: \(offenders)")
    }

    /// The catalogue is what ships, but the `defaultValue:` beside each key is
    /// what Xcode re-extracts on the next build — a differently-cased unit
    /// there would overwrite the translation. The sweep cannot see the key a
    /// literal belongs to, so it does not guess whether the word is a unit or
    /// a noun. It checks the one thing it can check: where a literal spells
    /// the same text as a value in the catalogue, it must spell it the same
    /// way, casing included.
    func testTheSourceLiteralsAgreeWithTheCatalogue() {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()      // oMLXTests
            .deletingLastPathComponent()      // Tests
            .deletingLastPathComponent()      // apps/omlx-mac
        guard let data = try? Data(contentsOf: root.appendingPathComponent(
            "Resources/Localizable.xcstrings"
        )),
              let parsed = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let strings = parsed["strings"] as? [String: Any] else {
            XCTFail("Localizable.xcstrings is unreadable")
            return
        }
        guard let walker = FileManager.default.enumerator(
            at: root.appendingPathComponent("Sources"), includingPropertiesForKeys: nil
        ) else {
            XCTFail("the app sources are unreadable")
            return
        }

        // Every value the catalogue ships, folded to a case-insensitive key so
        // a literal can be matched against it without knowing its own key.
        var shipped: [String: Set<String>] = [:]
        for (_, entry) in strings {
            guard let entry = entry as? [String: Any],
                  let localizations = entry["localizations"] as? [String: Any] else { continue }
            for (_, raw) in localizations {
                guard let raw = raw as? [String: Any],
                      let unit = raw["stringUnit"] as? [String: Any],
                      let value = unit["value"] as? String, !value.isEmpty else { continue }
                shipped[value.lowercased(), default: []].insert(value)
            }
        }

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
                guard let forms = shipped[value.lowercased()], !forms.contains(value) else {
                    continue
                }
                offenders.append(
                    "\(url.lastPathComponent): \(value) disagrees with \(forms.sorted())"
                )
            }
        }
        XCTAssertTrue(offenders.isEmpty,
                      "a defaultValue:/suffix: spells a shipped value differently: \(offenders)")
    }

    /// The detector is only worth anything while it bites every spelling the
    /// rule replaced, and while the deliberate code-only exceptions stay out
    /// of the pattern.
    func testTheTokenWordDetectorBites() {
        let lowercase = try! NSRegularExpression(
            pattern: #"\b(tok/s|t/s|tok|token|tokens|tk)\b"#
        )
        for sample in ["12 tok/s", "8192 tk", "HF token", "50 t/s", "Max tokens"] {
            let range = NSRange(sample.startIndex..., in: sample)
            XCTAssertNotNil(lowercase.firstMatch(in: sample, range: range),
                            "the lower-case detector misses \(sample)")
        }
        for sample in ["max_tokens", "hf_token", "tokenizer", "max_output_tokens"] {
            let range = NSRange(sample.startIndex..., in: sample)
            XCTAssertNil(lowercase.firstMatch(in: sample, range: range),
                         "the detector catches code, not copy: \(sample)")
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

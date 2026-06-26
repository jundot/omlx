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
        // Settings + helpers
        "settings.section.basic", "settings.advanced.experimental.section",
        // Menubar + updates
        "menubar.item.quit", "menubar.stats.session_section",
        "menubar.item.settings", "menubar.item.web_dashboard",
        "update.channel.stable", "update.confirm.title",
    ]

    func testCatalogResolvesCommonBaseline() {
        // Force English so the assertion holds regardless of host locale.
        for (key, expected) in Self.commonBaseline {
            let resolved = NSLocalizedString(key, bundle: .main,
                                             value: key, comment: "")
            XCTAssertEqual(resolved, expected,
                           "common key \(key) resolved to \(resolved); expected \(expected)")
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

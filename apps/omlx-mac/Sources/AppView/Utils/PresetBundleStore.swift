// Source of truth for the per-model settings "Preset" chip strip. Mirrors
// HTML's localStorage-cached preset bundle:
//
//   1. Disk cache (~/Library/Application Support/oMLX/preset_cache.json)
//   2. Bundled fixture (Contents/Resources/omlx/admin/static/omlx_preset.json)
//      — staged into the .app by Scripts/build.sh from the omlx package.
//   3. Remote refresh via POST /admin/api/presets/refresh (omlx.ai proxy).
//
// Replaces the previous behavior of treating `/api/profile-templates`
// entries with `is_builtin: true` as presets. Server-side builtins are
// retired (see Phase 1 — omlx/template/default_global_profile.json
// deleted), so the JSON bundle is now the only preset source for both
// Swift and HTML.

import Foundation
import SwiftUI

@MainActor
@Observable
final class PresetBundleStore {
    private(set) var entries: [PresetEntry] = []
    private(set) var isRefreshing: Bool = false
    private(set) var lastError: String?

    /// Where we persist the most recent successful bundle. Same role as
    /// HTML's `localStorage.omlx_preset_cache`.
    private let cacheURL: URL

    init() {
        self.cacheURL = Self.defaultCacheURL()
        // Load synchronously so the first SwiftUI render of the preset
        // chip strip already has entries — avoids a flash of empty state
        // while a background load completes.
        if let bundle = Self.loadDisk(at: cacheURL) ?? Self.loadFixture() {
            self.entries = bundle.presets
        }
    }

    /// POST /admin/api/presets/refresh, persist the response, swap in the
    /// new entries. On failure, `lastError` is set and `entries` is left
    /// untouched so the chip strip keeps its current contents.
    func refresh(client: OMLXClient) async {
        guard !isRefreshing else { return }
        isRefreshing = true
        defer { isRefreshing = false }

        do {
            let bundle = try await client.refreshPresetBundle()
            self.entries = bundle.presets
            self.lastError = nil
            Self.writeDisk(bundle, to: cacheURL)
        } catch {
            self.lastError = (error as? OMLXClientError)
                .map { String(describing: $0) }
                ?? error.localizedDescription
        }
    }

    // MARK: - Helpers

    private static func defaultCacheURL() -> URL {
        let base = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first ?? FileManager.default.temporaryDirectory
        let dir = base.appendingPathComponent("oMLX", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("preset_cache.json")
    }

    private static func loadDisk(at url: URL) -> PresetBundleDTO? {
        guard FileManager.default.fileExists(atPath: url.path),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(PresetBundleDTO.self, from: data)
    }

    /// Pull the JSON bundled inside the app at
    /// `Contents/Resources/omlx/admin/static/omlx_preset.json`. The file
    /// is staged into the app by `Scripts/build.sh` (omlx package rsync),
    /// not by Xcode — so this lookup uses the on-disk path under
    /// `Bundle.main.resourceURL` rather than `Bundle.main.url(forResource:)`
    /// which only finds Xcode-managed resources.
    private static func loadFixture() -> PresetBundleDTO? {
        guard let resources = Bundle.main.resourceURL else { return nil }
        let url = resources
            .appendingPathComponent("omlx", isDirectory: true)
            .appendingPathComponent("admin", isDirectory: true)
            .appendingPathComponent("static", isDirectory: true)
            .appendingPathComponent("omlx_preset.json")
        guard FileManager.default.fileExists(atPath: url.path),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(PresetBundleDTO.self, from: data)
    }

    private static func writeDisk(_ bundle: PresetBundleDTO, to url: URL) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(bundle) else { return }
        try? data.write(to: url, options: [.atomic])
    }
}

// Shipped preset bundle — the canonical source for "preset" chips in
// the per-model settings UI. Mirrors `omlx/admin/static/omlx_preset.json`
// (the same file HTML's `loadPresets()` reads from localStorage cache or
// the static fixture). The remote-refresh endpoint
// `POST /admin/api/presets/refresh` proxies omlx.ai and returns this
// same shape so we can decode it identically whether it came from
// the bundled fixture or a live fetch.

import Foundation

struct PresetBundleDTO: Codable, Sendable, Equatable {
    let version: Int
    let presets: [PresetEntry]
}

struct PresetEntry: Codable, Sendable, Equatable, Identifiable {
    let name: String
    let displayName: String
    let description: String?
    /// Sampling/profile settings to apply when this preset is selected.
    /// Free-form key-value to mirror `ProfileTemplate.settings` — see the
    /// ProfileDTO/AnyCodable bridge used by ModelSettingsScreen's apply path.
    let settings: [String: AnyCodable]

    var id: String { name }

    enum CodingKeys: String, CodingKey {
        case name
        case displayName = "display_name"
        case description
        case settings
    }
}

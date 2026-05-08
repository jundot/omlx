// PR 7 — slice of GET /admin/api/global-settings used by ServerScreen.
// Fields the screens don't render are intentionally absent so adding new
// endpoints in PR 8/9 doesn't risk decoding regressions on changes to fields
// we don't care about.
//
// The patch shape is FLAT (the request body), not nested — that's how the
// server's `GlobalSettingsRequest` Pydantic model is defined (admin/routes.py).

import Foundation

struct GlobalSettingsDTO: Codable, Equatable, Sendable {
    let basePath: String?
    let server: ServerSettings
    let model: ModelSettings?
    let scheduler: SchedulerSettings?
    let cache: CacheSettings?
    let auth: AuthSettings?
    let system: SystemInfo?

    struct ServerSettings: Codable, Equatable, Sendable {
        let host: String
        let port: Int
        let logLevel: String
        let serverAliases: [String]
        let sseKeepaliveMode: String?
    }

    struct ModelSettings: Codable, Equatable, Sendable {
        let modelDirs: [String]?
        let maxModelMemory: String?
        let modelFallback: Bool?
    }

    struct SchedulerSettings: Codable, Equatable, Sendable {
        let maxConcurrentRequests: Int
    }

    struct CacheSettings: Codable, Equatable, Sendable {
        let enabled: Bool
        let ssdCacheDir: String?
        let ssdCacheMaxSize: String?
        let hotCacheOnly: Bool?
        let hotCacheMaxSize: String?
    }

    struct AuthSettings: Codable, Equatable, Sendable {
        let apiKeySet: Bool
        let skipApiKeyVerification: Bool?
    }

    struct SystemInfo: Codable, Equatable, Sendable {
        let totalMemoryBytes: Int64?
        let totalMemory: String?
    }
}

/// Patch body for POST /admin/api/global-settings. Fields are flat (not
/// nested) — the server merges any non-nil field. We expose only those fields
/// PR 7's ServerScreen mutates.
struct GlobalSettingsPatch: Encodable, Equatable, Sendable {
    var host: String? = nil
    var port: Int? = nil
    var logLevel: String? = nil
    var maxConcurrentRequests: Int? = nil
}

struct UpdateGlobalSettingsResponse: Decodable, Sendable {
    let success: Bool
    let message: String?
    let runtimeApplied: [String]?
}

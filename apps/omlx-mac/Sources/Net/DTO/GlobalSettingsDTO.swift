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
    let claudeCode: ClaudeCodeSettings?
    let integrations: IntegrationsSettings?

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
        let apiKey: String?
        let skipApiKeyVerification: Bool?
        let subKeys: [SubKeyDTO]?
    }

    struct SystemInfo: Codable, Equatable, Sendable {
        let totalMemoryBytes: Int64?
        let totalMemory: String?
    }

    struct ClaudeCodeSettings: Codable, Equatable, Sendable {
        let contextScalingEnabled: Bool?
        let targetContextSize: Int?
        let mode: String?
        let opusModel: String?
        let sonnetModel: String?
        let haikuModel: String?
    }

    struct IntegrationsSettings: Codable, Equatable, Sendable {
        let codexModel: String?
        let opencodeModel: String?
        let openclawModel: String?
        let piModel: String?
        let openclawToolsProfile: String?
    }
}

/// Patch body for POST /admin/api/global-settings. Fields are flat (not
/// nested) — the server merges any non-nil field. PR 7 wires the server tab's
/// fields; PR 9 adds the Claude Code + integrations + auth fields needed by
/// IntegrationsScreen and SecurityScreen.
struct GlobalSettingsPatch: Encodable, Equatable, Sendable {
    // Server (PR 7)
    var host: String? = nil
    var port: Int? = nil
    var logLevel: String? = nil
    var maxConcurrentRequests: Int? = nil

    // Claude Code (PR 9)
    var claudeCodeContextScalingEnabled: Bool? = nil
    var claudeCodeTargetContextSize: Int? = nil
    var claudeCodeMode: String? = nil
    var claudeCodeOpusModel: String? = nil
    var claudeCodeSonnetModel: String? = nil
    var claudeCodeHaikuModel: String? = nil

    // Other integrations (PR 9)
    var integrationsCodexModel: String? = nil
    var integrationsOpencodeModel: String? = nil
    var integrationsOpenclawModel: String? = nil
    var integrationsPiModel: String? = nil
    var integrationsOpenclawToolsProfile: String? = nil

    // Auth (PR 9)
    var skipApiKeyVerification: Bool? = nil
}

struct UpdateGlobalSettingsResponse: Decodable, Sendable {
    let success: Bool
    let message: String?
    let runtimeApplied: [String]?
}

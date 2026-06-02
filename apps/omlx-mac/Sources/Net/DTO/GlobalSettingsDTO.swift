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
    let memory: MemorySettings?
    let scheduler: SchedulerSettings?
    let cache: CacheSettings?
    let idleTimeout: IdleTimeoutSettings?
    let auth: AuthSettings?
    let system: SystemInfo?
    /// Server-wide default sampling parameters. Patched through the flat
    /// `sampling_*` keys on `GlobalSettingsPatch` (the Python endpoint is
    /// non-nested for write but the read response is nested under
    /// `sampling`). Falls back per-model when a model profile leaves a
    /// field empty — this is the "server defaults" the design's profile
    /// fallback chain points to.
    let sampling: SamplingDTO?
    let huggingface: HuggingFaceDTO?
    let modelscope: ModelScopeDTO?
    let network: NetworkDTO?
    let claudeCode: ClaudeCodeSettings?
    let integrations: IntegrationsSettings?
    let mcp: MCPSettings?

    struct ServerSettings: Codable, Equatable, Sendable {
        let host: String
        let port: Int
        let logLevel: String
        let serverAliases: [String]
        let sseKeepaliveMode: String?
    }

    struct ModelSettings: Codable, Equatable, Sendable {
        let modelDirs: [String]?
        let modelFallback: Bool?
    }

    struct SchedulerSettings: Codable, Equatable, Sendable {
        let maxConcurrentRequests: Int
        let embeddingBatchSize: Int?
        let chunkedPrefill: Bool?
    }

    struct CacheSettings: Codable, Equatable, Sendable {
        let enabled: Bool
        let ssdCacheDir: String?
        let ssdCacheMaxSize: String?
        let hotCacheOnly: Bool?
        let hotCacheMaxSize: String?
        let initialCacheBlocks: Int?
    }

    /// Mirrors the `memory.*` block of GET /admin/api/global-settings.
    /// The prefill guard and tier are runtime-applied. When enabled, the
    /// server preflights prefill memory before kicking the engine.
    struct MemorySettings: Codable, Equatable, Sendable {
        let prefillMemoryGuard: Bool?
        let memoryGuardTier: String?
        let memoryGuardCustomCeilingGb: Double?
    }

    /// Mirrors the `idle_timeout.*` block. `idle_timeout_seconds == nil`
    /// disables the global fallback; per-model overrides may still apply.
    /// Server enforces `>= 60` on patch.
    struct IdleTimeoutSettings: Codable, Equatable, Sendable {
        let idleTimeoutSeconds: Int?
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

    /// Mirrors `omlx.settings.HuggingFaceSettings`. Empty string means
    /// "use HF default" (huggingface.co). When set, the server applies
    /// the value via env var (HF_ENDPOINT) so the HF library picks it up.
    struct HuggingFaceDTO: Codable, Equatable, Sendable {
        let endpoint: String
    }

    /// Mirrors `omlx.settings.SamplingSettings`. The full server surface
    /// today is six fields; min_p / presence_penalty / TTL / behavior flags
    /// the design mocks at the server level don't exist server-side and
    /// stay per-model.
    struct SamplingDTO: Codable, Equatable, Sendable {
        let maxContextWindow: Int
        let maxTokens: Int
        let temperature: Double
        let topP: Double
        let topK: Int
        let repetitionPenalty: Double
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
        let hermesModel: String?
        let copilotModel: String?
    }

    /// Mirrors `omlx.settings.MCPSettings`. The server stores a single path to
    /// an MCP config file consumed by every integration launcher (Claude
    /// Code, OpenClaw, Hermes, …). Empty / nil means no MCP server is wired.
    struct MCPSettings: Codable, Equatable, Sendable {
        let configPath: String?
    }

    /// Mirrors `omlx.settings.ModelScopeSettings`. Empty string means
    /// "use the default" (modelscope.cn). Patched via `ms_endpoint`.
    struct ModelScopeDTO: Codable, Equatable, Sendable {
        let endpoint: String
    }

    /// Mirrors `omlx.settings.NetworkSettings`. All four fields are simple
    /// strings; empty string = unset. Patched via `network_*` flat keys.
    struct NetworkDTO: Codable, Equatable, Sendable {
        let httpProxy: String
        let httpsProxy: String
        let noProxy: String
        let caBundle: String
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
    var embeddingBatchSize: Int? = nil

    // Server — Advanced (Phase 4).
    /// Extra host names the server identifies as for cookie/host-header
    /// purposes. Empty array clears. Encoded as a JSON array under
    /// `server_aliases`.
    var serverAliases: [String]? = nil
    /// SSE keep-alive line strategy: `"chunk"` (default), `"comment"`, or
    /// `"off"`. Server rejects anything else with a 400.
    var sseKeepaliveMode: String? = nil

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
    var integrationsHermesModel: String? = nil
    var integrationsCopilotModel: String? = nil

    /// Path to an MCP server config file. Empty string clears the field on
    /// the server (`global_settings.mcp.config_path = None`). Shared across
    /// every integration launcher.
    var mcpConfig: String? = nil

    // Auth (PR 9)
    var skipApiKeyVerification: Bool? = nil
    /// Update the configured API key. Server applies and persists via
    /// /admin/api/global-settings (`api_key` field). Only valid when an
    /// admin session is already authenticated — first-time setup still
    /// goes through /admin/api/setup-api-key.
    var apiKey: String? = nil

    // Server-wide default sampling parameters. The Python `GlobalSettingsRequest`
    // accepts these as flat snake-cased fields (sampling_temperature, etc.) —
    // see `omlx/admin/routes.py:229-234`. They patch in-place; non-nil fields
    // overwrite the corresponding `GlobalSettings.sampling.*` value.
    var samplingMaxContextWindow: Int? = nil
    var samplingMaxTokens: Int? = nil
    var samplingTemperature: Double? = nil
    var samplingTopP: Double? = nil
    var samplingTopK: Int? = nil
    var samplingRepetitionPenalty: Double? = nil

    /// Hugging Face mirror endpoint. Empty string resets the server-side
    /// HF_ENDPOINT env var to the HF default (huggingface.co). Patches in-
    /// place via `omlx/admin/routes.py:2804`.
    var hfEndpoint: String? = nil

    /// ModelScope mirror endpoint. Empty string = use modelscope.cn.
    /// Patched via `ms_endpoint` (encoder converts to snake_case).
    var msEndpoint: String? = nil

    /// Process-wide outbound HTTP proxy. Empty string = unset. The server
    /// applies via env vars (HTTP_PROXY / HTTPS_PROXY / NO_PROXY /
    /// SSL_CERT_FILE) so HF, MS, and Sparkle all pick them up.
    var networkHttpProxy: String? = nil
    var networkHttpsProxy: String? = nil
    var networkNoProxy: String? = nil
    var networkCaBundle: String? = nil

    // Phase 3 — Performance / Memory / Cache / Lifecycle.
    //
    // All flat (snake-cased on the wire by `convertToSnakeCase`). Server
    // applies live wherever possible — see `omlx/admin/routes.py` for the
    // per-field apply paths. `initial_cache_blocks` requires restart;
    // memory guard settings are hot-applied.

    var memoryPrefillMemoryGuard: Bool? = nil
    /// Memory guard tier: `"safe"`, `"balanced"`, `"aggressive"`, or
    /// `"custom"`. For custom, pair with `memoryGuardCustomCeilingGb`.
    var memoryGuardTier: String? = nil
    var memoryGuardCustomCeilingGb: Double? = nil

    /// When the requested model isn't loaded, fall back to any loaded
    /// model rather than 404.
    var modelFallback: Bool? = nil

    /// Multi-block prefill — splits long prompts across scheduler ticks.
    var chunkedPrefill: Bool? = nil

    var cacheEnabled: Bool? = nil
    var hotCacheOnly: Bool? = nil
    var hotCacheMaxSize: String? = nil
    var ssdCacheDir: String? = nil
    var ssdCacheMaxSize: String? = nil
    /// Starting cache block count. Requires server restart to take effect.
    var initialCacheBlocks: Int? = nil

    /// Server-wide model auto-unload after N seconds idle. Server enforces
    /// `>= 60`. Pass `nil` to leave unchanged; the Swift VM models the
    /// "disabled" case separately (server-side disable isn't a patch op
    /// today — see PerformanceScreen).
    var idleTimeoutSeconds: Int? = nil
}

struct UpdateGlobalSettingsResponse: Decodable, Sendable {
    let success: Bool
    let message: String?
    let runtimeApplied: [String]?
}

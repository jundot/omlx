// PR 7 — GET /admin/api/stats. The response is large; we only decode the
// slice rendered by StatusScreen + the menubar (PR 4 already polls this
// endpoint via a custom dictionary read; PR 8 will migrate the menubar
// poller onto this typed DTO).

import Foundation

struct StatsDTO: Codable, Equatable, Sendable {
    let totalTokensServed: Int
    let totalCachedTokens: Int
    let cacheEfficiency: Double
    let totalPromptTokens: Int
    let totalCompletionTokens: Int
    let totalRequests: Int
    let avgPrefillTps: Double
    let avgGenerationTps: Double
    let uptimeSeconds: Double

    let host: String?
    let port: Int?
    /// Server-configured API key, surfaced for the Integrations setup-command
    /// builders. Empty string when the server has no key configured.
    let apiKey: String?
    /// Absolute CLI invocation prefix used for `omlx launch <tool>` strings.
    /// `nil` on older servers; callers fall back to the bare `"omlx"` token
    /// (matches the dashboard JS `_launchCmd`).
    let cliPrefix: String?

    let activeModels: ActiveModelsDTO
    /// Disk-side SSD cache observability. Present on `scope=session` reads;
    /// `nil` if the server can't compute it (no global settings yet).
    let runtimeCache: RuntimeCacheDTO?

    struct ActiveModelsDTO: Codable, Equatable, Sendable {
        let models: [ActiveModelDTO]
        let modelMemoryUsed: Int64?
        let modelMemoryMax: Int64?
        let totalActiveRequests: Int?
        let totalWaitingRequests: Int?
    }

    struct ActiveModelDTO: Codable, Equatable, Sendable, Identifiable {
        let id: String
        let estimatedSize: Int64?
        let estimatedSizeFormatted: String?
        let pinned: Bool?
        let isLoading: Bool?
        let activeRequests: Int?
        let waitingRequests: Int?
    }

    /// Mirrors `_build_runtime_cache_observability` in `omlx/admin/routes.py`.
    /// SSD totals + hot-cache (memory tier) totals are both surfaced so
    /// StatusScreen can show the same two-tier picture the HTML dashboard
    /// does.
    struct RuntimeCacheDTO: Codable, Equatable, Sendable {
        let basePath: String?
        let ssdCacheDir: String?
        let totalNumFiles: Int
        let totalSizeBytes: Int64
        let effectiveBlockSizes: [Int]?
        /// Memory-tier (`hot_cache_*`) totals. `hot_cache_max_bytes == 0`
        /// signals that the memory tier is disabled — the UI hides those
        /// rows in that case to match the HTML behaviour.
        let hotCacheMaxBytes: Int64?
        let hotCacheSizeBytes: Int64?
        let hotCacheEntries: Int?
    }
}

/// Response from `POST /admin/api/ssd-cache/clear`. `totalDeleted` counts
/// files removed across loaded-model managers + direct filesystem cleanup.
struct ClearSsdCacheResponse: Codable, Sendable {
    let status: String?
    let totalDeleted: Int
}

/// Response from `POST /admin/api/hot-cache/clear`. Mirrors the SSD-clear
/// shape so callers can treat both clear endpoints uniformly.
struct ClearHotCacheResponse: Codable, Sendable {
    let status: String?
    let totalCleared: Int?
}

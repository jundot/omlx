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

    let activeModels: ActiveModelsDTO

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
}

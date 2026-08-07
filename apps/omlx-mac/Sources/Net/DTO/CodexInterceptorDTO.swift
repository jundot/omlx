import Foundation

struct CodexInterceptorMetricsDTO: Codable, Equatable, Sendable {
    let durationMs: Double?
    let firstByteMs: Double?
    let firstVisibleMs: Double?
    let connectMs: Double?
    let connectionReused: Bool?
    let inputTokens: Int?
    let outputTokens: Int?
    let cachedTokens: Int?
    let cacheHitPercent: Double?
    let tokensPerSecond: Double?
    let residencyStatus: String?
    let prefixPrefillStatus: String?
    let performanceWarning: String?
}

struct CodexInterceptorStatusDTO: Codable, Equatable, Sendable {
    let phase: String
    let running: Bool
    let error: String?
    let sessionId: String?
    let startedAt: Double?
    let model: String?
    let activeModel: String?
    let activeContextWindow: Int?
    let activeModelLoaded: Bool?
    let activeModelLoading: Bool?
    let pendingModel: String?
    let pendingContextWindow: Int?
    let pendingModelLoaded: Bool?
    let pendingModelLoading: Bool?
    let modelSwitching: Bool?
    let modelSwitchLoading: Bool?
    let modelSwitchError: String?
    let localSlot: String?
    let project: String?
    let proxyPid: Int?
    let codexPid: Int?
    let codexRunning: Bool
    let activeLocalRequests: Int
    let localRequests: Int
    let cloudRequests: Int
    let completedRequests: Int
    let failedRequests: Int
    let lastRoute: String?
    let lastRequestedModel: String?
    let lastEffectiveModel: String?
    let latestMetrics: CodexInterceptorMetricsDTO
    let warmupStatus: String
    let warmupModel: String?
    let warmupModelLoaded: Bool?
    let warmupModelLoading: Bool?
    let diagnosticsPath: String?
    let configPath: String
    let configModified: Bool
}

struct CodexInterceptorDoctorDTO: Codable, Equatable, Sendable {
    let ready: Bool
    let codexAppInstalled: Bool
    let codexAppPath: String?
    let codexRunning: Bool
    let mitmproxyAvailable: Bool
    let mitmproxySource: String?
    let configPath: String
    let configWillBeModified: Bool
}

struct CodexInterceptorStartRequest: Codable, Sendable {
    let model: String
    let project: String
    let localSlot: String?
    let contextWindow: Int?
    let launchApp: Bool
    let replaceExisting: Bool
}

struct CodexInterceptorSwitchRequest: Codable, Sendable {
    let model: String
    let contextWindow: Int?
}

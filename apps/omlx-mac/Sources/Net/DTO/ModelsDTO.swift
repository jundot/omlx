// PR 8 — GET /admin/api/models response + per-model settings shape, plus the
// patch body for PUT /admin/api/models/{id}/settings.
//
// Server returns a much larger settings dict than we expose here; we decode
// only the fields ModelSettingsScreen actually renders. The patch is
// intentionally narrower than `ModelSettingsRequest` (admin/routes.py:99) —
// PR 8 wires the Basic + Advanced tabs; experimental flags (DFlash,
// SpecPrefill, MTP, TurboQuant, trust_remote_code) stay browser-only until
// they have native UI affordances.

import Foundation

struct ListModelsResponse: Codable, Sendable {
    let models: [ModelDTO]
}

struct ModelDTO: Codable, Equatable, Sendable, Identifiable {
    let id: String
    let modelPath: String?
    let loaded: Bool
    let isLoading: Bool
    let estimatedSize: Int64
    let estimatedSizeFormatted: String?
    let pinned: Bool?
    let isDefault: Bool?
    let engineType: String?
    let modelType: String?
    let configModelType: String?
    let settings: ModelSettingsDTO?
}

struct ModelSettingsDTO: Codable, Equatable, Sendable {
    let modelAlias: String?
    let modelTypeOverride: String?
    let maxContextWindow: Int?
    let maxTokens: Int?
    let temperature: Double?
    let topP: Double?
    let topK: Int?
    let minP: Double?
    let presencePenalty: Double?
    let repetitionPenalty: Double?
    let forceSampling: Bool?
    let maxToolResultTokens: Int?
    let enableThinking: Bool?
    let thinkingBudgetEnabled: Bool?
    let thinkingBudgetTokens: Int?
    let ttlSeconds: Int?
    let isPinned: Bool?
    let isDefault: Bool?
    let displayName: String?
    let activeProfileName: String?
}

/// Patch body for PUT /admin/api/models/{id}/settings. Flat snake-cased keys
/// (Encoder converts via `.convertToSnakeCase`).
struct ModelSettingsPatch: Encodable, Equatable, Sendable {
    var modelAlias: String? = nil
    var modelTypeOverride: String? = nil
    var maxContextWindow: Int? = nil
    var maxTokens: Int? = nil
    var temperature: Double? = nil
    var topP: Double? = nil
    var topK: Int? = nil
    var minP: Double? = nil
    var presencePenalty: Double? = nil
    var repetitionPenalty: Double? = nil
    var ttlSeconds: Int? = nil
    var enableThinking: Bool? = nil
    var thinkingBudgetEnabled: Bool? = nil
    var thinkingBudgetTokens: Int? = nil
    var maxToolResultTokens: Int? = nil
    var forceSampling: Bool? = nil
    var isPinned: Bool? = nil
}

struct SimpleStatusResponse: Codable, Sendable {
    let status: String?
    let message: String?
    let success: Bool?
}

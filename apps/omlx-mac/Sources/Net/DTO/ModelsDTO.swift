// PR 8 (+ PR 14 advanced/experimental expansion) — GET /admin/api/models
// response + per-model settings shape, plus the patch body for
// PUT /admin/api/models/{id}/settings.
//
// All fields the HTML admin modal renders are now exposed here so the
// Swift Advanced tab can reach feature parity. Experimental flags
// (TurboQuant KV, IndexCache, SpecPrefill, DFlash, native MTP) decode
// alongside compatibility readouts (`dflash_compatible`,
// `mtp_compatible`, etc.) sourced from the server so the UI can disable
// switches the runtime won't accept.

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
    /// Lower-level config-derived model class (e.g. `deepseek_v32`,
    /// `glm_moe_dsa`). Used to gate the IndexCache row to DSA models.
    let configModelType: String?
    /// Server-side default for `enable_thinking` derived from the model
    /// (chat template, config). UI shows it as the inherited value when
    /// `enable_thinking` is unset and offers a one-click reset to it.
    let thinkingDefault: Bool?
    /// True when the model is structurally compatible with DFlash (block
    /// diffusion speculative decoding). The toggle stays disabled when false.
    let dflashCompatible: Bool?
    /// Human-readable explanation when `dflashCompatible` is false. Surfaced
    /// as a tooltip on the disabled DFlash toggle.
    let dflashCompatibilityReason: String?
    /// True when the global paged-SSD cache directory is configured. The
    /// DFlash SSD-cache sub-toggle stays disabled when false.
    let dflashSsdCacheAvailable: Bool?
    /// True when the model is structurally compatible with native MTP.
    let mtpCompatible: Bool?
    let mtpCompatibilityReason: String?
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
    let reasoningParser: String?
    let ttlSeconds: Int?
    let isPinned: Bool?
    let isDefault: Bool?
    let displayName: String?
    let activeProfileName: String?
    // Security
    let trustRemoteCode: Bool?
    // Chat-template kwargs (free-form dict + a sibling list of keys the
    // user wants to *force* — those go to `forced_ct_kwargs` server-side
    // and override the request's `chat_template_kwargs`).
    let chatTemplateKwargs: [String: AnyCodable]?
    let forcedCtKwargs: [String]?
    // Experimental: TurboQuant KV cache
    let turboquantKvEnabled: Bool?
    let turboquantKvBits: Double?
    // Experimental: IndexCache (DSA models only)
    let indexCacheFreq: Int?
    // Experimental: SpecPrefill
    let specprefillEnabled: Bool?
    let specprefillDraftModel: String?
    let specprefillKeepPct: Double?
    let specprefillThreshold: Int?
    // Experimental: DFlash (block diffusion speculative decoding)
    let dflashEnabled: Bool?
    let dflashDraftModel: String?
    let dflashDraftQuantEnabled: Bool?
    let dflashDraftQuantWeightBits: Int?
    let dflashDraftQuantActivationBits: Int?
    let dflashDraftQuantGroupSize: Int?
    let dflashMaxCtx: Int?
    let dflashInMemoryCache: Bool?
    let dflashInMemoryCacheMaxEntries: Int?
    /// Stored in bytes server-side; the editor row exposes a GiB-scaled
    /// view via `DflashByteSize.gibToBytes` / `bytesToGib`.
    let dflashInMemoryCacheMaxBytes: Int64?
    let dflashSsdCache: Bool?
    let dflashSsdCacheMaxBytes: Int64?
    let dflashDraftWindowSize: Int?
    let dflashDraftSinkSize: Int?
    let dflashVerifyMode: String?
    // Experimental: native MTP (mlx-lm PR 990 / PR 15 monkey-patch)
    let mtpEnabled: Bool?
}

/// Patch body for PUT /admin/api/models/{id}/settings. Flat snake-cased
/// keys (Encoder converts via `.convertToSnakeCase`). `nil` fields are
/// omitted by `encodeIfPresent` so the server merges instead of resetting.
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
    // Security
    var trustRemoteCode: Bool? = nil
    var reasoningParser: String? = nil
    // Chat-template kwargs
    var chatTemplateKwargs: [String: AnyCodable]? = nil
    var forcedCtKwargs: [String]? = nil
    // Experimental: TurboQuant KV
    var turboquantKvEnabled: Bool? = nil
    var turboquantKvBits: Double? = nil
    // Experimental: IndexCache
    var indexCacheFreq: Int? = nil
    // Experimental: SpecPrefill
    var specprefillEnabled: Bool? = nil
    var specprefillDraftModel: String? = nil
    var specprefillKeepPct: Double? = nil
    var specprefillThreshold: Int? = nil
    // Experimental: DFlash
    var dflashEnabled: Bool? = nil
    var dflashDraftModel: String? = nil
    var dflashDraftQuantEnabled: Bool? = nil
    var dflashDraftQuantWeightBits: Int? = nil
    var dflashDraftQuantActivationBits: Int? = nil
    var dflashDraftQuantGroupSize: Int? = nil
    var dflashMaxCtx: Int? = nil
    var dflashInMemoryCache: Bool? = nil
    var dflashInMemoryCacheMaxEntries: Int? = nil
    var dflashInMemoryCacheMaxBytes: Int64? = nil
    var dflashSsdCache: Bool? = nil
    var dflashSsdCacheMaxBytes: Int64? = nil
    var dflashDraftWindowSize: Int? = nil
    var dflashDraftSinkSize: Int? = nil
    var dflashVerifyMode: String? = nil
    // Experimental: native MTP
    var mtpEnabled: Bool? = nil
}

/// Generic acknowledgment shape returned by non-streaming admin endpoints
/// that just need to signal completion (model load/unload, settings patch,
/// task cancel/remove, stats clear, sub-key CRUD, etc.). Server responses
/// vary in which subset of `status`/`message`/`success` they populate, so
/// all three are optional and callers typically only check that the call
/// did not throw.
struct SimpleStatusResponse: Codable, Sendable {
    let status: String?
    let message: String?
    let success: Bool?
}

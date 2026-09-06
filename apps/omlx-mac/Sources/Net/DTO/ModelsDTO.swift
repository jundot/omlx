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
    let displayName: String?
    let modelPath: String?
    let loaded: Bool
    let isLoading: Bool
    let estimatedSize: Int64
    let estimatedSizeFormatted: String?
    let actualSize: Int64?
    let actualSizeFormatted: String?
    let pinned: Bool?
    let isDefault: Bool?
    let isFavorite: Bool?
    let engineType: String?
    let modelType: String?
    /// Lower-level config-derived model class (e.g. `deepseek_v32`,
    /// `glm_moe_dsa`). Used to gate the IndexCache row to DSA models.
    let configModelType: String?
    /// Native context window from the model's config.json. The Context
    /// Bench target selector hides presets beyond it.
    let modelContextLength: Int?
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
    /// Qwen4-Exp PLE mmap capability and server-side forced residency decision.
    let qwen4PleSsdOffloadSupported: Bool?
    let qwen4PleSsdOffloadForced: Bool?
    let qwen4PleResidentBytes: Int64?
    let qwen4PleMmapBytes: Int64?
    /// MoE expert streaming capability and the currently projected resident
    /// footprint for pinned experts, PLE, and configured hot-cache slots.
    let expertStreamingSupported: Bool?
    let expertStreamingResidentBytes: Int64?
    /// True for builtin virtual entries (e.g. the MarkItDown document
    /// converter) that have no real load/unload lifecycle.
    let virtual: Bool?
    let settings: ModelSettingsDTO?
}

extension ModelDTO {
    /// Single size figure for compact UI: the observed footprint once the
    /// model has settled, the estimate while loading or before one exists.
    var sizeLabel: String {
        if isLoading {
            return estimatedSizeFormatted ?? ""
        }
        return actualSizeFormatted ?? estimatedSizeFormatted ?? ""
    }
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
    let qwen4PleSsdOffload: Bool?
    // SSD Expert Streaming / Soft-REAP
    let expertStreamingEnabled: Bool?
    let expertStreamingMode: String?
    let expertStreamingManifest: String?
    let expertStreamingCacheExperts: Int?
    let expertStreamingScratchExperts: Int?
    let expertStreamingCachePolicy: String?
    let expertStreamingFastResourceLoading: Bool?
    let expertStreamingDirectIo: Bool?
    let expertStreamingNativeDemand: Bool?
    let expertStreamingDecodeScratchAsCache: Bool?
    let expertStreamingIoCoalescingKib: Int?
    let thinkingBudgetEnabled: Bool?
    let thinkingBudgetTokens: Int?
    let reasoningParser: String?
    let ttlSeconds: Int?
    let isPinned: Bool?
    let isDefault: Bool?
    let isFavorite: Bool?
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
    // Experimental: private Qwen3.5/3.6/3.8 ANE/GPU prefill
    let qwen35AnePrefillEnabled: Bool?
    let qwen35AnePrefillSequenceLength: Int?
    let qwen35AnePrefillTailPaddingMinTokens: Int?
    let qwen35AnePrefillFraction: Double?
    let qwen35AnePrefillFusedDown: Bool?
    let qwen35AnePrefillMaxLayers: Int?
    let qwen35AnePrefillDualAne: Bool?
    let qwen35AnePrefillGdn: Bool?
    let qwen35AnePrefillGdnFraction: Double?
    let qwen35AnePrefillGdnMaxLayers: Int?
    let qwen35AnePrefillCpuEnabled: Bool?
    let qwen35AnePrefillCpuFraction: Double?
    let qwen35AnePrefillCpuDownFraction: Double?
    let qwen35AnePrefillCpuGdnFraction: Double?
    let qwen35AnePrefillCpuThreads: Int?
    let qwen35AnePrefillCpuSharedResource: Bool?
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
    let dflashBlockSize: Int?
    let dflashVerifyMode: String?
    // Experimental: native MTP (mlx-lm PR 990 / PR 15 monkey-patch)
    let mtpEnabled: Bool?
    // Experimental: VLM MTP (mlx-vlm assistant-drafter speculative decoding)
    let vlmMtpEnabled: Bool?
    let vlmMtpDraftModel: String?
    let vlmMtpDraftBlockSize: Int?
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
    var qwen4PleSsdOffload: Bool? = nil
    // SSD Expert Streaming / Soft-REAP
    var expertStreamingEnabled: Bool? = nil
    var expertStreamingMode: String? = nil
    var expertStreamingManifest: String? = nil
    var expertStreamingCacheExperts: Int? = nil
    var expertStreamingScratchExperts: Int? = nil
    var expertStreamingCachePolicy: String? = nil
    var expertStreamingFastResourceLoading: Bool? = nil
    var expertStreamingDirectIo: Bool? = nil
    var expertStreamingNativeDemand: Bool? = nil
    var expertStreamingDecodeScratchAsCache: Bool? = nil
    var expertStreamingIoCoalescingKib: Int? = nil
    var thinkingBudgetEnabled: Bool? = nil
    var thinkingBudgetTokens: Int? = nil
    var maxToolResultTokens: Int? = nil
    var forceSampling: Bool? = nil
    var isPinned: Bool? = nil
    var isFavorite: Bool? = nil
    // Security
    var trustRemoteCode: Bool? = nil
    var reasoningParser: String? = nil
    // Chat-template kwargs
    var chatTemplateKwargs: [String: AnyCodable]? = nil
    var forcedCtKwargs: [String]? = nil
    // Experimental: TurboQuant KV
    var turboquantKvEnabled: Bool? = nil
    var turboquantKvBits: Double? = nil
    // Experimental: private Qwen3.5/3.6/3.8 ANE/GPU prefill
    var qwen35AnePrefillEnabled: Bool? = nil
    var qwen35AnePrefillSequenceLength: Int? = nil
    var qwen35AnePrefillTailPaddingMinTokens: Int? = nil
    var qwen35AnePrefillFraction: Double? = nil
    var qwen35AnePrefillFusedDown: Bool? = nil
    var qwen35AnePrefillMaxLayers: Int? = nil
    var qwen35AnePrefillDualAne: Bool? = nil
    var qwen35AnePrefillGdn: Bool? = nil
    var qwen35AnePrefillGdnFraction: Double? = nil
    var qwen35AnePrefillGdnMaxLayers: Int? = nil
    var qwen35AnePrefillCpuEnabled: Bool? = nil
    var qwen35AnePrefillCpuFraction: Double? = nil
    var qwen35AnePrefillCpuDownFraction: Double? = nil
    var qwen35AnePrefillCpuGdnFraction: Double? = nil
    var qwen35AnePrefillCpuThreads: Int? = nil
    var qwen35AnePrefillCpuSharedResource: Bool? = nil
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
    var dflashBlockSize: Int? = nil
    var dflashVerifyMode: String? = nil
    // Experimental: native MTP
    var mtpEnabled: Bool? = nil
    // Experimental: VLM MTP
    var vlmMtpEnabled: Bool? = nil
    var vlmMtpDraftModel: String? = nil
    var vlmMtpDraftBlockSize: Int? = nil
}

/// Result returned after the server validates and stores a Soft-REAP JSON
/// manifest for a model.
struct ExpertManifestUploadResponse: Codable, Equatable, Sendable {
    let success: Bool
    let manifestPath: String
    let layers: Int
    let pinnedCountMin: Int
    let pinnedCountMax: Int
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

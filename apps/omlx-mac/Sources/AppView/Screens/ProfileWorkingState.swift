// The Profiles tab's three-state model (named / working / defaults), plus the
// helpers that translate between the per-field VM state and the loose
// `settings` dict the server wants on /admin/api/profile-templates and
// /admin/api/models/{id}/profiles.
//
// The "Working profile" is a client-only construct: as soon as the user
// edits any field in Basic/Advanced, the active named profile detaches
// into an in-memory scratch the user can Apply / Save as new / Update or
// Revert. The server never sees a working profile — only the result of
// one of those four actions.

import Foundation

/// What the Profiles banner + chip group show as the model's current
/// attach state. The view layer renders one of three banner variants
/// from this enum.
enum ActiveProfileState: Equatable {
    /// The model is using a named profile (preset/global/model). No
    /// unsaved edits.
    case named(scope: ProfileScope, name: String)

    /// The user has edited fields since the last load. `basedOn` is the
    /// profile the edits forked from (nil when no profile was active).
    case working(basedOn: NamedProfileRef?)

    /// No profile assigned and no edits — the model falls back to the
    /// server's `GlobalSettings.sampling` defaults.
    case defaults

    struct NamedProfileRef: Equatable, Hashable {
        let scope: ProfileScope
        let name: String
    }
}

/// Snapshot of all editable model-settings fields used to detect dirty
/// state. The VM stores one of these on `load()` and compares the live
/// `@Published` values against it to decide whether to surface the
/// Working banner.
///
/// Equality is intentionally permissive: empty string == nil, since the
/// editor uses `""` for "unset" and the server sends `null`. Same for
/// numeric strings parsing as the same number.
struct ModelSettingsSnapshot: Equatable {
    var alias: String
    var modelTypeOverride: String
    var contextLength: String
    var maxTokens: String
    var temperature: String
    var topP: String
    var topK: String
    var minP: String
    var repetitionPenalty: String
    var presencePenalty: String
    var ttlSeconds: String

    var enableThinking: Bool
    var thinkingBudgetEnabled: Bool
    var thinkingBudgetTokens: String
    var limitToolResults: Bool
    var toolResultLimitTokens: String
    var forceSampling: Bool
    var isPinned: Bool

    var trustRemoteCode: Bool
    var reasoningParser: String

    var chatTemplateEntries: [ChatTemplateKwargEntry]

    var turboquantKvEnabled: Bool
    var turboquantKvBits: String
    var indexCacheEnabled: Bool
    var indexCacheFreq: String
    var specprefillEnabled: Bool
    var specprefillDraftModel: String
    var specprefillKeepPct: String
    var specprefillThreshold: String
    var dflashEnabled: Bool
    var dflashDraftModel: String
    var dflashDraftQuantEnabled: Bool
    var dflashDraftQuantWeightBits: String
    var dflashDraftQuantActivationBits: String
    var dflashDraftQuantGroupSize: String
    var dflashMaxCtx: String
    var dflashVerifyMode: String
    var dflashDraftWindowSize: String
    var dflashDraftSinkSize: String
    var dflashInMemoryCache: Bool
    var dflashInMemoryCacheGib: String
    var dflashInMemoryCacheMaxEntries: String
    var dflashSsdCache: Bool
    var dflashSsdCacheGib: String
    var mtpEnabled: Bool
}

/// Settings keys the server understands inside the free-form `settings`
/// dict on /profile-templates and /models/{id}/profiles. Mirrors
/// `UNIVERSAL_PROFILE_FIELDS` + `MODEL_SPECIFIC_PROFILE_FIELDS` from
/// `omlx/model_profiles.py`. Each key is snake_case; values are JSON-
/// compatible scalars.
///
/// The list is kept here (not derived from the snapshot) so a field
/// renamed on either side surfaces as a build break rather than a
/// silent drop on save.
enum ProfileSettingsKey {
    // Universal — both templates and model profiles
    static let modelAlias = "model_alias"
    static let maxContextWindow = "max_context_window"
    static let maxTokens = "max_tokens"
    static let temperature = "temperature"
    static let topP = "top_p"
    static let topK = "top_k"
    static let minP = "min_p"
    static let repetitionPenalty = "repetition_penalty"
    static let presencePenalty = "presence_penalty"
    static let ttlSeconds = "ttl_seconds"
    static let enableThinking = "enable_thinking"
    static let thinkingBudgetEnabled = "thinking_budget_enabled"
    static let thinkingBudgetTokens = "thinking_budget_tokens"
    static let reasoningParser = "reasoning_parser"
    static let maxToolResultTokens = "max_tool_result_tokens"
    static let forceSampling = "force_sampling"
    static let isPinned = "is_pinned"
    static let chatTemplateKwargs = "chat_template_kwargs"
    static let forcedCtKwargs = "forced_ct_kwargs"

    // Model-specific
    static let modelTypeOverride = "model_type_override"
    static let trustRemoteCode = "trust_remote_code"
    static let turboquantKvEnabled = "turboquant_kv_enabled"
    static let turboquantKvBits = "turboquant_kv_bits"
    static let indexCacheFreq = "index_cache_freq"
    static let specprefillEnabled = "specprefill_enabled"
    static let specprefillDraftModel = "specprefill_draft_model"
    static let specprefillKeepPct = "specprefill_keep_pct"
    static let specprefillThreshold = "specprefill_threshold"
    static let dflashEnabled = "dflash_enabled"
    static let dflashDraftModel = "dflash_draft_model"
    static let dflashDraftQuantEnabled = "dflash_draft_quant_enabled"
    static let dflashDraftQuantWeightBits = "dflash_draft_quant_weight_bits"
    static let dflashDraftQuantActivationBits = "dflash_draft_quant_activation_bits"
    static let dflashDraftQuantGroupSize = "dflash_draft_quant_group_size"
    static let dflashMaxCtx = "dflash_max_ctx"
    static let dflashVerifyMode = "dflash_verify_mode"
    static let dflashDraftWindowSize = "dflash_draft_window_size"
    static let dflashDraftSinkSize = "dflash_draft_sink_size"
    static let dflashInMemoryCache = "dflash_in_memory_cache"
    static let dflashInMemoryCacheMaxBytes = "dflash_in_memory_cache_max_bytes"
    static let dflashInMemoryCacheMaxEntries = "dflash_in_memory_cache_max_entries"
    static let dflashSsdCache = "dflash_ssd_cache"
    static let dflashSsdCacheMaxBytes = "dflash_ssd_cache_max_bytes"
    static let mtpEnabled = "mtp_enabled"
    static let vlmMtpEnabled = "vlm_mtp_enabled"
    static let vlmMtpDraftModel = "vlm_mtp_draft_model"
    static let vlmMtpDraftBlockSize = "vlm_mtp_draft_block_size"
}

/// Resolves the *display* scope for the model's currently active profile.
/// The server stores active state as a per-model profile reference, but the
/// user-visible scope follows the profile's `source_template` so applying
/// the "Balanced" preset chip lights up the Preset row, not the Model row.
///
/// - Parameters:
///   - activeName: server's `active_profile_name` for this model.
///   - modelProfiles: list of per-model profiles for this model.
///   - templates: list of profile templates (preset + global).
/// - Returns: scope + the display name (template name if source-template
///   resolves, profile name otherwise), or nil when no profile is active.
func resolveActiveProfileDisplay(
    activeName: String?,
    modelProfiles: [ProfileDTO],
    templates: [ProfileDTO]
) -> (scope: ProfileScope, name: String)? {
    guard let activeName, !activeName.isEmpty else { return nil }

    if let profile = modelProfiles.first(where: { $0.name == activeName }),
       let source = profile.sourceTemplate,
       let template = templates.first(where: { $0.name == source }) {
        return (template.templateScope, template.name)
    }
    if let template = templates.first(where: { $0.name == activeName }) {
        return (template.templateScope, template.name)
    }
    if modelProfiles.contains(where: { $0.name == activeName }) {
        return (.model, activeName)
    }
    return (.model, activeName)
}

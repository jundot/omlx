import SwiftUI

/// Sub-pages of the per-model settings screen. The top page (Basic and
/// Generation Defaults) is `nil` in `AppServices.modelDetailPage`.
enum ModelSettingsPage: Hashable, Sendable {
    case profiles, acceleration, memory, advanced, experimental

    var title: String {
        switch self {
        case .profiles:
            return String(localized: "settings.section.profiles",
                          defaultValue: "Profiles",
                          comment: "Title of the per-model Profiles page and its row")
        case .acceleration:
            return String(localized: "settings.acceleration.section",
                          defaultValue: "Acceleration",
                          comment: "Section header above the Acceleration settings group")
        case .memory:
            return String(localized: "settings.memory.section",
                          defaultValue: "Memory Saving",
                          comment: "Title of the per-model Memory Saving page and its row")
        case .advanced:
            return String(localized: "settings.advanced.section",
                          defaultValue: "Advanced Settings",
                          comment: "Section header above the Advanced tab fields")
        case .experimental:
            return String(localized: "settings.advanced.experimental.section",
                          defaultValue: "Experimental Features",
                          comment: "Section header above the Experimental settings group")
        }
    }
}

@MainActor
@Observable
final class ModelSettingsScreenVM {
    enum Field: Sendable {
        case alias, modelType, contextLength, maxTokens
        case temperature, topP, topK, minP
        case repetitionPenalty, presencePenalty, ttl
        case enableThinking, ssdNgramOffload
        case deepseekV41CedPrefill, qwen4GdnDecodeWideProj
        case thinkingBudgetEnabled, thinkingBudgetTokens
        case limitToolResults, toolResultLimitTokens
        case forceSampling, isPinned, isFavorite
        case trustRemoteCode, externalEngine
        case embeddingAudioEnabled, embeddingAudioMaxSeconds
        case reasoningParser
        case chatTemplateKwargs
        case turboquantKvEnabled, turboquantKvScheme, turboquantKvBits
        case qwen35AnePrefillSharedFraction
        case qwen35OqA8Enabled, qwen35OqA8MinTokens
        case qwen35AnePrefillEnabled, qwen35AnePrefillSequenceLength
        case qwen35AnePrefillTailPaddingMinTokens
        case qwen35AnePrefillFraction, qwen35AnePrefillMaxLayers
        case qwen35AnePrefillDualAne, qwen35AnePrefillGdn
        case qwen35AnePrefillGdnFraction, qwen35AnePrefillGdnMaxLayers
        case qwen35AnePrefillCpuEnabled, qwen35AnePrefillCpuFraction
        case qwen35AnePrefillCpuDownFraction
        case qwen35AnePrefillCpuGdnFraction
        case qwen35AnePrefillCpuThreads, qwen35AnePrefillCpuSharedResource
        case indexCacheEnabled, indexCacheFreq
        case specprefillEnabled, specprefillDraftModel, specprefillKeepPct, specprefillThreshold
        case dflashEnabled, dflashDraftModel
        case dflashDraftQuantEnabled, dflashDraftQuantWeightBits, dflashDraftQuantGroupSize
        case dflashVerifyMode, dflashDraftWindowSize, dflashBlockSize
        case mtpEnabled, mtpAdaptiveMaxDepth
        case vlmMtpEnabled, vlmMtpDraftModel, vlmMtpDraftBlockSize
    }

    static var modelTypeOptions: [(String, String)] {
        [
            ("", String(localized: "settings.model_type.auto_detect",
                        defaultValue: "Auto-detect",
                        comment: "Model type option meaning the server should auto-detect")),
            ("llm", String(localized: "settings.model_type.llm",
                           defaultValue: "LLM",
                           comment: "Model type option label for text language models")),
            ("vlm", String(localized: "settings.model_type.vlm",
                           defaultValue: "VLM",
                           comment: "Model type option label for vision-language models")),
            ("embedding", String(localized: "settings.model_type.embed",
                                 defaultValue: "Embedding",
                                 comment: "Model type option label for embedding models")),
            ("reranker", String(localized: "settings.model_type.rerank",
                                defaultValue: "Reranker",
                                comment: "Model type option label for reranker models")),
            ("audio_stt", String(localized: "settings.model_type.audio_stt",
                                 defaultValue: "Audio STT",
                                 comment: "Model type option label for speech-to-text models")),
            ("audio_tts", String(localized: "settings.model_type.audio_tts",
                                 defaultValue: "Audio TTS",
                                 comment: "Model type option label for text-to-speech models")),
            ("audio_sts", String(localized: "settings.model_type.audio_sts",
                                 defaultValue: "Audio STS",
                                 comment: "Model type option label for speech-to-speech models")),
            ("decision", String(localized: "settings.model_type.decision",
                                defaultValue: "Decision",
                                comment: "Model type option label for decision models served by /v1/systemone")),
        ]
    }

    static var turboquantKvSchemeOptions: [(String, String)] {
        [
            ("turboquant", "TurboQuant"),
            ("affine4", "Affine4 · 4-bit"),
            ("affine8", "Affine8 · 8-bit"),
        ]
    }

    static var turboquantKvBitsOptions: [(String, String)] {
        [
            ("2", String(localized: "settings.turboquant.bits.2",
                         defaultValue: "2-bit",
                         comment: "TurboQuant KV bit-width option")),
            ("2.5", String(localized: "settings.turboquant.bits.2_5",
                           defaultValue: "2.5-bit",
                           comment: "TurboQuant KV bit-width option")),
            ("3", String(localized: "settings.turboquant.bits.3",
                         defaultValue: "3-bit",
                         comment: "TurboQuant KV bit-width option")),
            ("3.5", String(localized: "settings.turboquant.bits.3_5",
                           defaultValue: "3.5-bit",
                           comment: "TurboQuant KV bit-width option")),
            ("4", String(localized: "settings.turboquant.bits.4",
                         defaultValue: "4-bit",
                         comment: "TurboQuant KV bit-width option")),
            ("6", String(localized: "settings.turboquant.bits.6",
                         defaultValue: "6-bit",
                         comment: "TurboQuant KV bit-width option")),
            ("8", String(localized: "settings.turboquant.bits.8",
                         defaultValue: "8-bit",
                         comment: "TurboQuant KV bit-width option")),
        ]
    }

    /// Keep-pct labels mirror the HTML editor's tradeoff annotations
    /// so the user picks an approximate speedup, not a raw fraction.
    static var specprefillKeepPctOptions: [(String, String)] {
        [
            ("0.1", String(localized: "settings.specprefill.keep.10",
                           defaultValue: "10% — Aggressive (~5-7x, some quality loss)",
                           comment: "SpecPrefill keep-rate dropdown option")),
            ("0.2", String(localized: "settings.specprefill.keep.20",
                           defaultValue: "20% — Balanced (~3x, recommended)",
                           comment: "SpecPrefill keep-rate dropdown option")),
            ("0.25", String(localized: "settings.specprefill.keep.25",
                            defaultValue: "25% — Conservative+ (~2.5x)",
                            comment: "SpecPrefill keep-rate dropdown option")),
            ("0.3", String(localized: "settings.specprefill.keep.30",
                           defaultValue: "30% — Conservative (~2.2x)",
                           comment: "SpecPrefill keep-rate dropdown option")),
            ("0.4", String(localized: "settings.specprefill.keep.40",
                           defaultValue: "40% — Mild (~1.8x)",
                           comment: "SpecPrefill keep-rate dropdown option")),
            ("0.5", String(localized: "settings.specprefill.keep.50",
                           defaultValue: "50% — Minimal (~1.5x)",
                           comment: "SpecPrefill keep-rate dropdown option")),
        ]
    }

    static var dflashDraftQuantWeightBitsOptions: [(String, String)] {
        [
            ("2", String(localized: "settings.dflash.quant.weight.2",
                         defaultValue: "2-bit",
                         comment: "DFlash draft quantization weight bits option")),
            ("4", String(localized: "settings.dflash.quant.weight.4",
                         defaultValue: "4-bit",
                         comment: "DFlash draft quantization weight bits option")),
            ("8", String(localized: "settings.dflash.quant.weight.8",
                         defaultValue: "8-bit",
                         comment: "DFlash draft quantization weight bits option")),
        ]
    }

    static var dflashDraftQuantGroupSizeOptions: [(String, String)] {
        [
            ("", String(localized: "settings.dflash.quant.group.default",
                        defaultValue: "default",
                        comment: "DFlash draft quantization group size — use server default")),
            ("32", "32"),
            ("64", "64"),
            ("128", "128"),
        ]
    }

    static var mtpDepthOptions: [(String, String)] {
        let adaptive = String(localized: "settings.acceleration.mtp.depth.adaptive",
                              defaultValue: "3 tokens (Default)",
                              comment: "Default Lightning MTP adaptive maximum draft depth")
        return [("3", adaptive)] + (4...6).map { depth in
            ("\(depth)", String(localized: "settings.acceleration.mtp.depth.option",
                                defaultValue: "\(depth) tokens",
                                comment: "Lightning MTP depth option; placeholder is the maximum draft token count"))
        }
    }

    static var dflashVerifyModeOptions: [(String, String)] {
        [
            ("", String(localized: "settings.dflash.verify_mode.default",
                        defaultValue: "default (adaptive)",
                        comment: "DFlash verify mode option meaning the server default is used")),
            ("adaptive", String(localized: "settings.dflash.verify_mode.adaptive",
                                defaultValue: "adaptive",
                                comment: "DFlash verify mode: shrinks block size when acceptance drops")),
            ("dflash", String(localized: "settings.dflash.verify_mode.dflash",
                              defaultValue: "dflash",
                              comment: "DFlash verify mode: standard dflash verifier")),
            ("off", String(localized: "settings.dflash.verify_mode.off",
                           defaultValue: "off",
                           comment: "DFlash verify mode: always verify the whole block")),
        ]
    }

    /// `config_model_type` values that surface IndexCache in the HTML
    /// admin. Mirrored from `dashboard.js:5-7` (`DSA_MODEL_TYPES`).
    static let dsaConfigModelTypes: Set<String> = [
        "deepseek_v32", "glm_moe_dsa",
    ]
    static let diffusionConfigModelTypes: Set<String> = [
        "diffusion_gemma",
    ]
    /// `config_model_type` values accepted by the HTML admin's VLM MTP
    /// assistant-drafter picker. Mirrored from `dashboard.js`
    /// (`VLM_MTP_DRAFTER_CONFIG_MODEL_TYPES`).
    static let vlmMtpDrafterConfigModelTypes: Set<String> = [
        "gemma4_assistant", "gemma4_unified_assistant", "qwen3_5_mtp",
    ]
    static let diffusionUnsupportedCtKwargKeys: Set<String> = [
        "enable_thinking", "reasoning_effort", "preserve_thinking",
    ]

    /// Web admin input range for the resident expert share, in percent.
    static let moeResidentPercentRange: ClosedRange<Double> = 5...95

    var model: ModelDTO?
    /// Snapshot of every other model on the server, used to populate the
    /// SpecPrefill / DFlash draft-model dropdowns. Reloaded with `load()`.
    var allModels: [ModelDTO] = []
    var modelID: String = ""
    var lastError: String?
    var isImportingMtplxSidecar: Bool = false

    // Basic
    var alias: String = ""
    var modelTypeOverride: String = ""
    var contextLength: String = ""
    var maxTokens: String = ""
    var temperature: String = ""
    var topP: String = ""
    var topK: String = ""
    var minP: String = ""
    var repetitionPenalty: String = ""
    var presencePenalty: String = ""
    var ttlSeconds: String = ""

    // Advanced
    var enableThinking: Bool = true
    var qwen4PleSsdOffload: Bool = false
    var qwen4PleSsdOffloadSupported: Bool = false
    var qwen4PleSsdOffloadForced: Bool = false
    var deepseekV41EngramSsdOffload: Bool = false
    var deepseekV41EngramSsdOffloadSupported: Bool = false
    var deepseekV41EngramSsdOffloadForced: Bool = false
    var deepseekV41CedPrefillEnabled: Bool = false
    var qwen4GdnDecodeWideProj: Bool = false
    var thinkingBudgetEnabled: Bool = false
    var thinkingBudgetTokens: String = "8192"
    var limitToolResults: Bool = false
    /// Token cap when `limitToolResults` is on. Defaults to the HTML
    /// admin's seeded value so the first save after enabling sends a
    /// sensible number instead of zero (which the server interprets as
    /// "disabled").
    var toolResultLimitTokens: String = "4096"
    var forceSampling: Bool = false
    var isPinned: Bool = false
    var isFavorite: Bool = false

    // Security
    var trustRemoteCode: Bool = false

    /// External engine serving the model; empty runs it in-process.
    var externalEngine: String = ""

    // Embedding audio tower. Empty seconds keeps the 30 s processor default.
    var embeddingAudioEnabled: Bool = false
    var embeddingAudioMaxSeconds: String = ""

    // Reasoning parser (free-form override; empty = auto)
    var reasoningParser: String = ""

    var guidedGrammarEnabled: Bool = false
    var guidedGrammar: String = ""

    var moeExpertOffloadEnabled: Bool = false
    /// Resident expert share in percent, as typed. The wire value is a fraction.
    var moeExpertOffloadResidentPercent: String = "25"
    /// Server fraction at the last load. An unchanged field sends it back even
    /// when it is outside the UI range, as the web admin does.
    private var loadedMoeResidentFraction: Double = 0.25

    // Chat-template kwargs — entries are the editor's view of the
    // (chat_template_kwargs, forced_ct_kwargs) server pair.
    var chatTemplateEntries: [ChatTemplateKwargEntry] = []

    // Experimental: TurboQuant KV
    var turboquantKvEnabled: Bool = false
    var turboquantKvScheme: String = "turboquant"
    var turboquantKvBits: String = "4"

    // Experimental: oQ mixed-bit INT8-activation prefill kernels. There is no
    // layout to choose: the kernel reads the checkpoint's own packed weight
    // stream, so it is both the fastest option and the one that costs no extra
    // memory. The tile is picked per bit width by the dispatcher.
    var qwen35OqA8Enabled: Bool = false
    var qwen35OqA8MinTokens: String = "128"

    // Experimental: private Qwen3.5/3.6/3.8 ANE/GPU fixed-shape prefill.
    // These defaults are the measured M3 Ultra optimum for the 2,048-token
    // benchmark path. The feature itself remains opt-in.
    var qwen35AnePrefillSharedFraction = "1"
    var qwen35AnePrefillEnabled: Bool = false
    var qwen35AnePrefillSequenceLength: String = "2048"
    var qwen35AnePrefillTailPaddingMinTokens: String = "0"
    var qwen35AnePrefillFraction: String = "0.53"
    var qwen35AnePrefillMaxLayers: String = "64"
    var qwen35AnePrefillDualAne: Bool = true
    var qwen35AnePrefillGdn: Bool = true
    var qwen35AnePrefillGdnFraction: String = "0.5"
    var qwen35AnePrefillGdnMaxLayers: String = "48"
    var qwen35AnePrefillCpuEnabled: Bool = false
    var qwen35AnePrefillCpuFraction: String = "0.135"
    var qwen35AnePrefillCpuDownFraction: String = "0"
    var qwen35AnePrefillCpuGdnFraction: String = "0"
    var qwen35AnePrefillCpuThreads: String = "8"
    var qwen35AnePrefillCpuSharedResource: Bool = true
    var qwen35AnePrefillFusedDown: Bool = false
    var aneTuningID: String?
    var aneTuningIsRunning: Bool = false
    var aneTuningStatus: ANETuningStatusResponse?
    var aneTuningAllowCPU: Bool = true
    var aneTuningAllowCPUGate: Bool = true
    var aneTuningAllowCPUDown: Bool = true
    var aneTuningAllowANEGDN: Bool = true
    var aneTuningAllowCPUGDN: Bool = true
    var aneTuningAllowCPUSharedResource: Bool = true

    // Header snapshot actions: reset / optimal (omlx.ai) / custom recipe.
    var isApplyingSettings: Bool = false
    var isLoadingGenerationDefaults: Bool = false
    var pendingReset: Bool = false
    var applyOutcome: SettingsApplyOutcome?
    /// Error shown inside the snapshot sheet so the user can retry.
    var applyError: String?

    // Experimental: IndexCache (DSA-only)
    var indexCacheEnabled: Bool = false
    var indexCacheFreq: String = "4"

    // Experimental: SpecPrefill
    var specprefillEnabled: Bool = false
    var specprefillDraftModel: String = ""
    var specprefillKeepPct: String = "0.2"
    var specprefillThreshold: String = "8192"

    // Experimental: DFlash
    var dflashEnabled: Bool = false
    var dflashDraftModel: String = ""
    var dflashDraftQuantEnabled: Bool = false
    var dflashDraftQuantWeightBits: String = "4"
    var dflashDraftQuantGroupSize: String = ""
    var dflashVerifyMode: String = ""
    var dflashDraftWindowSize: String = ""
    var dflashBlockSize: String = ""

    // Experimental: native MTP
    var mtpEnabled: Bool = false
    /// Empty = adaptive depth.
    var mtpAdaptiveMaxDepth: String = "3"

    // Experimental: VLM MTP (assistant-drafter speculative decoding for VLMs).
    // Block size is held as a string for the editor; empty = mlx-vlm default.
    var vlmMtpEnabled: Bool = false
    var vlmMtpDraftModel: String = ""
    var vlmMtpDraftBlockSize: String = ""

    // Profiles
    var profiles: [ProfileDTO] = []
    var templates: [ProfileDTO] = []
    var activeProfileName: String?
    /// Server's `GlobalSettings.sampling` snapshot, loaded alongside the
    /// per-model settings so the Profiles tab's "Server Defaults" card
    /// can render without a second round-trip.
    var serverDefaultSampling: GlobalSettingsDTO.SamplingDTO?
    /// Display scope for the active profile (derived from `source_template`).
    var activeProfileScope: ProfileScope = .model
    /// True when profile-eligible values differ from the last load / apply /
    /// save, until the pending auto-save writes them. Per-model fields
    /// (alias / modelType / ttl / isPinned / trustRemoteCode) save on commit
    /// and never set this.
    var profileDirty: Bool = false
    /// Profile values at the last load; nil before the first one.
    private var loadedSettings: [String: AnyCodable]?

    /// Pause after the last edit before profile fields save, so typing in a
    /// number field does not write every keystroke.
    @ObservationIgnored var autoSaveDelay: Duration = .milliseconds(800)
    @ObservationIgnored private var autoSaveClient: OMLXClient?
    @ObservationIgnored private var autoSaveTimer: Task<Void, Never>?
    @ObservationIgnored private var autoSaveWrite: Task<Void, Never>?

    /// Record the current values as the clean, loaded state.
    func resetWorkingBaseline() {
        loadedSettings = currentSettingsDict()
        profileDirty = false
    }

    /// State machine the banner and ProfileDetailCard render against.
    /// Cheap to recompute — pure function of (profileDirty, activeProfileScope,
    /// activeProfileName).
    var activeProfileState: ActiveProfileState {
        if profileDirty {
            if let name = activeProfileName {
                return .working(basedOn: .init(scope: activeProfileScope, name: name))
            }
            return .working(basedOn: nil)
        }
        if let name = activeProfileName {
            return .named(scope: activeProfileScope, name: name)
        }
        return .defaults
    }

    func profileDisplayName(scope: ProfileScope, name: String) -> String {
        let collection = scope == .model ? profiles : templates
        return collection.first(where: { $0.name == name })?.displayName ?? name
    }

    var displayProfileState: ActiveProfileState {
        switch activeProfileState {
        case .named(let scope, let name):
            return .named(scope: scope, name: profileDisplayName(scope: scope, name: name))
        case .working(let basedOn):
            return .working(basedOn: basedOn.map {
                .init(scope: $0.scope, name: profileDisplayName(scope: $0.scope, name: $0.name))
            })
        case .defaults: return .defaults
        }
    }

    private var normalizedConfigModelType: String {
        (model?.configModelType ?? "")
            .lowercased()
            .replacingOccurrences(of: "-", with: "_")
    }

    var isDiffusionModel: Bool {
        Self.diffusionConfigModelTypes.contains(normalizedConfigModelType)
    }

    /// LLM gate of the web admin: an unknown model type counts as an LLM.
    var isLLMLike: Bool {
        guard let type = model?.modelType, !type.isEmpty else { return true }
        return type == "llm" || type == "vlm"
    }

    var showsGenerationSection: Bool { isLLMLike }

    /// Embedding requests without max_length truncate input to the context window.
    var showsEmbeddingContextWindow: Bool { model?.modelType == "embedding" }

    /// Acceleration, Memory Saving and Experimental only apply to the MLX
    /// LLM engines.
    var showsEnginePages: Bool { isLLMLike && !isDiffusionModel }

    var thinkingForced: Bool { model?.thinkingForced == true }

    static func aneFractionOptions(current: String, presets: [Double]) -> [(String, String)] {
        let value = Double(current)
        var options = presets.map { ($0 == value ? current : String($0), $0.formatted(.percent.precision(.fractionLength(0)))) }
        if let value, !presets.contains(value) {
            options.insert((current, value.formatted(.percent.precision(.fractionLength(0...2)))), at: 0)
        }
        return options
    }

    var reasoningEffortPresets: [String] {
        model?.reasoningEffortOptions ?? []
    }

    var isQwen4Exp: Bool { normalizedConfigModelType == "qwen4_exp" }

    var isDeepseekV41: Bool { normalizedConfigModelType == "deepseek_v41" }

    /// One SSD N-gram Offload switch serves Qwen4 PLE and DeepSeek V4.1
    /// Engram. Qwen4 is checked first, as in the web admin.
    private var usesQwen4PleOffload: Bool { isQwen4Exp && qwen4PleSsdOffloadSupported }

    var ssdNgramOffloadSupported: Bool {
        usesQwen4PleOffload || (isDeepseekV41 && deepseekV41EngramSsdOffloadSupported)
    }

    var ssdNgramOffloadForced: Bool {
        usesQwen4PleOffload ? qwen4PleSsdOffloadForced : deepseekV41EngramSsdOffloadForced
    }

    var ssdNgramOffload: Bool {
        get { usesQwen4PleOffload ? qwen4PleSsdOffload : deepseekV41EngramSsdOffload }
        set {
            if usesQwen4PleOffload {
                qwen4PleSsdOffload = newValue
            } else {
                deepseekV41EngramSsdOffload = newValue
            }
        }
    }

    var moeExpertOffloadSupported: Bool { model?.moeExpertOffloadSupported == true }

    /// Wire fraction for the resident share, or nil when an edited value is
    /// outside the UI range.
    var moeExpertOffloadResidentFraction: Double? {
        let text = moeExpertOffloadResidentPercent.trimmingCharacters(in: .whitespaces)
        if text == Self.formatPct(loadedMoeResidentFraction * 100) {
            return loadedMoeResidentFraction
        }
        guard let percent = Double(text), percent.isFinite,
              Self.moeResidentPercentRange.contains(percent) else { return nil }
        return percent / 100
    }

    var moeResidentRangeError: String {
        String(localized: "settings.memory.moe_offload.resident.range_error",
               defaultValue: "Enter a number from 5 to 95.",
               comment: "Error shown when the MoE resident expert percent is outside 5-95")
    }

    var embeddingAudioSupported: Bool { model?.embeddingAudioSupported == true }

    private func isDiffusionUnsupportedField(_ field: Field) -> Bool {
        switch field {
        case .topP, .topK, .minP, .repetitionPenalty, .presencePenalty:
            return true
        case .enableThinking, .ssdNgramOffload,
             .deepseekV41CedPrefill, .qwen4GdnDecodeWideProj,
             .thinkingBudgetEnabled, .thinkingBudgetTokens:
            return true
        case .limitToolResults, .toolResultLimitTokens:
            return true
        case .forceSampling, .reasoningParser:
            return true
        case .turboquantKvEnabled, .turboquantKvScheme, .turboquantKvBits:
            return true
        case .qwen35AnePrefillSharedFraction,
             .qwen35OqA8Enabled, .qwen35OqA8MinTokens:
            return true
        case .qwen35AnePrefillEnabled, .qwen35AnePrefillSequenceLength,
             .qwen35AnePrefillTailPaddingMinTokens:
            return true
        case .qwen35AnePrefillFraction, .qwen35AnePrefillMaxLayers:
            return true
        case .qwen35AnePrefillDualAne, .qwen35AnePrefillGdn:
            return true
        case .qwen35AnePrefillGdnFraction, .qwen35AnePrefillGdnMaxLayers:
            return true
        case .qwen35AnePrefillCpuEnabled, .qwen35AnePrefillCpuFraction,
             .qwen35AnePrefillCpuDownFraction, .qwen35AnePrefillCpuGdnFraction:
            return true
        case .qwen35AnePrefillCpuThreads, .qwen35AnePrefillCpuSharedResource:
            return true
        case .indexCacheEnabled, .indexCacheFreq:
            return true
        case .specprefillEnabled, .specprefillDraftModel:
            return true
        case .specprefillKeepPct, .specprefillThreshold:
            return true
        case .dflashEnabled, .dflashDraftModel:
            return true
        case .dflashDraftQuantEnabled, .dflashDraftQuantWeightBits, .dflashDraftQuantGroupSize:
            return true
        case .dflashVerifyMode, .dflashDraftWindowSize, .dflashBlockSize:
            return true
        case .mtpEnabled, .mtpAdaptiveMaxDepth, .vlmMtpEnabled, .vlmMtpDraftModel:
            return true
        case .vlmMtpDraftBlockSize:
            return true
        case .alias, .modelType, .contextLength, .maxTokens:
            return false
        case .temperature, .ttl, .isPinned, .isFavorite, .trustRemoteCode:
            return false
        case .externalEngine:
            return true
        case .embeddingAudioEnabled, .embeddingAudioMaxSeconds:
            return false
        case .chatTemplateKwargs:
            return false
        }
    }

    private func diffusionCompatibleChatTemplateEntries(
        _ entries: [ChatTemplateKwargEntry]
    ) -> [ChatTemplateKwargEntry] {
        guard isDiffusionModel else { return entries }
        return entries.filter { entry in
            guard let key = entry.resolvedKey else { return true }
            return !Self.diffusionUnsupportedCtKwargKeys.contains(key)
        }
    }

    func bind<T: Equatable>(
        _ binding: Binding<T>,
        save: @escaping () -> Void
    ) -> Binding<T> {
        Binding(
            get: { binding.wrappedValue },
            set: { newValue in
                let changed = binding.wrappedValue != newValue
                binding.wrappedValue = newValue
                if changed { save() }
            }
        )
    }

    /// Binding helper for profile-eligible fields. An edit updates
    /// `profileDirty` and schedules one auto-save of all profile fields
    /// instead of a per-field PUT.
    func bindProfile<T: Equatable>(_ binding: Binding<T>) -> Binding<T> {
        Binding(
            get: { binding.wrappedValue },
            set: { newValue in
                let changed = binding.wrappedValue != newValue
                binding.wrappedValue = newValue
                if changed { self.refreshProfileDirty() }
            }
        )
    }

    /// Compare the working values with the last load and schedule an
    /// auto-save when they differ.
    func refreshProfileDirty() {
        profileDirty = loadedSettings.map { currentSettingsDict() != $0 } ?? true
        if profileDirty { scheduleAutoSave() }
    }

    /// Debounced write of the profile fields. Writes run one at a time, and
    /// the task keeps the VM alive so leaving the screen does not drop it.
    private func scheduleAutoSave() {
        guard let client = autoSaveClient else { return }
        autoSaveTimer?.cancel()
        autoSaveTimer = Task {
            try? await Task.sleep(for: autoSaveDelay)
            guard !Task.isCancelled else { return }
            await autoSaveWrite?.value
            guard !Task.isCancelled, profileDirty else { return }
            let write = Task { await saveWorkingToModel(client: client) }
            autoSaveWrite = write
            await write.value
        }
    }

    /// Wait for a scheduled auto-save, including its write.
    func waitForAutoSave() async {
        await autoSaveTimer?.value
    }

    private var loadSequence = 0

    func load(modelID: String, client: OMLXClient, preservingEdits: Bool = false) async {
        autoSaveClient = client
        if preservingEdits && profileDirty { return }
        loadSequence += 1
        let sequence = loadSequence
        if self.modelID != modelID {
            aneTuningID = nil
            aneTuningIsRunning = false
            aneTuningStatus = nil
        }
        self.modelID = modelID
        do {
            let models = try await client.listModels().models
            let profiles = try await client.listModelProfiles(id: modelID).profiles
            let templates = try await client.listProfileTemplates().templates
            let defaults = try await client.getGlobalSettings().sampling
            guard sequence == loadSequence else { return }
            if preservingEdits && profileDirty { return }
            self.allModels = models
            if let m = models.first(where: { $0.id == modelID }) {
                self.model = m
                // A never-customized model has no settings record on the
                // server, so `settings` arrives nil. Repopulate with the
                // defaults anyway; skipping would leave edited values on
                // screen after Discard (#2182).
                let s = m.settings
                self.alias = s?.modelAlias ?? ""
                self.modelTypeOverride = s?.modelTypeOverride ?? ""
                self.contextLength = s?.maxContextWindow.map(String.init) ?? ""
                self.maxTokens = s?.maxTokens.map(String.init) ?? ""
                self.temperature = s?.temperature.map { String($0) } ?? ""
                self.topP = s?.topP.map { String($0) } ?? ""
                self.topK = s?.topK.map(String.init) ?? ""
                self.minP = s?.minP.map { String($0) } ?? ""
                self.repetitionPenalty = s?.repetitionPenalty.map { String($0) } ?? ""
                self.presencePenalty = s?.presencePenalty.map { String($0) } ?? ""
                self.ttlSeconds = s?.ttlSeconds.map(String.init) ?? ""
                self.enableThinking = s?.enableThinking ?? true
                self.qwen4PleSsdOffloadForced =
                    m.qwen4PleSsdOffloadForced ?? false
                self.qwen4PleSsdOffloadSupported =
                    m.qwen4PleSsdOffloadSupported ?? false
                self.qwen4PleSsdOffload = self.qwen4PleSsdOffloadForced
                    || (s?.qwen4PleSsdOffload ?? false)
                self.deepseekV41EngramSsdOffloadForced =
                    m.deepseekV41EngramSsdOffloadForced ?? false
                self.deepseekV41EngramSsdOffloadSupported =
                    m.deepseekV41EngramSsdOffloadSupported ?? false
                self.deepseekV41EngramSsdOffload = self.deepseekV41EngramSsdOffloadForced
                    || (s?.deepseekV41EngramSsdOffload ?? false)
                self.deepseekV41CedPrefillEnabled = s?.deepseekV41CedPrefillEnabled ?? false
                self.qwen4GdnDecodeWideProj = s?.qwen4GdnDecodeWideProj ?? false
                self.thinkingBudgetEnabled = s?.thinkingBudgetEnabled ?? false
                self.thinkingBudgetTokens = s?.thinkingBudgetTokens.map(String.init) ?? "8192"
                self.limitToolResults = (s?.maxToolResultTokens ?? 0) > 0
                if let n = s?.maxToolResultTokens, n > 0 {
                    self.toolResultLimitTokens = String(n)
                } else {
                    self.toolResultLimitTokens = "4096"
                }
                self.forceSampling = s?.forceSampling ?? false
                self.isPinned = s?.isPinned ?? false
                self.isFavorite = s?.isFavorite ?? false
                self.trustRemoteCode = s?.trustRemoteCode ?? false
                self.externalEngine = s?.externalEngine ?? ""
                self.embeddingAudioEnabled = s?.embeddingAudioEnabled ?? false
                self.embeddingAudioMaxSeconds = s?.embeddingAudioMaxSeconds.map { Self.formatPct($0) } ?? ""
                self.reasoningParser = s?.reasoningParser ?? ""
                self.guidedGrammarEnabled = s?.guidedGrammarEnabled ?? false
                self.guidedGrammar = s?.guidedGrammar ?? ""
                self.moeExpertOffloadEnabled = m.moeExpertOffloadSupported == true
                    && (s?.moeExpertOffloadEnabled ?? false)
                self.loadedMoeResidentFraction = s?.moeExpertOffloadResidentFraction ?? 0.25
                self.moeExpertOffloadResidentPercent = Self.formatPct(self.loadedMoeResidentFraction * 100)
                self.chatTemplateEntries = diffusionCompatibleChatTemplateEntries(
                    ChatTemplateKwargsCodec.decode(
                        kwargs: s?.chatTemplateKwargs,
                        forced: s?.forcedCtKwargs
                    )
                )
                self.turboquantKvEnabled = s?.turboquantKvEnabled ?? false
                self.turboquantKvScheme = s?.turboquantKvScheme ?? "turboquant"
                self.turboquantKvBits = s?.turboquantKvBits.map { Self.formatBits($0) } ?? "4"
                self.qwen35AnePrefillSharedFraction = s?.qwen35AnePrefillSharedFraction.map { String($0) } ?? "1"
                self.qwen35OqA8Enabled = s?.qwen35OqA8Enabled ?? false
                self.qwen35OqA8MinTokens = s?.qwen35OqA8MinTokens.map(String.init) ?? "128"
                self.qwen35AnePrefillEnabled = s?.qwen35AnePrefillEnabled ?? false
                self.qwen35AnePrefillSequenceLength = s?.qwen35AnePrefillSequenceLength.map(String.init) ?? "2048"
                self.qwen35AnePrefillTailPaddingMinTokens = s?.qwen35AnePrefillTailPaddingMinTokens.map(String.init) ?? "0"
                self.qwen35AnePrefillFraction = String(s?.qwen35AnePrefillFraction ?? m.anePrefillDefaultFraction ?? 0.53)
                self.qwen35AnePrefillMaxLayers = s?.qwen35AnePrefillMaxLayers.map(String.init) ?? "64"
                self.qwen35AnePrefillDualAne = s?.qwen35AnePrefillDualAne ?? true
                self.qwen35AnePrefillGdn = s?.qwen35AnePrefillGdn ?? true
                self.qwen35AnePrefillGdnFraction = s?.qwen35AnePrefillGdnFraction.map { Self.formatPct($0) } ?? "0.5"
                self.qwen35AnePrefillGdnMaxLayers = s?.qwen35AnePrefillGdnMaxLayers.map(String.init) ?? "48"
                self.qwen35AnePrefillCpuEnabled = s?.qwen35AnePrefillCpuEnabled ?? false
                self.qwen35AnePrefillCpuFraction = s?.qwen35AnePrefillCpuFraction.map { Self.formatPct($0) } ?? "0.135"
                self.qwen35AnePrefillCpuDownFraction = s?.qwen35AnePrefillCpuDownFraction.map { Self.formatPct($0) } ?? "0"
                self.qwen35AnePrefillCpuGdnFraction = s?.qwen35AnePrefillCpuGdnFraction.map { Self.formatPct($0) } ?? "0"
                self.qwen35AnePrefillCpuThreads = s?.qwen35AnePrefillCpuThreads.map(String.init) ?? "8"
                self.qwen35AnePrefillCpuSharedResource = s?.qwen35AnePrefillCpuSharedResource ?? true
                self.qwen35AnePrefillFusedDown = s?.qwen35AnePrefillFusedDown ?? false
                self.indexCacheEnabled = s?.indexCacheFreq != nil
                self.indexCacheFreq = s?.indexCacheFreq.map(String.init) ?? "4"
                self.specprefillEnabled = s?.specprefillEnabled ?? false
                self.specprefillDraftModel = s?.specprefillDraftModel ?? ""
                self.specprefillKeepPct = s?.specprefillKeepPct.map { Self.formatPct($0) } ?? "0.2"
                self.specprefillThreshold = s?.specprefillThreshold.map(String.init) ?? "8192"
                self.dflashEnabled = s?.dflashEnabled ?? false
                self.dflashDraftModel = s?.dflashDraftModel ?? ""
                self.dflashDraftQuantEnabled = s?.dflashDraftQuantEnabled ?? false
                self.dflashDraftQuantWeightBits = s?.dflashDraftQuantWeightBits.map(String.init) ?? "4"
                self.dflashDraftQuantGroupSize = s?.dflashDraftQuantGroupSize.map(String.init) ?? ""
                self.dflashVerifyMode = s?.dflashVerifyMode ?? ""
                self.dflashDraftWindowSize = s?.dflashDraftWindowSize.map(String.init) ?? ""
                self.dflashBlockSize = s?.dflashBlockSize.map(String.init) ?? ""
                self.mtpEnabled = s?.mtpEnabled ?? false
                self.mtpAdaptiveMaxDepth = s?.mtpAdaptiveMaxDepth.flatMap { (3...6).contains($0) ? String($0) : nil } ?? "3"
                self.vlmMtpEnabled = s?.vlmMtpEnabled ?? false
                self.vlmMtpDraftModel = s?.vlmMtpDraftModel ?? ""
                self.vlmMtpDraftBlockSize = s?.vlmMtpDraftBlockSize.map(String.init) ?? ""
                self.activeProfileName = s?.activeProfileName
            }
            self.profiles = profiles
            self.templates = templates
            self.serverDefaultSampling = defaults
            // Resolve display scope from the source_template of the active
            // model profile (if any) — so applying the "Balanced" preset
            // lights up the Preset chip, not the local model copy.
            if let display = resolveActiveProfileDisplay(
                activeName: self.activeProfileName,
                modelProfiles: self.profiles,
                templates: self.templates
            ) {
                self.activeProfileScope = display.scope
                self.activeProfileName = display.name
            } else {
                self.activeProfileScope = .model
                self.activeProfileName = nil
            }
            // Reload always re-establishes the baseline.
            resetWorkingBaseline()
            self.lastError = nil
        } catch {
            self.lastError = error.omlxDescription
        }
    }

    func save(_ field: Field, client: OMLXClient) async {
        if isDiffusionModel && isDiffusionUnsupportedField(field) {
            return
        }
        var patch = ModelSettingsPatch()
        switch field {
        case .alias:                   patch.modelAlias = alias.isEmpty ? nil : alias
        case .modelType:               patch.modelTypeOverride = modelTypeOverride.isEmpty ? nil : modelTypeOverride
        case .contextLength:           patch.maxContextWindow = Int(contextLength)
        case .maxTokens:               patch.maxTokens = Int(maxTokens)
        case .temperature:
            switch SamplingValidator.temperature(temperature) {
            case .success(let v): patch.temperature = v
            case .failure(let e): self.lastError = e.message; return
            }
        case .topP:
            switch SamplingValidator.topP(topP) {
            case .success(let v): patch.topP = v
            case .failure(let e): self.lastError = e.message; return
            }
        case .topK:
            switch SamplingValidator.topK(topK) {
            case .success(let v): patch.topK = v
            case .failure(let e): self.lastError = e.message; return
            }
        case .minP:
            switch SamplingValidator.minP(minP) {
            case .success(let v): patch.minP = v
            case .failure(let e): self.lastError = e.message; return
            }
        case .repetitionPenalty:
            switch SamplingValidator.penalty(repetitionPenalty,
                                             name: String(localized: "settings.validator.repetition_penalty.name",
                                                          defaultValue: "Repetition Penalty",
                                                          comment: "Field name embedded in validation errors for repetition penalty")) {
            case .success(let v): patch.repetitionPenalty = v
            case .failure(let e): self.lastError = e.message; return
            }
        case .presencePenalty:
            switch SamplingValidator.penalty(presencePenalty,
                                             name: String(localized: "settings.validator.presence_penalty.name",
                                                          defaultValue: "Presence Penalty",
                                                          comment: "Field name embedded in validation errors for presence penalty")) {
            case .success(let v): patch.presencePenalty = v
            case .failure(let e): self.lastError = e.message; return
            }
        case .ttl:                     patch.ttlSeconds = Int(ttlSeconds)
        case .enableThinking:          patch.enableThinking = enableThinking
        case .ssdNgramOffload:
            // A forced offload is a runtime decision, so sending nothing
            // keeps the stored choice for when the model fits again.
            guard ssdNgramOffloadSupported, !ssdNgramOffloadForced else { return }
            if usesQwen4PleOffload {
                patch.qwen4PleSsdOffload = qwen4PleSsdOffload
            } else {
                patch.deepseekV41EngramSsdOffload = deepseekV41EngramSsdOffload
            }
        case .deepseekV41CedPrefill:
            guard isDeepseekV41 else { return }
            patch.deepseekV41CedPrefillEnabled = deepseekV41CedPrefillEnabled
        case .qwen4GdnDecodeWideProj:
            guard isQwen4Exp else { return }
            patch.qwen4GdnDecodeWideProj = qwen4GdnDecodeWideProj
        case .thinkingBudgetEnabled:   patch.thinkingBudgetEnabled = thinkingBudgetEnabled
        case .thinkingBudgetTokens:    patch.thinkingBudgetTokens = Int(thinkingBudgetTokens)
        case .limitToolResults:
            // Toggling on resends the current token count (or the default);
            // toggling off sends 0 — the server's documented "disable" sentinel.
            if limitToolResults {
                patch.maxToolResultTokens = Int(toolResultLimitTokens) ?? 4096
            } else {
                patch.maxToolResultTokens = 0
            }
        case .toolResultLimitTokens:
            // Only saved while the toggle is on; a blank/non-numeric value
            // is silently ignored to match the HTML editor's behavior.
            guard limitToolResults else { return }
            guard let n = Int(toolResultLimitTokens), n > 0 else { return }
            patch.maxToolResultTokens = n
        case .forceSampling:           patch.forceSampling = forceSampling
        case .isPinned:                patch.isPinned = isPinned
        case .isFavorite:              patch.isFavorite = isFavorite
        case .trustRemoteCode:         patch.trustRemoteCode = trustRemoteCode
        case .externalEngine:
            patch.externalEngine = .some(externalEngine.isEmpty ? nil : externalEngine)
        case .embeddingAudioEnabled:   patch.embeddingAudioEnabled = embeddingAudioEnabled
        case .embeddingAudioMaxSeconds:
            let text = embeddingAudioMaxSeconds.trimmingCharacters(in: .whitespaces)
            if text.isEmpty {
                patch.embeddingAudioMaxSeconds = .some(nil)
            } else if let seconds = Double(text), seconds > 0 {
                patch.embeddingAudioMaxSeconds = seconds
            } else {
                lastError = "Max audio length must be a positive number of seconds."
                return
            }
        case .reasoningParser:
            patch.reasoningParser = reasoningParser.isEmpty ? nil : reasoningParser
        case .chatTemplateKwargs:
            let pair = ChatTemplateKwargsCodec.encode(
                diffusionCompatibleChatTemplateEntries(chatTemplateEntries)
            )
            patch.chatTemplateKwargs = pair.kwargs ?? [:]
            patch.forcedCtKwargs = pair.forced ?? []
        case .turboquantKvEnabled:     patch.turboquantKvEnabled = turboquantKvEnabled
        case .turboquantKvScheme:
            patch.turboquantKvScheme = turboquantKvScheme
            if turboquantKvScheme == "affine4" {
                turboquantKvBits = "4"
                patch.turboquantKvBits = 4
            } else if turboquantKvScheme == "affine8" {
                turboquantKvBits = "8"
                patch.turboquantKvBits = 8
            }
        case .turboquantKvBits:        patch.turboquantKvBits = Double(turboquantKvBits)
        case .qwen35AnePrefillSharedFraction:
            guard validateAneWorkingSettings() else { return }
            patch.qwen35AnePrefillSharedFraction = Double(qwen35AnePrefillSharedFraction)
        case .qwen35OqA8Enabled:  patch.qwen35OqA8Enabled = qwen35OqA8Enabled
        case .qwen35OqA8MinTokens:
            guard let value = Int(qwen35OqA8MinTokens), value >= 1 else {
                lastError = "oQ A8 minimum prompt tokens must be a positive integer."
                return
            }
            patch.qwen35OqA8MinTokens = value
        case .qwen35AnePrefillEnabled: patch.qwen35AnePrefillEnabled = qwen35AnePrefillEnabled
        case .qwen35AnePrefillSequenceLength:
            guard validateAneWorkingSettings() else { return }
            patch.qwen35AnePrefillSequenceLength = Int(qwen35AnePrefillSequenceLength)
        case .qwen35AnePrefillTailPaddingMinTokens:
            switch QwenAneSettingsValidator.tailPadding(
                qwen35AnePrefillTailPaddingMinTokens,
                sequenceLength: qwen35AnePrefillSequenceLength
            ) {
            case .success(let value): patch.qwen35AnePrefillTailPaddingMinTokens = value
            case .failure(let error): lastError = error.message; return
            }
        case .qwen35AnePrefillFraction:
            guard validateAneWorkingSettings() else { return }
            patch.qwen35AnePrefillFraction = Double(qwen35AnePrefillFraction)
        case .qwen35AnePrefillMaxLayers:
            switch QwenAneSettingsValidator.mlpLayers(qwen35AnePrefillMaxLayers) {
            case .success(let value): patch.qwen35AnePrefillMaxLayers = value
            case .failure(let error): lastError = error.message; return
            }
        case .qwen35AnePrefillDualAne: patch.qwen35AnePrefillDualAne = qwen35AnePrefillDualAne
        case .qwen35AnePrefillGdn:     patch.qwen35AnePrefillGdn = qwen35AnePrefillGdn
        case .qwen35AnePrefillGdnFraction:
            switch QwenAneSettingsValidator.gdnFraction(
                qwen35AnePrefillGdnFraction,
                cpuFraction: qwen35AnePrefillCpuEnabled ? qwen35AnePrefillCpuGdnFraction : "0"
            ) {
            case .success(let value): patch.qwen35AnePrefillGdnFraction = value
            case .failure(let error): lastError = error.message; return
            }
        case .qwen35AnePrefillGdnMaxLayers:
            switch QwenAneSettingsValidator.gdnLayers(qwen35AnePrefillGdnMaxLayers) {
            case .success(let value): patch.qwen35AnePrefillGdnMaxLayers = value
            case .failure(let error): lastError = error.message; return
            }
        case .qwen35AnePrefillCpuEnabled:
            patch.qwen35AnePrefillCpuEnabled = qwen35AnePrefillCpuEnabled
        case .qwen35AnePrefillCpuFraction:
            switch QwenAneSettingsValidator.cpuFraction(
                qwen35AnePrefillCpuFraction,
                mlpFraction: qwen35AnePrefillFraction
            ) {
            case .success(let value): patch.qwen35AnePrefillCpuFraction = value
            case .failure(let error): lastError = error.message; return
            }
        case .qwen35AnePrefillCpuDownFraction:
            switch QwenAneSettingsValidator.cpuDownFraction(qwen35AnePrefillCpuDownFraction) {
            case .success(let value): patch.qwen35AnePrefillCpuDownFraction = value
            case .failure(let error): lastError = error.message; return
            }
        case .qwen35AnePrefillCpuGdnFraction:
            switch QwenAneSettingsValidator.cpuGdnFraction(
                qwen35AnePrefillCpuGdnFraction,
                gdnFraction: qwen35AnePrefillGdnFraction
            ) {
            case .success(let value): patch.qwen35AnePrefillCpuGdnFraction = value
            case .failure(let error): lastError = error.message; return
            }
        case .qwen35AnePrefillCpuThreads:
            switch QwenAneSettingsValidator.cpuThreads(qwen35AnePrefillCpuThreads) {
            case .success(let value): patch.qwen35AnePrefillCpuThreads = value
            case .failure(let error): lastError = error.message; return
            }
        case .qwen35AnePrefillCpuSharedResource:
            patch.qwen35AnePrefillCpuSharedResource = qwen35AnePrefillCpuSharedResource
        case .indexCacheEnabled:
            patch.indexCacheFreq = indexCacheEnabled ? (Int(indexCacheFreq) ?? 4) : 0
        case .indexCacheFreq:
            guard indexCacheEnabled, let n = Int(indexCacheFreq), n >= 2 else { return }
            patch.indexCacheFreq = n
        case .specprefillEnabled:      patch.specprefillEnabled = specprefillEnabled
        case .specprefillDraftModel:   patch.specprefillDraftModel = specprefillDraftModel.isEmpty ? nil : specprefillDraftModel
        case .specprefillKeepPct:      patch.specprefillKeepPct = Double(specprefillKeepPct)
        case .specprefillThreshold:    patch.specprefillThreshold = Int(specprefillThreshold)
        case .dflashEnabled:           patch.dflashEnabled = dflashEnabled
        case .dflashDraftModel:        patch.dflashDraftModel = dflashDraftModel.isEmpty ? nil : dflashDraftModel
        case .dflashDraftQuantEnabled:
            patch.dflashDraftQuantEnabled = dflashDraftQuantEnabled
            if !dflashDraftQuantEnabled {
                patch.dflashDraftQuantWeightBits = nil
                patch.dflashDraftQuantGroupSize = nil
            }
        case .dflashDraftQuantWeightBits:
            patch.dflashDraftQuantWeightBits = Int(dflashDraftQuantWeightBits)
        case .dflashDraftQuantGroupSize:
            patch.dflashDraftQuantGroupSize = Int(dflashDraftQuantGroupSize)
        case .dflashVerifyMode:        patch.dflashVerifyMode = dflashVerifyMode.isEmpty ? nil : dflashVerifyMode
        case .dflashDraftWindowSize:   patch.dflashDraftWindowSize = Int(dflashDraftWindowSize)
        case .dflashBlockSize:         patch.dflashBlockSize = Int(dflashBlockSize)
        case .mtpEnabled, .mtpAdaptiveMaxDepth:
            patch.mtpEnabled = mtpEnabled
            patch.mtpAdaptiveMaxDepth = Int(mtpAdaptiveMaxDepth) ?? 3
            patch.mtpFixedDepth = .some(nil)
        case .vlmMtpEnabled:           patch.vlmMtpEnabled = vlmMtpEnabled
        case .vlmMtpDraftModel:        patch.vlmMtpDraftModel = vlmMtpDraftModel.isEmpty ? nil : vlmMtpDraftModel
        case .vlmMtpDraftBlockSize:    patch.vlmMtpDraftBlockSize = Int(vlmMtpDraftBlockSize)
        }
        if thinkingForced { patch.enableThinking = .some(nil) }
        let targetModelID = modelID
        do {
            _ = try await client.updateModelSettings(id: targetModelID, patch: patch)
            self.lastError = nil
        } catch {
            // Keep pending profile edits; their auto-save reloads afterwards.
            await reportFailedSave(error, modelID: targetModelID, client: client,
                                   preservingEdits: true)
        }
    }

    func startANETuning(client: OMLXClient) async {
        guard !aneTuningIsRunning else { return }
        guard let sequenceLength = Int(qwen35AnePrefillSequenceLength) else {
            lastError = "ANE prompt block must be a number."
            return
        }
        aneTuningIsRunning = true
        aneTuningStatus = nil
        lastError = nil
        do {
            let started = try await client.startANETuning(
                ANETuningStartRequest(
                    modelId: modelID,
                    sequenceLength: sequenceLength,
                    repeats: 2,
                    allowCpu: aneTuningAllowCPU,
                    allowCpuGate: aneTuningAllowCPU && aneTuningAllowCPUGate,
                    allowCpuDown: aneTuningAllowCPU && aneTuningAllowCPUDown,
                    allowAneGdn: aneTuningAllowANEGDN,
                    allowCpuGdn: aneTuningAllowCPU
                        && aneTuningAllowANEGDN
                        && aneTuningAllowCPUGDN,
                    allowCpuSharedResource: aneTuningAllowCPU
                        && aneTuningAllowCPUSharedResource
                )
            )
            aneTuningID = started.tuningId
            while aneTuningIsRunning {
                let snapshot = try await client.getANETuningResults(
                    tuningId: started.tuningId
                )
                aneTuningStatus = snapshot
                if snapshot.status != "running" {
                    aneTuningIsRunning = false
                    // Benchmark termination is rendered with its partial
                    // matrix in the tuner row. Reserve lastError for transport
                    // and settings failures so the reason is not duplicated.
                    lastError = nil
                    break
                }
                try await Task.sleep(for: .seconds(1))
            }
        } catch is CancellationError {
            aneTuningIsRunning = false
        } catch {
            aneTuningIsRunning = false
            lastError = error.omlxDescription
        }
    }

    func cancelANETuning(client: OMLXClient) async {
        guard let tuningID = aneTuningID, aneTuningIsRunning else { return }
        do {
            _ = try await client.cancelANETuning(tuningId: tuningID)
        } catch {
            lastError = error.omlxDescription
        }
    }

    /// Stage the best tuner result like any other edit, so it auto-saves.
    func applyANETuningRecommendation() {
        guard let recommendation = aneTuningStatus?.recommendation else { return }
        qwen35AnePrefillEnabled = recommendation.enabled
        if recommendation.enabled {
            // The two prefill accelerators are mutually exclusive, and the
            // tuner's recommendation is the more specific answer here: it was
            // measured on this model's own layers.
            qwen35OqA8Enabled = false
        }
        qwen35AnePrefillSequenceLength = String(recommendation.sequenceLength)
        if let fraction = recommendation.mlpFraction { qwen35AnePrefillFraction = String(fraction) }
        if recommendation.backend == "k2" {
            if let fraction = recommendation.sharedFraction { qwen35AnePrefillSharedFraction = String(fraction) }
            refreshProfileDirty()
            lastError = nil
            return
        }
        qwen35AnePrefillTailPaddingMinTokens = String(
            recommendation.tailPaddingMinTokens ?? 0
        )
        if let fraction = recommendation.mlpFraction {
            qwen35AnePrefillFraction = Self.formatPct(fraction)
        }
        qwen35AnePrefillFusedDown = recommendation.fusedDown ?? false
        qwen35AnePrefillGdn = recommendation.gdnEnabled
        if let fraction = recommendation.gdnFraction {
            qwen35AnePrefillGdnFraction = Self.formatPct(fraction)
        }
        qwen35AnePrefillCpuEnabled = recommendation.cpuEnabled ?? false
        if let fraction = recommendation.cpuFraction {
            qwen35AnePrefillCpuFraction = Self.formatPct(fraction)
        }
        if let fraction = recommendation.cpuDownFraction {
            qwen35AnePrefillCpuDownFraction = Self.formatPct(fraction)
        }
        if let fraction = recommendation.cpuGdnFraction {
            qwen35AnePrefillCpuGdnFraction = Self.formatPct(fraction)
        }
        if let threads = recommendation.cpuThreads {
            qwen35AnePrefillCpuThreads = String(threads)
        }
        if let sharedResource = recommendation.cpuSharedResource {
            qwen35AnePrefillCpuSharedResource = sharedResource
        }
        refreshProfileDirty()
        lastError = nil
    }

    // MARK: - Chat-template kwarg list mutation

    func addKwarg(_ kind: ChatTemplateKwargEntryKind) {
        if thinkingForced && kind == .enableThinking { return }
        if isDiffusionModel {
            switch kind {
            case .enableThinking, .reasoningEffort:
                return
            case .custom:
                break
            }
        }
        let defaultValue: String
        switch kind {
        case .enableThinking:  defaultValue = "true"
        case .reasoningEffort: defaultValue = model?.reasoningEffortDefault ?? ""
        case .custom:          defaultValue = ""
        }
        chatTemplateEntries.append(
            ChatTemplateKwargEntry(kind: kind, value: defaultValue)
        )
        refreshProfileDirty()
    }

    func removeKwarg(id: UUID) {
        chatTemplateEntries.removeAll(where: { $0.id == id })
        refreshProfileDirty()
    }

    /// Options for SpecPrefill / DFlash draft-model dropdowns. Filters
    /// out the current model so it can't pick itself as its own draft.
    func draftModelOptions() -> [(String, String)] {
        var out: [(String, String)] = [
            ("", String(localized: "settings.draft_model.placeholder",
                        defaultValue: "Select draft model…",
                        comment: "Initial placeholder option in the SpecPrefill/DFlash draft-model picker")),
        ]
        for m in allModels where m.id != modelID {
            out.append((m.modelPath ?? m.id, m.id))
        }
        return out
    }

    /// Draft-model options for VLM MTP. mlx-vlm's MTP loop takes an
    /// assistant drafter. Match the HTML editor by accepting known
    /// config-derived drafter types first, then falling back to names that
    /// contain "assistant" or a standalone "mtp" token.
    ///
    /// The stored value is the model **id**, not its path: the server
    /// resolves the drafter by registry id (`engine_pool` looks it up in
    /// `_entries` keyed by model_id, then uses that entry's `model_path`).
    /// This matches the web modal, which binds `m.id`.
    func vlmMtpDraftModelOptions() -> [(String, String)] {
        var out: [(String, String)] = [
            ("", String(localized: "settings.vlm_mtp.draft.placeholder",
                        defaultValue: "Select assistant drafter…",
                        comment: "Initial placeholder option in the VLM MTP draft-model picker")),
        ]
        for m in allModels
        where Self.isVlmMtpDraftModelCandidate(m, currentModelID: modelID) {
            out.append((m.id, m.id))
        }
        return out
    }

    static func isVlmMtpDraftModelCandidate(_ model: ModelDTO, currentModelID: String) -> Bool {
        guard model.id != currentModelID else { return false }

        if let type = model.configModelType?.lowercased(),
           vlmMtpDrafterConfigModelTypes.contains(type) {
            return true
        }

        let searchText = [model.id, model.modelPath]
            .compactMap { $0 }
            .joined(separator: " ")
        if searchText.range(of: "assistant", options: .caseInsensitive) != nil {
            return true
        }
        return searchText.range(
            of: #"(^|[-_/\s])mtp($|[-_/\s])"#,
            options: [.regularExpression, .caseInsensitive]
        ) != nil
    }

    var isDSAConfigModel: Bool {
        guard let type = model?.configModelType else { return false }
        return Self.dsaConfigModelTypes.contains(type)
    }

    var isQwenOqA8Model: Bool {
        // qwen4_exp (Qwen3.8-Flash-Next) matches exactly: only that validated
        // family has routed-expert A8, not every qwen4*.
        if isQwen4Exp { return true }
        let type = (model?.configModelType ?? "").lowercased().replacingOccurrences(of: "-", with: "_")
        return ["qwen3_5", "qwen3_6", "qwen3_8"].contains { type.hasPrefix($0) }
    }

    var isQwen35AnePrefillModel: Bool { model?.anePrefillBackend == "qwen" }

    /// The INT8 kernels run only on GPUs with native INT8 tensor ops
    /// (M5-series or newer). A saved "on" keeps the row so it can be
    /// turned off.
    var showsQwenOqA8: Bool {
        isQwenOqA8Model
            && (model?.qwen35OqA8Supported == true || model?.settings?.qwen35OqA8Enabled == true)
    }

    /// Native Lightning MTP can't co-exist with the other speculative
    /// decoders. TurboQuant KV supports its decode-shaped multi-row verify
    /// path, so it is intentionally not a conflict here.
    var mtpConflictReason: String? {
        if dflashEnabled {
            return String(localized: "settings.mtp.conflict.dflash",
                          defaultValue: "Disable DFlash before enabling MTP.",
                          comment: "Tooltip / sublabel shown when MTP can't be enabled because DFlash is on")
        }
        if vlmMtpEnabled {
            return String(localized: "settings.mtp.conflict.vlm_mtp",
                          defaultValue: "Disable VLM MTP before enabling Lightning MTP.",
                          comment: "Tooltip / sublabel shown when Lightning MTP can't be enabled because VLM MTP is on")
        }
        if moeExpertOffloadEnabled && model?.moeOffloadAllowsMtp != true {
            return String(localized: "settings.mtp.conflict.moe_offload",
                          defaultValue: "Disable MoE Expert Offload before enabling Lightning MTP.",
                          comment: "Tooltip / sublabel shown when Lightning MTP can't be enabled because MoE expert offload is on")
        }
        return nil
    }

    /// Shown wherever VLM MTP owns the speculative path (DFlash, TurboQuant,
    /// SpecPrefill).
    static var vlmMtpOwnsSpeculativePathReason: String {
        String(localized: "settings.speculative.conflict.vlm_mtp",
               defaultValue: "Disable VLM MTP before enabling this feature.",
               comment: "Tooltip / sublabel shown when another speculative feature can't be enabled because VLM MTP is on")
    }

    var dflashConflictReason: String? {
        if moeExpertOffloadEnabled {
            return String(localized: "settings.dflash.conflict.moe_offload",
                          defaultValue: "Disable MoE Expert Offload before enabling DFlash.",
                          comment: "Tooltip / sublabel shown when DFlash can't be enabled because MoE expert offload is on")
        }
        if mtpEnabled {
            return String(localized: "settings.dflash.conflict.mtp",
                          defaultValue: "Disable Lightning MTP before enabling DFlash.",
                          comment: "Tooltip / sublabel shown when DFlash can't be enabled because Lightning MTP is on")
        }
        if vlmMtpEnabled { return Self.vlmMtpOwnsSpeculativePathReason }
        return nil
    }

    /// Streamed experts cannot back a speculative decoder, except the
    /// Lightning MTP head of families that keep it resident. Mirrors
    /// `validate_moe_expert_offload` on the server.
    var moeExpertOffloadConflictReason: String? {
        if mtpEnabled && model?.moeOffloadAllowsMtp != true {
            return String(localized: "settings.moe_offload.conflict.mtp",
                          defaultValue: "Disable Lightning MTP before enabling MoE Expert Offload.",
                          comment: "Tooltip / sublabel shown when MoE expert offload can't be enabled because Lightning MTP is on")
        }
        if vlmMtpEnabled {
            return String(localized: "settings.moe_offload.conflict.vlm_mtp",
                          defaultValue: "Disable VLM MTP before enabling MoE Expert Offload.",
                          comment: "Tooltip / sublabel shown when MoE expert offload can't be enabled because VLM MTP is on")
        }
        if dflashEnabled {
            return String(localized: "settings.moe_offload.conflict.dflash",
                          defaultValue: "Disable DFlash before enabling MoE Expert Offload.",
                          comment: "Tooltip / sublabel shown when MoE expert offload can't be enabled because DFlash is on")
        }
        return nil
    }

    /// The oQ INT8-activation kernels and ANE prefill both wrap the same
    /// Qwen3.5 MLP call, so enabling both leaves whichever patched last in
    /// charge and the other silently inert. The server rejects the pair; these
    /// mirror that so the losing toggle disables itself and says why instead
    /// of the save returning a 400 with the switch already flipped.
    var qwen35OqA8ConflictReason: String? {
        guard qwen35AnePrefillEnabled else { return nil }
        return String(localized: "settings.qwen_oq_a8.conflict.ane",
                      defaultValue: "Disable Qwen ANE Prefill before enabling INT8 activation prefill.",
                      comment: "Tooltip / sublabel shown when INT8 activation prefill can't be enabled because ANE prefill is on")
    }

    var qwen35AnePrefillConflictReason: String? {
        guard qwen35OqA8Enabled else { return nil }
        return String(localized: "settings.qwen_ane.conflict.oq_a8",
                      defaultValue: "Disable Qwen INT8 Activation Prefill before enabling ANE prefill.",
                      comment: "Tooltip / sublabel shown when ANE prefill can't be enabled because INT8 activation prefill is on")
    }

    /// MTPLX checkpoints keep MTP weights in a sidecar until the server
    /// imports them into the model index. Match the server-provided
    /// compatibility marker rather than guessing from model names.
    var canImportMtplxSidecar: Bool {
        guard model?.mtpCompatible == false,
              let reason = model?.mtpCompatibilityReason else {
            return false
        }
        return reason.localizedCaseInsensitiveContains("MTPLX side-car")
    }

    func importMtplxSidecar(client: OMLXClient) async {
        guard !isImportingMtplxSidecar, canImportMtplxSidecar else { return }
        isImportingMtplxSidecar = true
        defer { isImportingMtplxSidecar = false }

        do {
            _ = try await client.importMtplxSidecar(id: modelID)
            await load(modelID: modelID, client: client, preservingEdits: true)
        } catch {
            lastError = error.omlxDescription
        }
    }

    /// VLM MTP wraps mlx-vlm's MTP loop and is mutually exclusive with the
    /// other speculative-decoding / KV-quant features. Mirrors the HTML
    /// editor's gating so the toggle disables itself and surfaces why.
    var vlmMtpConflictReason: String? {
        if moeExpertOffloadEnabled {
            return String(localized: "settings.vlm_mtp.conflict.moe_offload",
                          defaultValue: "Disable MoE Expert Offload before enabling VLM MTP.",
                          comment: "Tooltip / sublabel shown when VLM MTP can't be enabled because MoE expert offload is on")
        }
        if dflashEnabled {
            return String(localized: "settings.vlm_mtp.conflict.dflash",
                          defaultValue: "Disable DFlash before enabling VLM MTP.",
                          comment: "Tooltip / sublabel shown when VLM MTP can't be enabled because DFlash is on")
        }
        if specprefillEnabled {
            return String(localized: "settings.vlm_mtp.conflict.specprefill",
                          defaultValue: "Disable SpecPrefill before enabling VLM MTP.",
                          comment: "Tooltip / sublabel shown when VLM MTP can't be enabled because SpecPrefill is on")
        }
        if mtpEnabled {
            return String(localized: "settings.vlm_mtp.conflict.mtp",
                          defaultValue: "Disable Lightning MTP before enabling VLM MTP.",
                          comment: "Tooltip / sublabel shown when VLM MTP can't be enabled because Lightning MTP is on")
        }
        if turboquantKvEnabled {
            return String(localized: "settings.vlm_mtp.conflict.turboquant",
                          defaultValue: "Disable TurboQuant KV before enabling VLM MTP.",
                          comment: "Tooltip / sublabel shown when VLM MTP can't be enabled because TurboQuant KV is on")
        }
        if vlmMtpProcessorConflict {
            return String(localized: "settings.vlm_mtp.conflict.processors",
                          defaultValue: "Unset repetition / presence penalty and turn off Guided Grammar before enabling VLM MTP.",
                          comment: "Tooltip / sublabel shown when VLM MTP can't be enabled because penalty or grammar settings are set")
        }
        return nil
    }

    /// Settings that materialize as per-request logits processors, which the
    /// vlm_mtp decode path cannot apply (#2399). Mirrors
    /// vlm_mtp_processor_conflicts() in model_settings.py; neutral values
    /// (repetition 1.0, presence 0.0) do not conflict. Thinking budget is
    /// exempt: it is applied on the vlm_mtp path at verify time
    /// (MTPProcessingSampler).
    var vlmMtpProcessorConflict: Bool {
        if let rep = Double(repetitionPenalty), rep != 1.0 { return true }
        if let pres = Double(presencePenalty), pres != 0.0 { return true }
        return guidedGrammarEnabled
    }

    /// Sublabel / tooltip for the sampling rows locked while VLM MTP is on.
    var vlmMtpProcessorLockedReason: String {
        String(localized: "settings.sampling.locked.vlm_mtp",
               defaultValue: "Disable VLM MTP to edit this setting.",
               comment: "Tooltip / sublabel shown when a penalty or thinking-budget row is locked because VLM MTP is on")
    }

    // MARK: - Page summaries

    /// Trailing value of a page row: the first two enabled features, then a
    /// count of the rest.
    nonisolated static func summary(_ items: [String], empty: String) -> String {
        switch items.count {
        case 0: return empty
        case 1, 2: return items.joined(separator: ", ")
        default: return "\(items[0]), \(items[1]) +\(items.count - 2)"
        }
    }

    private static var summaryOff: String {
        String(localized: "settings.summary.off",
               defaultValue: "Off",
               comment: "Page row value when no feature on that page is enabled")
    }

    private static var summaryExternalEngine: String {
        String(localized: "settings.summary.external_engine",
               defaultValue: "Not used with external engine",
               comment: "Page row value for MLX-only pages while an external engine serves the model")
    }

    func summary(for page: ModelSettingsPage) -> String {
        switch page {
        case .profiles:
            // The pending auto-save does not count as a different profile.
            guard let name = activeProfileName else {
                return String(localized: "profile.banner.defaults.title",
                              defaultValue: "No profile",
                              comment: "Profile banner title when the model uses server defaults")
            }
            return profileDisplayName(scope: activeProfileScope, name: name)
        case .acceleration, .memory, .experimental:
            if !externalEngine.isEmpty { return Self.summaryExternalEngine }
            return Self.summary(enabledFeatures(on: page), empty: Self.summaryOff)
        case .advanced:
            return Self.summary(enabledFeatures(on: .advanced), empty: Self.summaryOff)
        }
    }

    /// Enabled features on a page, gated like the page's rows.
    func enabledFeatures(on page: ModelSettingsPage) -> [String] {
        var items: [String] = []
        switch page {
        case .profiles:
            break
        case .acceleration:
            if mtpEnabled {
                items.append(String(localized: "settings.acceleration.mtp.label",
                                    defaultValue: "Lightning MTP",
                                    comment: "Row label for the Lightning MTP toggle"))
            }
            if dflashEnabled {
                items.append(String(localized: "settings.experimental.dflash.label",
                                    defaultValue: "DFlash",
                                    comment: "Row label for the DFlash toggle"))
            }
            if vlmMtpEnabled {
                items.append(String(localized: "settings.experimental.vlm_mtp.label",
                                    defaultValue: "VLM MTP",
                                    comment: "Row label for the VLM MTP toggle"))
            }
            if isDeepseekV41 && deepseekV41CedPrefillEnabled {
                items.append(String(localized: "settings.summary.ced_prefill",
                                    defaultValue: "CED Prefill",
                                    comment: "Short name of DeepSeek V4.1 CED prefill in the Acceleration row value"))
            }
            if isQwen4Exp && qwen4GdnDecodeWideProj {
                items.append(String(localized: "settings.summary.qwen4_wide_decode",
                                    defaultValue: "Extended Fused Decode",
                                    comment: "Short name of Qwen4 extended fused decode in the Acceleration row value"))
            }
            if showsQwenOqA8 && qwen35OqA8Enabled {
                items.append(String(localized: "settings.summary.qwen_oq_a8",
                                    defaultValue: "INT8 Prefill",
                                    comment: "Short name of Qwen INT8 activation prefill in the Acceleration row value"))
            }
        case .memory:
            if ssdNgramOffloadSupported && ssdNgramOffload {
                items.append(String(localized: "settings.memory.ssd_ngram_offload.label",
                                    defaultValue: "SSD N-gram Offload",
                                    comment: "Row label for the shared Qwen4 PLE / DeepSeek V4.1 Engram SSD offload toggle"))
            }
            if moeExpertOffloadSupported && moeExpertOffloadEnabled {
                let percent = Self.formatPct((moeExpertOffloadResidentFraction ?? loadedMoeResidentFraction) * 100)
                items.append(String(localized: "settings.summary.moe_offload",
                                    defaultValue: "MoE Offload \(percent)%",
                                    comment: "MoE expert offload in the Memory Saving row value; placeholder is the resident percent"))
            }
            if turboquantKvEnabled {
                items.append(String(localized: "settings.summary.turboquant",
                                    defaultValue: "TurboQuant \(turboquantKvBits)-bit",
                                    comment: "TurboQuant KV in the Memory Saving row value; placeholder is the bit width"))
            }
        case .advanced:
            if !isDiffusionModel, !externalEngine.isEmpty {
                let label = model?.externalEngines?.first(where: { $0.id == externalEngine })?.label
                items.append(label ?? externalEngine)
            }
            if trustRemoteCode {
                items.append(String(localized: "settings.advanced.trust_remote_code.label",
                                    defaultValue: "Trust Remote Code",
                                    comment: "Row label for the security-sensitive trust-remote-code toggle"))
            }
            if isLLMLike && !isDiffusionModel {
                if thinkingBudgetEnabled {
                    items.append(String(localized: "settings.advanced.thinking_budget.label",
                                        defaultValue: "Thinking Budget",
                                        comment: "Row label for the thinking budget field"))
                }
                if limitToolResults {
                    items.append(String(localized: "settings.summary.tool_result_limit",
                                        defaultValue: "Tool Result Limit",
                                        comment: "Short name of the tool-result token limit in the Advanced row value"))
                }
                if guidedGrammarEnabled {
                    items.append(String(localized: "settings.advanced.guided_grammar.label",
                                        defaultValue: "Guided Grammar",
                                        comment: "Row label for the default guided grammar toggle"))
                }
                let parser = reasoningParser.trimmingCharacters(in: .whitespaces)
                if !parser.isEmpty { items.append(parser) }
            }
            if isLLMLike && !chatTemplateEntries.isEmpty {
                items.append(String(localized: "settings.summary.kwargs",
                                    defaultValue: "\(chatTemplateEntries.count) kwargs",
                                    comment: "Chat-template kwargs in the Advanced row value; placeholder is the entry count"))
            }
            if embeddingAudioSupported && embeddingAudioEnabled {
                items.append(String(localized: "settings.advanced.embedding_audio.label",
                                    defaultValue: "Audio Input",
                                    comment: "Row label for the embedding audio-input toggle"))
            }
        case .experimental:
            if model?.anePrefillBackend != nil && qwen35AnePrefillEnabled {
                items.append(String(localized: "settings.summary.ane_prefill",
                                    defaultValue: "ANE Prefill",
                                    comment: "Short name of ANE prompt processing in the Experimental row value"))
            }
            if isDSAConfigModel && indexCacheEnabled {
                items.append(String(localized: "settings.experimental.indexcache.label",
                                    defaultValue: "IndexCache",
                                    comment: "Row label for the DSA IndexCache toggle"))
            }
            if specprefillEnabled {
                items.append(String(localized: "settings.experimental.specprefill.label",
                                    defaultValue: "SpecPrefill",
                                    comment: "Row label for the SpecPrefill toggle"))
            }
        }
        return items
    }

    // MARK: - Generation defaults

    func loadGenerationDefaults(client: OMLXClient) async {
        guard !isLoadingGenerationDefaults else { return }
        isLoadingGenerationDefaults = true
        defer { isLoadingGenerationDefaults = false }
        do {
            applyGenerationDefaults(try await client.getGenerationConfig(id: modelID))
        } catch OMLXClientError.http(let status, _) where status == 404 {
            lastError = String(localized: "settings.generation.load_defaults.none",
                               defaultValue: "No config defaults found for this model",
                               comment: "Error when the model ships no generation defaults to load")
        } catch {
            lastError = error.omlxDescription
        }
    }

    /// Stage the model's recommended values for Generation Defaults only.
    /// Fields outside this section keep their values; the toolbar Reset
    /// covers every setting.
    func applyGenerationDefaults(_ config: GenerationConfigDTO) {
        contextLength = config.maxContextWindow.map(String.init) ?? ""
        temperature = config.temperature.map { String($0) } ?? ""
        topP = config.topP.map { String($0) } ?? ""
        topK = config.topK.map(String.init) ?? ""
        repetitionPenalty = config.repetitionPenalty.map { String($0) } ?? ""
        maxTokens = ""
        minP = ""
        presencePenalty = ""
        forceSampling = false
        refreshProfileDirty()
        lastError = nil
    }

    // MARK: - Working profile dict assembly

    /// Snapshot the current profile-eligible field values into the
    /// loose `settings` dict the server stores on profiles + templates.
    /// Keys are snake_case (the server's wire shape). Empty / unparseable
    /// fields are dropped — the server treats absent keys as "use defaults".
    func currentSettingsDict() -> [String: AnyCodable] {
        var out: [String: AnyCodable] = [:]
        let isDiffusion = isDiffusionModel

        func putInt(_ key: String, _ raw: String) {
            let t = raw.trimmingCharacters(in: .whitespaces)
            if t.isEmpty { return }
            if let n = Int(t) { out[key] = AnyCodable(n) }
        }
        func putDouble(_ key: String, _ raw: String) {
            let t = raw.trimmingCharacters(in: .whitespaces)
            if t.isEmpty { return }
            if let n = Double(t) { out[key] = AnyCodable(n) }
        }
        func putBool(_ key: String, _ v: Bool) {
            out[key] = AnyCodable(v)
        }
        func putString(_ key: String, _ raw: String) {
            let t = raw.trimmingCharacters(in: .whitespaces)
            if t.isEmpty { return }
            out[key] = AnyCodable(t)
        }

        // Universal — sampling
        putInt(ProfileSettingsKey.maxContextWindow, contextLength)
        putInt(ProfileSettingsKey.maxTokens, maxTokens)
        putDouble(ProfileSettingsKey.temperature, temperature)
        if !isDiffusion {
            putDouble(ProfileSettingsKey.topP, topP)
            putInt(ProfileSettingsKey.topK, topK)
            putDouble(ProfileSettingsKey.minP, minP)
            putDouble(ProfileSettingsKey.repetitionPenalty, repetitionPenalty)
            putDouble(ProfileSettingsKey.presencePenalty, presencePenalty)
        }

        // Universal — thinking / tool / reasoning
        if !isDiffusion {
            if !thinkingForced { putBool(ProfileSettingsKey.enableThinking, enableThinking) }
            putBool(ProfileSettingsKey.thinkingBudgetEnabled, thinkingBudgetEnabled)
            putInt(ProfileSettingsKey.thinkingBudgetTokens, thinkingBudgetTokens)
            putBool(ProfileSettingsKey.forceSampling, forceSampling)
            putString(ProfileSettingsKey.reasoningParser, reasoningParser)
            putBool(ProfileSettingsKey.guidedGrammarEnabled, guidedGrammarEnabled)
            if guidedGrammarEnabled {
                putString(ProfileSettingsKey.guidedGrammar, guidedGrammar)
            }
            // Server uses 0 as the "disable" sentinel; encode that exactly.
            out[ProfileSettingsKey.maxToolResultTokens] = AnyCodable(
                limitToolResults ? (Int(toolResultLimitTokens) ?? 4096) : 0
            )
        }

        // Universal — chat template kwargs. AnyCodable's encode walks a
        // [String: AnyCodable] / [AnyCodable] explicitly, so nest those
        // shapes rather than `Any` so the Sendable check is satisfied.
        let kwargs = ChatTemplateKwargsCodec.encode(
            diffusionCompatibleChatTemplateEntries(chatTemplateEntries)
        )
        if let dict = kwargs.kwargs {
            out[ProfileSettingsKey.chatTemplateKwargs] = AnyCodable(dict)
        }
        if let forced = kwargs.forced, !forced.isEmpty {
            out[ProfileSettingsKey.forcedCtKwargs] = AnyCodable(
                forced.map { AnyCodable($0) }
            )
        }

        // Model-specific — experimental
        if !isDiffusion {
            if moeExpertOffloadSupported {
                putBool(ProfileSettingsKey.moeExpertOffloadEnabled, moeExpertOffloadEnabled)
                if let fraction = moeExpertOffloadResidentFraction {
                    out[ProfileSettingsKey.moeExpertOffloadResidentFraction] = AnyCodable(fraction)
                }
            }
            putBool(ProfileSettingsKey.turboquantKvEnabled, turboquantKvEnabled)
            if turboquantKvEnabled {
                putString(ProfileSettingsKey.turboquantKvScheme, turboquantKvScheme)
                let fixedBits = turboquantKvScheme == "affine4"
                    ? 4.0 : turboquantKvScheme == "affine8" ? 8.0 : nil
                if let bits = fixedBits ?? Double(turboquantKvBits) {
                    out[ProfileSettingsKey.turboquantKvBits] = AnyCodable(bits)
                }
            }
            putBool(ProfileSettingsKey.qwen35OqA8Enabled, qwen35OqA8Enabled)
            if qwen35OqA8Enabled {
                putInt(ProfileSettingsKey.qwen35OqA8MinTokens, qwen35OqA8MinTokens)
            }
            putBool(ProfileSettingsKey.qwen35AnePrefillEnabled, qwen35AnePrefillEnabled)
            if qwen35AnePrefillEnabled {
                putInt(ProfileSettingsKey.qwen35AnePrefillSequenceLength, qwen35AnePrefillSequenceLength)
                putDouble(ProfileSettingsKey.qwen35AnePrefillFraction, qwen35AnePrefillFraction)
                if model?.anePrefillBackend == "k2" {
                    putDouble(ProfileSettingsKey.qwen35AnePrefillSharedFraction, qwen35AnePrefillSharedFraction)
                }
            }
            if qwen35AnePrefillEnabled && isQwen35AnePrefillModel {
                putInt(ProfileSettingsKey.qwen35AnePrefillTailPaddingMinTokens, qwen35AnePrefillTailPaddingMinTokens)
                putInt(ProfileSettingsKey.qwen35AnePrefillMaxLayers, qwen35AnePrefillMaxLayers)
                putBool(ProfileSettingsKey.qwen35AnePrefillDualAne, qwen35AnePrefillDualAne)
                putBool(ProfileSettingsKey.qwen35AnePrefillGdn, qwen35AnePrefillGdn)
                if qwen35AnePrefillGdn {
                    putDouble(ProfileSettingsKey.qwen35AnePrefillGdnFraction, qwen35AnePrefillGdnFraction)
                    putInt(ProfileSettingsKey.qwen35AnePrefillGdnMaxLayers, qwen35AnePrefillGdnMaxLayers)
                }
                putBool(ProfileSettingsKey.qwen35AnePrefillCpuEnabled, qwen35AnePrefillCpuEnabled)
                putBool(ProfileSettingsKey.qwen35AnePrefillFusedDown, qwen35AnePrefillFusedDown)
                if qwen35AnePrefillCpuEnabled {
                    putDouble(ProfileSettingsKey.qwen35AnePrefillCpuFraction, qwen35AnePrefillCpuFraction)
                    putDouble(ProfileSettingsKey.qwen35AnePrefillCpuDownFraction, qwen35AnePrefillCpuDownFraction)
                    putDouble(ProfileSettingsKey.qwen35AnePrefillCpuGdnFraction, qwen35AnePrefillCpuGdnFraction)
                    putInt(ProfileSettingsKey.qwen35AnePrefillCpuThreads, qwen35AnePrefillCpuThreads)
                    putBool(ProfileSettingsKey.qwen35AnePrefillCpuSharedResource, qwen35AnePrefillCpuSharedResource)
                }
            }
            if indexCacheEnabled, let n = Int(indexCacheFreq), n >= 2 {
                out[ProfileSettingsKey.indexCacheFreq] = AnyCodable(n)
            }
            putBool(ProfileSettingsKey.specprefillEnabled, specprefillEnabled)
            if specprefillEnabled {
                putString(ProfileSettingsKey.specprefillDraftModel, specprefillDraftModel)
                putDouble(ProfileSettingsKey.specprefillKeepPct, specprefillKeepPct)
                putInt(ProfileSettingsKey.specprefillThreshold, specprefillThreshold)
            }
            putBool(ProfileSettingsKey.dflashEnabled, dflashEnabled)
            if dflashEnabled {
                putString(ProfileSettingsKey.dflashDraftModel, dflashDraftModel)
                putBool(ProfileSettingsKey.dflashDraftQuantEnabled, dflashDraftQuantEnabled)
                if dflashDraftQuantEnabled {
                    putInt(ProfileSettingsKey.dflashDraftQuantWeightBits, dflashDraftQuantWeightBits)
                    putInt(ProfileSettingsKey.dflashDraftQuantGroupSize, dflashDraftQuantGroupSize)
                }
                if !dflashVerifyMode.isEmpty {
                    out[ProfileSettingsKey.dflashVerifyMode] = AnyCodable(dflashVerifyMode)
                }
                putInt(ProfileSettingsKey.dflashDraftWindowSize, dflashDraftWindowSize)
                putInt(ProfileSettingsKey.dflashBlockSize, dflashBlockSize)
            }
            putBool(ProfileSettingsKey.mtpEnabled, mtpEnabled)
            if mtpEnabled {
                putInt(ProfileSettingsKey.mtpAdaptiveMaxDepth, mtpAdaptiveMaxDepth)
            }
            putBool(ProfileSettingsKey.vlmMtpEnabled, vlmMtpEnabled)
            if vlmMtpEnabled {
                putString(ProfileSettingsKey.vlmMtpDraftModel, vlmMtpDraftModel)
                putInt(ProfileSettingsKey.vlmMtpDraftBlockSize, vlmMtpDraftBlockSize)
            }
        }

        return out
    }

    // MARK: - Profile actions

    /// Apply a chip's profile to the model. Discards any working-profile
    /// state per chat2.md: "Any unsaved work is silently dispatched."
    /// `.preset` is routed through `applyPreset(_:client:)` — that path
    /// receives the bundle entry directly since presets aren't stored as
    /// server templates.
    func applyChip(scope: ProfileScope, name: String, client: OMLXClient) async {
        let targetModelID = modelID
        do {
            switch scope {
            case .preset:
                // Caller dispatches via applyPreset(_:client:) — this
                // branch is a defensive no-op so misrouted calls don't
                // hit a template lookup that's guaranteed to miss.
                return
            case .model:
                _ = try await client.applyModelProfile(id: targetModelID, name: name)
            case .global:
                _ = try await client.applyModelTemplate(id: targetModelID, name: name)
            }
            if modelID == targetModelID {
                await load(modelID: targetModelID, client: client)
            }
        } catch {
            self.lastError = error.omlxDescription
        }
    }

    /// Rename a global template via PUT /api/profile-templates/{name}.
    /// Keep the internal reference stable when editing the display name.
    func renameTemplate(from original: String, to renamed: String, client: OMLXClient) async {
        do {
            _ = try await client.updateProfileTemplate(
                name: original,
                body: UpdateTemplateRequest(displayName: renamed)
            )
            await load(modelID: modelID, client: client)
        } catch {
            self.lastError = error.omlxDescription
        }
    }

    /// Rename a per-model profile via PUT /api/models/{id}/profiles/{name}.
    /// Editing the display name leaves the active reference and API ID intact.
    func renameModelProfile(from original: String, to renamed: String, client: OMLXClient) async {
        do {
            _ = try await client.updateModelProfile(
                id: modelID,
                name: original,
                body: UpdateProfileRequest(displayName: renamed)
            )
            await load(modelID: modelID, client: client)
        } catch {
            self.lastError = error.omlxDescription
        }
    }

    /// Flip `expose_as_model` on a per-model profile via PUT. The body
    /// carries only the flag (absent fields are merge-no-ops server-side);
    /// reload picks up the derived `model_id` the profile serves under.
    func setExposeAsModel(name: String, exposed: Bool, client: OMLXClient) async {
        do {
            _ = try await client.updateModelProfile(
                id: modelID,
                name: name,
                body: UpdateProfileRequest(exposeAsModel: exposed)
            )
            await load(modelID: modelID, client: client)
        } catch {
            self.lastError = error.omlxDescription
        }
    }

    /// Apply a bundled preset entry to the model. Seeds a per-model
    /// profile (named after the preset, no `sourceTemplate` since presets
    /// aren't stored as server templates) and activates it. Mirrors
    /// HTML's behavior of materializing a preset as a model profile on
    /// first apply.
    func applyPreset(_ entry: PresetEntry, client: OMLXClient) async {
        do {
            if !self.profiles.contains(where: { $0.name == entry.name }) {
                _ = try? await client.createModelProfile(
                    id: modelID,
                    body: CreateProfileRequest(
                        name: entry.name,
                        displayName: entry.displayName,
                        description: entry.description,
                        sourceTemplate: nil,
                        settings: entry.settings
                    )
                )
            }
            _ = try await client.applyModelProfile(id: modelID, name: entry.name)
            await load(modelID: modelID, client: client)
        } catch {
            self.lastError = error.omlxDescription
        }
    }


    // MARK: - Snapshot actions

    func resetDefaults(client: OMLXClient) async {
        guard !isApplyingSettings else { return }
        isApplyingSettings = true
        defer { isApplyingSettings = false }
        do {
            _ = try await client.resetModelSettings(id: modelID)
            await load(modelID: modelID, client: client)
        } catch {
            lastError = error.omlxDescription
        }
    }

    func loadOptimalCandidates(client: OMLXClient) async {
        guard !isApplyingSettings else { return }
        isApplyingSettings = true
        applyError = nil
        applyOutcome = .optimalCandidates(nil)
        defer { isApplyingSettings = false }
        do {
            let candidates = try await client.listOptimalCandidates(id: modelID)
            applyOutcome = .optimalCandidates(candidates)
        } catch {
            applyError = error.omlxDescription
        }
    }

    func applyOptimalCandidate(_ benchmarkId: String, client: OMLXClient) async {
        guard !isApplyingSettings else { return }
        isApplyingSettings = true
        applyError = nil
        defer { isApplyingSettings = false }
        do {
            let result = try await client.applyOptimalCandidate(id: modelID, benchmarkId: benchmarkId)
            await load(modelID: modelID, client: client)
            applyOutcome = .optimal(result)
        } catch {
            applyError = error.omlxDescription
        }
    }

    func applyRecipe(_ recipe: String, client: OMLXClient) async {
        let text = recipe.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isApplyingSettings else { return }
        isApplyingSettings = true
        applyError = nil
        defer { isApplyingSettings = false }
        do {
            let result = try await client.applyRecipe(id: modelID, recipe: text)
            await load(modelID: modelID, client: client)
            applyOutcome = .recipe(result)
        } catch {
            applyError = error.omlxDescription
        }
    }

    /// One line per skipped feature for the result sheet.
    nonisolated static func summarizeSkipped(_ skipped: [SkippedFeatureDTO]?) -> [String] {
        (skipped ?? []).map { "\($0.feature): \($0.reason)" }
    }

    /// Pretty JSON of the applied values, keys sorted so the sheet is stable.
    nonisolated static func appliedJSON(_ applied: [String: AnyCodable]?) -> String {
        guard let applied, !applied.isEmpty else { return "{}" }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(applied),
              let text = String(data: data, encoding: .utf8) else { return "{}" }
        return text
    }

    /// "PP 1309.1 tok/s · TG 59.6 tok/s · 128 GB · 4bit · oMLX 0.7.0" for a candidate row.
    nonisolated static func candidateStats(pp: Double?, tg: Double?, memoryGb: Int? = nil,
                                           quantization: String?, omlxVersion: String?) -> String {
        var parts: [String] = []
        if let pp { parts.append(String(format: "PP %.1f tok/s", pp)) }
        if let tg { parts.append(String(format: "TG %.1f tok/s", tg)) }
        if let memoryGb { parts.append("\(memoryGb) GB") }
        if let quantization, !quantization.isEmpty { parts.append(quantization) }
        if let omlxVersion, !omlxVersion.isEmpty { parts.append("oMLX \(omlxVersion)") }
        return parts.joined(separator: " · ")
    }

    /// Write the working values straight to the model's settings, like the
    /// web admin's Save. No profile is needed; the server detaches the
    /// active profile when the saved values no longer match it.
    func saveWorkingToModel(client: OMLXClient) async {
        let targetModelID = modelID
        if let message = samplingValidationError() {
            lastError = message
            return
        }
        guard validateWorkingSettings() else { return }
        let saved = currentSettingsDict()
        do {
            _ = try await client.updateModelSettings(id: targetModelID, settings: workingSettingsBody())
            guard modelID == targetModelID else { return }
            // Edits made during the request stay unsaved for the next write.
            loadedSettings = saved
            profileDirty = currentSettingsDict() != saved
            await load(modelID: targetModelID, client: client, preservingEdits: true)
        } catch {
            await reportFailedSave(error, modelID: targetModelID, client: client,
                                   preservingEdits: false)
        }
    }

    /// The server validates the whole write before storing any of it, so a
    /// rejected save leaves the stored values unchanged. Reload them so the
    /// controls do not keep showing the rejected values.
    private func reportFailedSave(_ error: Error, modelID targetModelID: String,
                                  client: OMLXClient, preservingEdits: Bool) async {
        let message = error.omlxDescription
        if modelID == targetModelID, case .http? = error as? OMLXClientError {
            await load(modelID: targetModelID, client: client, preservingEdits: preservingEdits)
        }
        lastError = message
    }

    /// PUT body for `saveWorkingToModel`. A profile omits empty fields,
    /// but a settings write must send null to clear them.
    func workingSettingsBody() -> [String: AnyCodable] {
        var body = currentSettingsDict()
        let clearable = [
            ProfileSettingsKey.maxContextWindow, ProfileSettingsKey.maxTokens,
            ProfileSettingsKey.temperature, ProfileSettingsKey.topP,
            ProfileSettingsKey.topK, ProfileSettingsKey.minP,
            ProfileSettingsKey.repetitionPenalty, ProfileSettingsKey.presencePenalty,
            ProfileSettingsKey.reasoningParser, ProfileSettingsKey.guidedGrammar,
            ProfileSettingsKey.chatTemplateKwargs, ProfileSettingsKey.forcedCtKwargs,
        ]
        for key in clearable where body[key] == nil {
            body[key] = AnyCodable(NSNull())
        }
        if thinkingForced {
            body[ProfileSettingsKey.enableThinking] = AnyCodable(NSNull())
        }
        if !isDiffusionModel {
            // 0 is the server's "disabled" value for IndexCache.
            if !indexCacheEnabled {
                body[ProfileSettingsKey.indexCacheFreq] = AnyCodable(0)
            }
            body["mtp_fixed_depth"] = AnyCodable(NSNull())
        }
        return body
    }

    private func samplingValidationError() -> String? {
        if case .failure(let error) = SamplingValidator.temperature(temperature) {
            return error.message
        }
        guard !isDiffusionModel else { return nil }
        if case .failure(let error) = SamplingValidator.topP(topP) { return error.message }
        if case .failure(let error) = SamplingValidator.topK(topK) { return error.message }
        if case .failure(let error) = SamplingValidator.minP(minP) { return error.message }
        if case .failure(let error) = SamplingValidator.penalty(
            repetitionPenalty,
            name: String(localized: "settings.validator.repetition_penalty.name",
                         defaultValue: "Repetition Penalty",
                         comment: "Field name embedded in validation errors for repetition penalty")
        ) {
            return error.message
        }
        if case .failure(let error) = SamplingValidator.penalty(
            presencePenalty,
            name: String(localized: "settings.validator.presence_penalty.name",
                         defaultValue: "Presence Penalty",
                         comment: "Field name embedded in validation errors for presence penalty")
        ) {
            return error.message
        }
        return nil
    }

    /// Save the current working settings as a new profile (model scope)
    /// or template (global scope), then activate it. Used by both the
    /// Active Profile banner's "Save as new" and a chip group's
    /// "Save current as new" pill.
    func saveWorkingAs(scope: ProfileScope, name: String, client: OMLXClient) async {
        let targetModelID = modelID
        let displayName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        let cleanName = "p-" + UUID().uuidString.lowercased().prefix(28)
        guard !displayName.isEmpty, scope != .preset else { return }
        guard validateWorkingSettings() else { return }
        let settings = currentSettingsDict()
        do {
            switch scope {
            case .global:
                _ = try await client.createProfileTemplate(
                    body: CreateTemplateRequest(
                        name: cleanName,
                        displayName: displayName,
                        description: nil,
                        settings: settings
                    )
                )
            case .model:
                _ = try await client.createModelProfile(
                    id: targetModelID,
                    body: CreateProfileRequest(
                        name: cleanName,
                        displayName: displayName,
                        settings: settings
                    )
                )
            case .preset:
                return
            }
            if scope == .global {
                _ = try await client.applyModelTemplate(id: targetModelID, name: cleanName)
            } else {
                _ = try await client.applyModelProfile(id: targetModelID, name: cleanName)
            }
            if modelID == targetModelID {
                await load(modelID: targetModelID, client: client)
            }
        } catch {
            self.lastError = error.omlxDescription
        }
    }

    /// Overwrite an existing profile/template with the current working
    /// settings. Used by the Active Profile banner's "Update X" and the
    /// ProfileDetailCard preview's "Update with working" button.
    func updateProfileWithWorking(scope: ProfileScope, name: String, client: OMLXClient) async {
        let targetModelID = modelID
        guard scope != .preset else { return }
        guard validateWorkingSettings() else { return }
        let settings = currentSettingsDict()
        do {
            switch scope {
            case .global:
                _ = try await client.updateProfileTemplate(
                    name: name,
                    body: UpdateTemplateRequest(settings: settings)
                )
            case .model:
                _ = try await client.updateModelProfile(
                    id: targetModelID,
                    name: name,
                    body: UpdateProfileRequest(settings: settings)
                )
            case .preset:
                return
            }
            // If this profile is the active one, re-apply so the runtime
            // picks up the new values; if not, just reload.
            if activeProfileName == name && activeProfileScope == scope {
                if scope == .global {
                    _ = try await client.applyModelTemplate(id: targetModelID, name: name)
                } else {
                    _ = try await client.applyModelProfile(id: targetModelID, name: name)
                }
            }
            if modelID == targetModelID {
                await load(modelID: targetModelID, client: client)
            }
        } catch {
            self.lastError = error.omlxDescription
        }
    }

    private func validateWorkingSettings() -> Bool {
        if moeExpertOffloadSupported, moeExpertOffloadEnabled,
           moeExpertOffloadResidentFraction == nil {
            lastError = moeResidentRangeError
            return false
        }
        return validateAneWorkingSettings()
    }

    private func validateAneWorkingSettings() -> Bool {
        guard qwen35AnePrefillEnabled else { return true }
        if model?.anePrefillBackend == "k2" {
            guard let width = Int(qwen35AnePrefillSequenceLength), width >= 32, width % 32 == 0,
                  let dense = Double(qwen35AnePrefillFraction), dense > 0, dense <= 1,
                  let shared = Double(qwen35AnePrefillSharedFraction), shared >= 0, shared <= 1 else {
                lastError = "ANE requires a tile divisible by 32, dense share in (0, 1], and shared share in [0, 1]."
                return false
            }
            return true
        }

        switch QwenAneSettingsValidator.promptBlock(qwen35AnePrefillSequenceLength) {
        case .success: break
        case .failure(let error): lastError = error.message; return false
        }
        switch QwenAneSettingsValidator.tailPadding(
            qwen35AnePrefillTailPaddingMinTokens,
            sequenceLength: qwen35AnePrefillSequenceLength
        ) {
        case .success: break
        case .failure(let error): lastError = error.message; return false
        }
        switch QwenAneSettingsValidator.mlpFraction(
            qwen35AnePrefillFraction,
            cpuFraction: qwen35AnePrefillCpuEnabled ? qwen35AnePrefillCpuFraction : "0"
        ) {
        case .success: break
        case .failure(let error): lastError = error.message; return false
        }
        switch QwenAneSettingsValidator.mlpLayers(qwen35AnePrefillMaxLayers) {
        case .success: break
        case .failure(let error): lastError = error.message; return false
        }
        guard qwen35AnePrefillGdn else { return true }
        switch QwenAneSettingsValidator.gdnFraction(
            qwen35AnePrefillGdnFraction,
            cpuFraction: qwen35AnePrefillCpuEnabled ? qwen35AnePrefillCpuGdnFraction : "0"
        ) {
        case .success: break
        case .failure(let error): lastError = error.message; return false
        }
        switch QwenAneSettingsValidator.gdnLayers(qwen35AnePrefillGdnMaxLayers) {
        case .success: return true
        case .failure(let error): lastError = error.message; return false
        }
    }

    /// Discard working changes by reloading the server's view.
    func revertWorking(client: OMLXClient) async {
        await load(modelID: modelID, client: client)
    }

    /// Suggest a unique default name for the Save-as popover.
    func suggestSaveAsName() -> String {
        let base: String
        if case .working(let basedOn) = activeProfileState, let basedOn {
            base = "\(profileDisplayName(scope: basedOn.scope, name: basedOn.name))-copy"
        } else {
            base = "profile-1"
        }
        let taken = Set(
            templates.map(\.displayName) + profiles.map(\.displayName)
        )
        if !taken.contains(base) { return base }
        var n = 2
        let trimmed = base.replacingOccurrences(
            of: #"-\d+$"#, with: "", options: .regularExpression
        )
        var candidate = "\(trimmed)-\(n)"
        while taken.contains(candidate) {
            n += 1
            candidate = "\(trimmed)-\(n)"
        }
        return candidate
    }

    func applyProfile(name: String, client: OMLXClient) async {
        do {
            _ = try await client.applyModelProfile(id: modelID, name: name)
            await load(modelID: modelID, client: client)
        } catch {
            self.lastError = error.omlxDescription
        }
    }

    func createProfile(name: String, client: OMLXClient) async {
        do {
            _ = try await client.createModelProfile(
                id: modelID,
                body: CreateProfileRequest(
                    name: name, displayName: name
                )
            )
            self.profiles = (try? await client.listModelProfiles(id: modelID).profiles) ?? []
        } catch {
            self.lastError = error.omlxDescription
        }
    }

    func deleteProfile(name: String, client: OMLXClient) async {
        guard name != "default" else { return }
        do {
            _ = try await client.deleteModelProfile(id: modelID, name: name)
            self.profiles = (try? await client.listModelProfiles(id: modelID).profiles) ?? []
            if activeProfileName == name {
                activeProfileName = "default"
            }
        } catch {
            self.lastError = error.omlxDescription
        }
    }

    func applyTemplate(template: ProfileDTO, client: OMLXClient) async {
        await applyChip(scope: .global, name: template.name, client: client)
    }

    /// `4.0` → `"4"`, `2.5` → `"2.5"`. The TurboQuant Popup options are
    /// declared as strings; preserving an integral display avoids the
    /// "4.0" mismatch that would prevent the option from highlighting.
    fileprivate static func formatBits(_ v: Double) -> String {
        v.rounded() == v ? String(Int(v)) : String(v)
    }

    /// Keep persisted fractions concise and stable when moving between the
    /// server DTO and editable text fields.
    static func formatPct(_ v: Double) -> String {
        var formatted = String(format: "%.6f", v)
        while formatted.last == "0" { formatted.removeLast() }
        if formatted.last == "." { formatted.removeLast() }
        return formatted
    }
}

enum QwenAneSettingsValidator {
    static func promptBlock(_ raw: String) -> Result<Int, SamplingValidationError> {
        integer(raw, label: "ANE prompt block") { value in
            value >= 1024 && value.isMultiple(of: 64)
                ? nil : "ANE prompt block must be a multiple of 64 and at least 1024."
        }
    }

    static func tailPadding(
        _ raw: String, sequenceLength: String
    ) -> Result<Int, SamplingValidationError> {
        let block: Int
        switch promptBlock(sequenceLength) {
        case .failure(let error): return .failure(error)
        case .success(let value): block = value
        }
        return integer(raw, label: "ANE tail padding threshold") { value in
            value >= 0 && value < block
                ? nil : "ANE tail padding threshold must be zero or less than the prompt block."
        }
    }

    static func mlpFraction(
        _ raw: String, cpuFraction: String
    ) -> Result<Double, SamplingValidationError> {
        switch fraction(raw, label: "MLP ANE fraction", range: 0.05...0.90) {
        case .failure(let error): return .failure(error)
        case .success(let value):
            let cpu: Double
            switch fraction(cpuFraction, label: "CPU MLP fraction", range: 0...0.25) {
            case .failure(let error): return .failure(error)
            case .success(let parsed): cpu = parsed
            }
            guard value + cpu < 1 else {
                return .failure(.init(message: "MLP ANE and CPU fractions must total less than 1.0."))
            }
            return .success(value)
        }
    }

    static func gdnFraction(
        _ raw: String, cpuFraction: String = "0"
    ) -> Result<Double, SamplingValidationError> {
        switch fraction(raw, label: "GDN ANE fraction", range: 0.05...0.90) {
        case .failure(let error): return .failure(error)
        case .success(let value):
            switch fraction(cpuFraction, label: "CPU GDN fraction", range: 0...0.50) {
            case .failure(let error): return .failure(error)
            case .success(let cpu):
                guard value + cpu < 1 else {
                    return .failure(.init(message: "GDN ANE and CPU fractions must total less than 1.0."))
                }
                return .success(value)
            }
        }
    }

    static func cpuFraction(
        _ raw: String, mlpFraction: String
    ) -> Result<Double, SamplingValidationError> {
        switch fraction(raw, label: "CPU MLP fraction", range: 0...0.25) {
        case .failure(let error): return .failure(error)
        case .success(let value):
            let ane: Double
            switch fraction(mlpFraction, label: "MLP ANE fraction", range: 0.05...0.90) {
            case .failure(let error): return .failure(error)
            case .success(let parsed): ane = parsed
            }
            guard value + ane < 1 else {
                return .failure(.init(message: "MLP ANE and CPU fractions must total less than 1.0."))
            }
            return .success(value)
        }
    }

    static func cpuDownFraction(_ raw: String) -> Result<Double, SamplingValidationError> {
        fraction(raw, label: "CPU MLP down fraction", range: 0...0.50)
    }

    static func cpuGdnFraction(
        _ raw: String, gdnFraction: String
    ) -> Result<Double, SamplingValidationError> {
        switch fraction(raw, label: "CPU GDN fraction", range: 0...0.50) {
        case .failure(let error): return .failure(error)
        case .success(let value):
            switch fraction(gdnFraction, label: "GDN ANE fraction", range: 0.05...0.90) {
            case .failure(let error): return .failure(error)
            case .success(let ane):
                guard value + ane < 1 else {
                    return .failure(.init(message: "GDN ANE and CPU fractions must total less than 1.0."))
                }
                return .success(value)
            }
        }
    }

    static func cpuThreads(_ raw: String) -> Result<Int, SamplingValidationError> {
        integer(raw, label: "CPU worker count") { (0...64).contains($0)
            ? nil : "CPU worker count must be between 0 and 64."
        }
    }

    static func mlpLayers(_ raw: String) -> Result<Int, SamplingValidationError> {
        integer(raw, label: "ANE MLP layer limit") { $0 >= 1
            ? nil : "ANE MLP layer limit must be positive."
        }
    }

    static func gdnLayers(_ raw: String) -> Result<Int, SamplingValidationError> {
        integer(raw, label: "ANE GDN layer limit") { $0 >= 0
            ? nil : "ANE GDN layer limit must be zero or greater."
        }
    }

    private static func fraction(
        _ raw: String, label: String, range: ClosedRange<Double>
    ) -> Result<Double, SamplingValidationError> {
        let trimmed = raw.trimmingCharacters(in: .whitespaces)
        guard let value = Double(trimmed), value.isFinite else {
            return .failure(.init(message: "\(label) must be a number."))
        }
        guard range.contains(value) else {
            return .failure(.init(message: "\(label) must be between \(range.lowerBound) and \(range.upperBound)."))
        }
        return .success(value)
    }

    private static func integer(
        _ raw: String, label: String, check: (Int) -> String?
    ) -> Result<Int, SamplingValidationError> {
        let trimmed = raw.trimmingCharacters(in: .whitespaces)
        guard let value = Int(trimmed) else {
            return .failure(.init(message: "\(label) must be an integer."))
        }
        if let message = check(value) {
            return .failure(.init(message: message))
        }
        return .success(value)
    }
}

/// Sheet state for the header snapshot actions on ModelSettingsScreen.
/// `optimalCandidates(nil)` is the loading state while omlx.ai is queried.
enum SettingsApplyOutcome {
    case recipeInput
    case optimalCandidates(OptimalCandidatesDTO?)
    case optimal(SettingsApplyResultDTO)
    case recipe(SettingsApplyResultDTO)
}

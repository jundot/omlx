import Foundation
import SwiftUI

@MainActor
@Observable
final class ModelSettingsScreenVM {
    enum Section: String, Hashable, CaseIterable, Sendable {
        case profiles, basic, advanced

        var label: String {
            switch self {
            case .profiles:
                return String(localized: "settings.section.profiles",
                              defaultValue: "Profiles",
                              comment: "Segmented control label for the Profiles tab")
            case .basic:
                return String(localized: "settings.section.basic",
                              defaultValue: "Basic",
                              comment: "Segmented control label for the Basic tab")
            case .advanced:
                return String(localized: "settings.section.advanced",
                              defaultValue: "Advanced",
                              comment: "Segmented control label for the Advanced tab")
            }
        }
    }

    enum Field: Sendable {
        case alias, modelType, contextLength, maxTokens
        case temperature, topP, topK, minP
        case repetitionPenalty, presencePenalty, ttl
        case enableThinking, qwen4PleSsdOffload
        case thinkingBudgetEnabled, thinkingBudgetTokens
        case limitToolResults, toolResultLimitTokens
        case forceSampling, isPinned, isFavorite
        case trustRemoteCode
        case reasoningParser
        case chatTemplateKwargs
        case turboquantKvEnabled, turboquantKvBits
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
        case dflashEnabled, dflashDraftModel, dflashMaxCtx
        case dflashDraftQuantEnabled, dflashDraftQuantWeightBits
        case dflashDraftQuantActivationBits, dflashDraftQuantGroupSize
        case dflashVerifyMode, dflashDraftWindowSize, dflashDraftSinkSize, dflashBlockSize
        case dflashInMemoryCache, dflashInMemoryCacheGib, dflashInMemoryCacheMaxEntries
        case dflashSsdCache, dflashSsdCacheGib
        case mtpEnabled
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

    static var expertStreamingModeOptions: [(String, String)] {
        [
            ("soft_reap", String(localized: "settings.expert_streaming.mode.soft_reap",
                                 defaultValue: "Soft-REAP pinning",
                                 comment: "SSD expert streaming mode that pins experts selected by a manifest")),
            ("cache_only", String(localized: "settings.expert_streaming.mode.cache_only",
                                  defaultValue: "Analytical cache only",
                                  comment: "SSD expert streaming mode with no automatically pinned experts")),
        ]
    }

    static var expertStreamingCachePolicyOptions: [(String, String)] {
        [
            ("route_frequency", String(localized: "settings.expert_streaming.cache_policy.frequency",
                                      defaultValue: "Route frequency",
                                      comment: "SSD expert cache policy based on decayed route frequency")),
            ("lru", String(localized: "settings.expert_streaming.cache_policy.lru",
                           defaultValue: "Least recently used",
                           comment: "SSD expert cache policy based on recent access")),
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

    static var dflashDraftQuantActivationBitsOptions: [(String, String)] {
        [
            ("", String(localized: "settings.dflash.quant.activation.default",
                        defaultValue: "default",
                        comment: "DFlash draft quantization activation bits — use server default")),
            ("16", String(localized: "settings.dflash.quant.activation.16",
                          defaultValue: "16-bit",
                          comment: "DFlash draft quantization activation bits option")),
            ("32", String(localized: "settings.dflash.quant.activation.32",
                          defaultValue: "32-bit",
                          comment: "DFlash draft quantization activation bits option")),
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
            ("ddtree", String(localized: "settings.dflash.verify_mode.ddtree",
                              defaultValue: "ddtree",
                              comment: "DFlash verify mode: DDTree verifier")),
            ("off", String(localized: "settings.dflash.verify_mode.off",
                           defaultValue: "off",
                           comment: "DFlash verify mode: disable speculative verify")),
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

    var section: Section = .basic

    var model: ModelDTO?
    /// Snapshot of every other model on the server, used to populate the
    /// SpecPrefill / DFlash draft-model dropdowns. Reloaded with `load()`.
    var allModels: [ModelDTO] = []
    var modelID: String = ""
    var lastError: String?

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
    var expertStreamingEnabled: Bool = false
    var expertStreamingMode: String = "soft_reap"
    var expertStreamingManifest: String = ""
    var expertStreamingCacheExperts: String = "32"
    var expertStreamingScratchExperts: String = "32"
    var expertStreamingCachePolicy: String = "route_frequency"
    var expertStreamingFastResourceLoading: Bool = true
    var expertStreamingDirectIo: Bool = true
    var expertStreamingNativeDemand: Bool = true
    var expertStreamingDecodeScratchAsCache: Bool = true
    var expertStreamingIoCoalescingKib: String = "64"
    var expertManifestUploadInProgress: Bool = false
    var expertManifestSummary: String?
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

    // Reasoning parser (free-form override; empty = auto)
    var reasoningParser: String = ""

    // Chat-template kwargs — entries are the editor's view of the
    // (chat_template_kwargs, forced_ct_kwargs) server pair.
    var chatTemplateEntries: [ChatTemplateKwargEntry] = []

    // Experimental: TurboQuant KV
    var turboquantKvEnabled: Bool = false
    var turboquantKvBits: String = "4"

    // Experimental: private Qwen3.5/3.6/3.8 ANE/GPU fixed-shape prefill.
    // These defaults are the measured M3 Ultra optimum for the 2,048-token
    // benchmark path. The feature itself remains opt-in.
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
    var dflashDraftQuantActivationBits: String = ""
    var dflashDraftQuantGroupSize: String = ""
    var dflashMaxCtx: String = ""
    var dflashVerifyMode: String = ""
    var dflashDraftWindowSize: String = ""
    var dflashDraftSinkSize: String = "0"
    var dflashBlockSize: String = ""
    var dflashInMemoryCache: Bool = false
    var dflashInMemoryCacheGib: String = "8"
    var dflashInMemoryCacheMaxEntries: String = "4"
    var dflashSsdCache: Bool = false
    var dflashSsdCacheGib: String = "20"

    // Experimental: native MTP
    var mtpEnabled: Bool = false

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
    /// True when one or more profile-eligible fields have been edited
    /// since the last load / apply / save. Flips the screen into the
    /// "Working profile" state. Per-model fields (alias / modelType /
    /// ttl / isPinned / trustRemoteCode) auto-save and never set this.
    var profileDirty: Bool = false

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

    var isDiffusionModel: Bool {
        let type = (model?.configModelType ?? "")
            .lowercased()
            .replacingOccurrences(of: "-", with: "_")
        return Self.diffusionConfigModelTypes.contains(type)
    }

    var isQwen4Exp: Bool {
        (model?.configModelType ?? "")
            .lowercased()
            .replacingOccurrences(of: "-", with: "_") == "qwen4_exp"
    }

    var expertManifestFileName: String {
        guard !expertStreamingManifest.isEmpty else { return "" }
        return (expertStreamingManifest as NSString).lastPathComponent
    }

    private func isDiffusionUnsupportedField(_ field: Field) -> Bool {
        switch field {
        case .topP, .topK, .minP, .repetitionPenalty, .presencePenalty:
            return true
        case .enableThinking, .qwen4PleSsdOffload,
             .thinkingBudgetEnabled, .thinkingBudgetTokens:
            return true
        case .limitToolResults, .toolResultLimitTokens:
            return true
        case .forceSampling, .reasoningParser:
            return true
        case .turboquantKvEnabled, .turboquantKvBits:
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
        case .dflashEnabled, .dflashDraftModel, .dflashMaxCtx:
            return true
        case .dflashDraftQuantEnabled, .dflashDraftQuantWeightBits:
            return true
        case .dflashDraftQuantActivationBits, .dflashDraftQuantGroupSize:
            return true
        case .dflashVerifyMode, .dflashDraftWindowSize, .dflashDraftSinkSize, .dflashBlockSize:
            return true
        case .dflashInMemoryCache, .dflashInMemoryCacheGib:
            return true
        case .dflashInMemoryCacheMaxEntries:
            return true
        case .dflashSsdCache, .dflashSsdCacheGib:
            return true
        case .mtpEnabled, .vlmMtpEnabled, .vlmMtpDraftModel:
            return true
        case .vlmMtpDraftBlockSize:
            return true
        case .alias, .modelType, .contextLength, .maxTokens:
            return false
        case .temperature, .ttl, .isPinned, .isFavorite, .trustRemoteCode:
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

    /// Binding helper for profile-eligible fields. Edits flip
    /// `profileDirty` (which activates the Working banner) instead of
    /// firing a per-field PUT. Network writes happen only when the user
    /// chooses Apply / Save as new / Update.
    func bindProfile<T: Equatable>(_ binding: Binding<T>) -> Binding<T> {
        Binding(
            get: { binding.wrappedValue },
            set: { newValue in
                let changed = binding.wrappedValue != newValue
                binding.wrappedValue = newValue
                if changed { self.profileDirty = true }
            }
        )
    }

    /// Flip the working-dirty flag from a non-binding callsite (e.g. the
    /// chat-template kwargs editor's add / remove buttons).
    func markProfileDirty() { self.profileDirty = true }

    func load(modelID: String, client: OMLXClient) async {
        if self.modelID != modelID {
            aneTuningID = nil
            aneTuningIsRunning = false
            aneTuningStatus = nil
        }
        self.modelID = modelID
        do {
            let models = try await client.listModels().models
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
                self.expertStreamingEnabled = s?.expertStreamingEnabled ?? false
                self.expertStreamingMode = s?.expertStreamingMode ?? "soft_reap"
                self.expertStreamingManifest = s?.expertStreamingManifest ?? ""
                self.expertStreamingCacheExperts = s?.expertStreamingCacheExperts.map(String.init) ?? "32"
                self.expertStreamingScratchExperts = s?.expertStreamingScratchExperts.map(String.init) ?? "32"
                self.expertStreamingCachePolicy = s?.expertStreamingCachePolicy ?? "route_frequency"
                self.expertStreamingFastResourceLoading = s?.expertStreamingFastResourceLoading ?? true
                self.expertStreamingDirectIo = s?.expertStreamingDirectIo ?? true
                self.expertStreamingNativeDemand = s?.expertStreamingNativeDemand ?? true
                self.expertStreamingDecodeScratchAsCache = s?.expertStreamingDecodeScratchAsCache ?? true
                self.expertStreamingIoCoalescingKib = s?.expertStreamingIoCoalescingKib.map(String.init) ?? "64"
                self.expertManifestSummary = nil
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
                self.reasoningParser = s?.reasoningParser ?? ""
                self.chatTemplateEntries = diffusionCompatibleChatTemplateEntries(
                    ChatTemplateKwargsCodec.decode(
                        kwargs: s?.chatTemplateKwargs,
                        forced: s?.forcedCtKwargs
                    )
                )
                self.turboquantKvEnabled = s?.turboquantKvEnabled ?? false
                self.turboquantKvBits = s?.turboquantKvBits.map { Self.formatBits($0) } ?? "4"
                self.qwen35AnePrefillEnabled = s?.qwen35AnePrefillEnabled ?? false
                self.qwen35AnePrefillSequenceLength = s?.qwen35AnePrefillSequenceLength.map(String.init) ?? "2048"
                self.qwen35AnePrefillTailPaddingMinTokens = s?.qwen35AnePrefillTailPaddingMinTokens.map(String.init) ?? "0"
                self.qwen35AnePrefillFraction = s?.qwen35AnePrefillFraction.map { Self.formatPct($0) } ?? "0.53"
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
                self.dflashDraftQuantActivationBits = s?.dflashDraftQuantActivationBits.map(String.init) ?? ""
                self.dflashDraftQuantGroupSize = s?.dflashDraftQuantGroupSize.map(String.init) ?? ""
                self.dflashMaxCtx = s?.dflashMaxCtx.map(String.init) ?? ""
                self.dflashVerifyMode = s?.dflashVerifyMode ?? ""
                self.dflashDraftWindowSize = s?.dflashDraftWindowSize.map(String.init) ?? ""
                self.dflashDraftSinkSize = s?.dflashDraftSinkSize.map(String.init) ?? "0"
                self.dflashBlockSize = s?.dflashBlockSize.map(String.init) ?? ""
                self.dflashInMemoryCache = s?.dflashInMemoryCache ?? false
                self.dflashInMemoryCacheGib = DflashByteSize.bytesToGib(s?.dflashInMemoryCacheMaxBytes)
                    .map(String.init) ?? "8"
                self.dflashInMemoryCacheMaxEntries = s?.dflashInMemoryCacheMaxEntries.map(String.init) ?? "4"
                self.dflashSsdCache = s?.dflashSsdCache ?? false
                self.dflashSsdCacheGib = DflashByteSize.bytesToGib(s?.dflashSsdCacheMaxBytes)
                    .map(String.init) ?? "20"
                self.mtpEnabled = s?.mtpEnabled ?? false
                self.vlmMtpEnabled = s?.vlmMtpEnabled ?? false
                self.vlmMtpDraftModel = s?.vlmMtpDraftModel ?? ""
                self.vlmMtpDraftBlockSize = s?.vlmMtpDraftBlockSize.map(String.init) ?? ""
                self.activeProfileName = s?.activeProfileName
            }
            self.profiles = (try? await client.listModelProfiles(id: modelID).profiles) ?? []
            self.templates = (try? await client.listProfileTemplates().templates) ?? []
            self.serverDefaultSampling = (try? await client.getGlobalSettings().sampling)
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
            self.profileDirty = false
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
        case .qwen4PleSsdOffload:
            guard isQwen4Exp, qwen4PleSsdOffloadSupported,
                  !qwen4PleSsdOffloadForced else { return }
            patch.qwen4PleSsdOffload = qwen4PleSsdOffload
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
        case .reasoningParser:
            patch.reasoningParser = reasoningParser.isEmpty ? nil : reasoningParser
        case .chatTemplateKwargs:
            let pair = ChatTemplateKwargsCodec.encode(
                diffusionCompatibleChatTemplateEntries(chatTemplateEntries)
            )
            patch.chatTemplateKwargs = pair.kwargs ?? [:]
            patch.forcedCtKwargs = pair.forced ?? []
        case .turboquantKvEnabled:     patch.turboquantKvEnabled = turboquantKvEnabled
        case .turboquantKvBits:        patch.turboquantKvBits = Double(turboquantKvBits)
        case .qwen35AnePrefillEnabled: patch.qwen35AnePrefillEnabled = qwen35AnePrefillEnabled
        case .qwen35AnePrefillSequenceLength:
            switch QwenAneSettingsValidator.promptBlock(qwen35AnePrefillSequenceLength) {
            case .success(let value): patch.qwen35AnePrefillSequenceLength = value
            case .failure(let error): lastError = error.message; return
            }
        case .qwen35AnePrefillTailPaddingMinTokens:
            switch QwenAneSettingsValidator.tailPadding(
                qwen35AnePrefillTailPaddingMinTokens,
                sequenceLength: qwen35AnePrefillSequenceLength
            ) {
            case .success(let value): patch.qwen35AnePrefillTailPaddingMinTokens = value
            case .failure(let error): lastError = error.message; return
            }
        case .qwen35AnePrefillFraction:
            switch QwenAneSettingsValidator.mlpFraction(
                qwen35AnePrefillFraction,
                cpuFraction: qwen35AnePrefillCpuEnabled ? qwen35AnePrefillCpuFraction : "0"
            ) {
            case .success(let value): patch.qwen35AnePrefillFraction = value
            case .failure(let error): lastError = error.message; return
            }
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
                patch.dflashDraftQuantActivationBits = nil
                patch.dflashDraftQuantGroupSize = nil
            }
        case .dflashDraftQuantWeightBits:
            patch.dflashDraftQuantWeightBits = Int(dflashDraftQuantWeightBits)
        case .dflashDraftQuantActivationBits:
            patch.dflashDraftQuantActivationBits = Int(dflashDraftQuantActivationBits)
        case .dflashDraftQuantGroupSize:
            patch.dflashDraftQuantGroupSize = Int(dflashDraftQuantGroupSize)
        case .dflashMaxCtx:            patch.dflashMaxCtx = Int(dflashMaxCtx)
        case .dflashVerifyMode:        patch.dflashVerifyMode = dflashVerifyMode.isEmpty ? nil : dflashVerifyMode
        case .dflashDraftWindowSize:   patch.dflashDraftWindowSize = Int(dflashDraftWindowSize)
        case .dflashDraftSinkSize:     patch.dflashDraftSinkSize = Int(dflashDraftSinkSize)
        case .dflashBlockSize:         patch.dflashBlockSize = Int(dflashBlockSize)
        case .dflashInMemoryCache:
            patch.dflashInMemoryCache = dflashInMemoryCache
            if !dflashInMemoryCache {
                // Mirror the HTML editor: turning the L1 cache off also
                // disables the L2 (SSD) sub-toggle.
                dflashSsdCache = false
                patch.dflashSsdCache = false
            }
        case .dflashInMemoryCacheGib:
            patch.dflashInMemoryCacheMaxBytes = DflashByteSize.gibToBytes(Int(dflashInMemoryCacheGib))
        case .dflashInMemoryCacheMaxEntries:
            patch.dflashInMemoryCacheMaxEntries = Int(dflashInMemoryCacheMaxEntries)
        case .dflashSsdCache:          patch.dflashSsdCache = dflashSsdCache
        case .dflashSsdCacheGib:
            patch.dflashSsdCacheMaxBytes = DflashByteSize.gibToBytes(Int(dflashSsdCacheGib))
        case .mtpEnabled:              patch.mtpEnabled = mtpEnabled
        case .vlmMtpEnabled:           patch.vlmMtpEnabled = vlmMtpEnabled
        case .vlmMtpDraftModel:        patch.vlmMtpDraftModel = vlmMtpDraftModel.isEmpty ? nil : vlmMtpDraftModel
        case .vlmMtpDraftBlockSize:    patch.vlmMtpDraftBlockSize = Int(vlmMtpDraftBlockSize)
        }
        do {
            _ = try await client.updateModelSettings(id: modelID, patch: patch)
            self.lastError = nil
        } catch {
            self.lastError = error.omlxDescription
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

    func uploadExpertManifest(url: URL, client: OMLXClient) async {
        guard !expertManifestUploadInProgress else { return }
        expertManifestUploadInProgress = true
        defer { expertManifestUploadInProgress = false }

        let accessed = url.startAccessingSecurityScopedResource()
        defer {
            if accessed { url.stopAccessingSecurityScopedResource() }
        }

        do {
            let fileSize = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize
            guard (fileSize ?? 0) <= 2 * 1024 * 1024 else {
                lastError = String(localized: "settings.expert_streaming.manifest.too_large",
                                   defaultValue: "The expert manifest must be 2 MiB or smaller.",
                                   comment: "Validation error when an expert manifest exceeds the upload limit")
                return
            }
            let data = try Data(contentsOf: url, options: .mappedIfSafe)
            guard data.count <= 2 * 1024 * 1024 else {
                lastError = String(localized: "settings.expert_streaming.manifest.too_large",
                                   defaultValue: "The expert manifest must be 2 MiB or smaller.",
                                   comment: "Validation error when an expert manifest exceeds the upload limit")
                return
            }
            let response = try await client.uploadExpertManifest(
                id: modelID,
                fileName: url.lastPathComponent,
                data: data
            )
            expertStreamingManifest = response.manifestPath
            expertManifestSummary = String(
                localized: "settings.expert_streaming.manifest.summary",
                defaultValue: "\(response.layers) layers · \(response.pinnedCountMin)–\(response.pinnedCountMax) pinned experts per layer",
                comment: "Summary shown after a Soft-REAP expert manifest is validated and uploaded"
            )
            profileDirty = true
            lastError = nil
        } catch {
            lastError = error.omlxDescription
        }
    }

    /// Stage the best tuner result in the working profile. The user can then
    /// update the active profile or save it as a new one without detaching the
    /// model from its current profile via a direct settings write.
    func applyANETuningRecommendation() {
        guard let recommendation = aneTuningStatus?.recommendation else { return }
        qwen35AnePrefillEnabled = recommendation.enabled
        qwen35AnePrefillSequenceLength = String(recommendation.sequenceLength)
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
        profileDirty = true
        lastError = nil
    }

    // MARK: - Chat-template kwarg list mutation

    func addKwarg(_ kind: ChatTemplateKwargEntryKind) {
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
        case .reasoningEffort: defaultValue = "low"
        case .custom:          defaultValue = ""
        }
        chatTemplateEntries.append(
            ChatTemplateKwargEntry(kind: kind, value: defaultValue)
        )
        markProfileDirty()
    }

    func removeKwarg(id: UUID) {
        chatTemplateEntries.removeAll(where: { $0.id == id })
        markProfileDirty()
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

    var isQwen35AnePrefillModel: Bool {
        guard let rawType = model?.configModelType else { return false }
        let type = rawType.lowercased().replacingOccurrences(of: "-", with: "_")
        return type.hasPrefix("qwen3_5")
            || type.hasPrefix("qwen3_6")
            || type.hasPrefix("qwen3_8")
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
        return nil
    }

    /// VLM MTP wraps mlx-vlm's MTP loop and is mutually exclusive with the
    /// other speculative-decoding / KV-quant features. Mirrors the HTML
    /// editor's gating so the toggle disables itself and surfaces why.
    var vlmMtpConflictReason: String? {
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
                          defaultValue: "Unset repetition / presence penalty before enabling VLM MTP.",
                          comment: "Tooltip / sublabel shown when VLM MTP can't be enabled because penalty settings are set")
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
        return false
    }

    /// Sublabel / tooltip for the sampling rows locked while VLM MTP is on.
    var vlmMtpProcessorLockedReason: String {
        String(localized: "settings.sampling.locked.vlm_mtp",
               defaultValue: "Disable VLM MTP to edit this setting.",
               comment: "Tooltip / sublabel shown when a penalty or thinking-budget row is locked because VLM MTP is on")
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
            putBool(ProfileSettingsKey.enableThinking, enableThinking)
            putBool(ProfileSettingsKey.thinkingBudgetEnabled, thinkingBudgetEnabled)
            putInt(ProfileSettingsKey.thinkingBudgetTokens, thinkingBudgetTokens)
            putBool(ProfileSettingsKey.forceSampling, forceSampling)
            putString(ProfileSettingsKey.reasoningParser, reasoningParser)
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
            putBool(ProfileSettingsKey.expertStreamingEnabled, expertStreamingEnabled)
            if expertStreamingEnabled {
                putString(ProfileSettingsKey.expertStreamingMode, expertStreamingMode)
                if expertStreamingMode == "soft_reap" {
                    putString(ProfileSettingsKey.expertStreamingManifest, expertStreamingManifest)
                }
                putInt(ProfileSettingsKey.expertStreamingCacheExperts, expertStreamingCacheExperts)
                putInt(ProfileSettingsKey.expertStreamingScratchExperts, expertStreamingScratchExperts)
                putBool(
                    ProfileSettingsKey.expertStreamingFastResourceLoading,
                    expertStreamingFastResourceLoading
                )
                putString(
                    ProfileSettingsKey.expertStreamingCachePolicy,
                    expertStreamingCachePolicy
                )
                putBool(ProfileSettingsKey.expertStreamingDirectIo, expertStreamingDirectIo)
                putBool(ProfileSettingsKey.expertStreamingNativeDemand, expertStreamingNativeDemand)
                putBool(
                    ProfileSettingsKey.expertStreamingDecodeScratchAsCache,
                    expertStreamingDecodeScratchAsCache
                )
                putInt(ProfileSettingsKey.expertStreamingIoCoalescingKib, expertStreamingIoCoalescingKib)
            }
            putBool(ProfileSettingsKey.turboquantKvEnabled, turboquantKvEnabled)
            if turboquantKvEnabled, let bits = Double(turboquantKvBits) {
                out[ProfileSettingsKey.turboquantKvBits] = AnyCodable(bits)
            }
            putBool(ProfileSettingsKey.qwen35AnePrefillEnabled, qwen35AnePrefillEnabled)
            if qwen35AnePrefillEnabled {
                putInt(ProfileSettingsKey.qwen35AnePrefillSequenceLength, qwen35AnePrefillSequenceLength)
                putInt(ProfileSettingsKey.qwen35AnePrefillTailPaddingMinTokens, qwen35AnePrefillTailPaddingMinTokens)
                putDouble(ProfileSettingsKey.qwen35AnePrefillFraction, qwen35AnePrefillFraction)
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
                    putInt(ProfileSettingsKey.dflashDraftQuantActivationBits, dflashDraftQuantActivationBits)
                    putInt(ProfileSettingsKey.dflashDraftQuantGroupSize, dflashDraftQuantGroupSize)
                }
                putInt(ProfileSettingsKey.dflashMaxCtx, dflashMaxCtx)
                if !dflashVerifyMode.isEmpty {
                    out[ProfileSettingsKey.dflashVerifyMode] = AnyCodable(dflashVerifyMode)
                }
                putInt(ProfileSettingsKey.dflashDraftWindowSize, dflashDraftWindowSize)
                putInt(ProfileSettingsKey.dflashDraftSinkSize, dflashDraftSinkSize)
                putInt(ProfileSettingsKey.dflashBlockSize, dflashBlockSize)
                putBool(ProfileSettingsKey.dflashInMemoryCache, dflashInMemoryCache)
                if dflashInMemoryCache {
                    if let bytes = DflashByteSize.gibToBytes(Int(dflashInMemoryCacheGib)) {
                        out[ProfileSettingsKey.dflashInMemoryCacheMaxBytes] = AnyCodable(Int(bytes))
                    }
                    putInt(ProfileSettingsKey.dflashInMemoryCacheMaxEntries, dflashInMemoryCacheMaxEntries)
                }
                putBool(ProfileSettingsKey.dflashSsdCache, dflashSsdCache)
                if dflashSsdCache, let bytes = DflashByteSize.gibToBytes(Int(dflashSsdCacheGib)) {
                    out[ProfileSettingsKey.dflashSsdCacheMaxBytes] = AnyCodable(Int(bytes))
                }
            }
            putBool(ProfileSettingsKey.mtpEnabled, mtpEnabled)
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
        do {
            switch scope {
            case .preset:
                // Caller dispatches via applyPreset(_:client:) — this
                // branch is a defensive no-op so misrouted calls don't
                // hit a template lookup that's guaranteed to miss.
                return
            case .model:
                _ = try await client.applyModelProfile(id: modelID, name: name)
            case .global:
                // Templates aren't directly applicable — seed a model
                // profile from the template, then apply it. Reuse the
                // template's name; if a same-named model profile already
                // exists we leave it alone (server returns 409, we
                // silently fall through to apply).
                if !self.profiles.contains(where: { $0.name == name }) {
                    if let tpl = self.templates.first(where: { $0.name == name }) {
                        _ = try? await client.createModelProfile(
                            id: modelID,
                            body: CreateProfileRequest(
                                name: tpl.name,
                                displayName: tpl.displayName,
                                description: tpl.description,
                                sourceTemplate: tpl.name,
                                settings: tpl.settings
                            )
                        )
                    }
                }
                _ = try await client.applyModelProfile(id: modelID, name: name)
            }
            await load(modelID: modelID, client: client)
        } catch {
            self.lastError = error.omlxDescription
        }
    }

    /// Rename a global template via PUT /api/profile-templates/{name}.
    /// Server validates the slug + duplicate; we already pre-checked
    /// in ProfileGroup, but the server stays the source of truth for
    /// the activated state — reload after success.
    func renameTemplate(from original: String, to renamed: String, client: OMLXClient) async {
        do {
            _ = try await client.updateProfileTemplate(
                name: original,
                body: UpdateTemplateRequest(newName: renamed)
            )
            await load(modelID: modelID, client: client)
        } catch {
            self.lastError = error.omlxDescription
        }
    }

    /// Rename a per-model profile via PUT /api/models/{id}/profiles/{name}.
    /// If the renamed profile was active, the server carries the active
    /// pointer to the new name; reload to pick that up.
    func renameModelProfile(from original: String, to renamed: String, client: OMLXClient) async {
        do {
            _ = try await client.updateModelProfile(
                id: modelID,
                name: original,
                body: UpdateProfileRequest(newName: renamed)
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

    /// Save the current working settings as a new profile (model scope)
    /// or template (global scope), then activate it. Used by both the
    /// Active Profile banner's "Save as new" and a chip group's
    /// "Save current as new" pill.
    func saveWorkingAs(scope: ProfileScope, name: String, client: OMLXClient) async {
        let cleanName = name.trimmingCharacters(in: .whitespaces)
        guard !cleanName.isEmpty, scope != .preset else { return }
        guard validateQwenAneWorkingSettings(), validateExpertStreamingWorkingSettings() else { return }
        let settings = currentSettingsDict()
        do {
            switch scope {
            case .global:
                _ = try await client.createProfileTemplate(
                    body: CreateTemplateRequest(
                        name: cleanName,
                        displayName: cleanName,
                        description: nil,
                        settings: settings
                    )
                )
                // Seed a per-model profile from the new template and apply it.
                _ = try? await client.createModelProfile(
                    id: modelID,
                    body: CreateProfileRequest(
                        name: cleanName,
                        displayName: cleanName,
                        sourceTemplate: cleanName,
                        settings: settings
                    )
                )
            case .model:
                _ = try await client.createModelProfile(
                    id: modelID,
                    body: CreateProfileRequest(
                        name: cleanName,
                        displayName: cleanName,
                        settings: settings
                    )
                )
            case .preset:
                return
            }
            _ = try await client.applyModelProfile(id: modelID, name: cleanName)
            await load(modelID: modelID, client: client)
        } catch {
            self.lastError = error.omlxDescription
        }
    }

    /// Overwrite an existing profile/template with the current working
    /// settings. Used by the Active Profile banner's "Update X" and the
    /// ProfileDetailCard preview's "Update with working" button.
    func updateProfileWithWorking(scope: ProfileScope, name: String, client: OMLXClient) async {
        guard scope != .preset else { return }
        guard validateQwenAneWorkingSettings(), validateExpertStreamingWorkingSettings() else { return }
        let settings = currentSettingsDict()
        do {
            switch scope {
            case .global:
                _ = try await client.updateProfileTemplate(
                    name: name,
                    body: UpdateTemplateRequest(settings: settings)
                )
                // Update the same-named model profile too so the next
                // /apply lands the latest settings.
                if self.profiles.contains(where: { $0.name == name }) {
                    _ = try? await client.updateModelProfile(
                        id: modelID,
                        name: name,
                        body: UpdateProfileRequest(settings: settings)
                    )
                }
            case .model:
                _ = try await client.updateModelProfile(
                    id: modelID,
                    name: name,
                    body: UpdateProfileRequest(settings: settings)
                )
            case .preset:
                return
            }
            // If this profile is the active one, re-apply so the runtime
            // picks up the new values; if not, just reload.
            if activeProfileName == name {
                _ = try? await client.applyModelProfile(id: modelID, name: name)
            }
            await load(modelID: modelID, client: client)
        } catch {
            self.lastError = error.omlxDescription
        }
    }

    private func validateQwenAneWorkingSettings() -> Bool {
        guard qwen35AnePrefillEnabled else { return true }

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

    private func validateExpertStreamingWorkingSettings() -> Bool {
        guard expertStreamingEnabled else { return true }
        guard Self.expertStreamingModeOptions.contains(where: { $0.0 == expertStreamingMode }) else {
            lastError = String(localized: "settings.expert_streaming.validation.mode",
                               defaultValue: "Choose a valid SSD Expert Streaming mode.",
                               comment: "Validation error for an unknown SSD expert streaming mode")
            return false
        }
        if expertStreamingMode == "soft_reap",
           expertStreamingManifest.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            lastError = String(localized: "settings.expert_streaming.validation.manifest",
                               defaultValue: "Choose a Soft-REAP expert manifest before saving.",
                               comment: "Validation error when Soft-REAP mode has no expert manifest")
            return false
        }
        guard let cacheExperts = Int(expertStreamingCacheExperts), (0...512).contains(cacheExperts) else {
            lastError = String(localized: "settings.expert_streaming.validation.cache",
                               defaultValue: "Hot cache size must be a whole number from 0 to 512 experts per layer.",
                               comment: "Validation error for SSD expert streaming hot cache size")
            return false
        }
        guard let scratchExperts = Int(expertStreamingScratchExperts),
              (0...512).contains(scratchExperts) else {
            lastError = String(
                localized: "settings.expert_streaming.validation.scratch",
                defaultValue: "Cold execution size must be a whole number from 0 to 512 experts per layer.",
                comment: "Validation error for SSD expert streaming scratch size"
            )
            return false
        }
        guard Self.expertStreamingCachePolicyOptions.contains(where: {
            $0.0 == expertStreamingCachePolicy
        }) else {
            lastError = String(localized: "settings.expert_streaming.validation.cache_policy",
                               defaultValue: "Choose a valid SSD Expert Streaming cache policy.",
                               comment: "Validation error for an unknown SSD expert cache policy")
            return false
        }
        guard let coalescingKib = Int(expertStreamingIoCoalescingKib),
              (0...4096).contains(coalescingKib) else {
            lastError = String(localized: "settings.expert_streaming.validation.coalescing",
                               defaultValue: "I/O coalescing must be a whole number from 0 to 4096 KiB.",
                               comment: "Validation error for SSD expert I/O coalescing")
            return false
        }
        guard !expertStreamingNativeDemand || expertStreamingFastResourceLoading,
              !expertStreamingDirectIo || expertStreamingFastResourceLoading else {
            lastError = String(localized: "settings.expert_streaming.validation.fast_resource",
                               defaultValue: "Native demand and direct I/O require Fast Resource Loading.",
                               comment: "Validation error for dependent SSD expert streaming settings")
            return false
        }
        return true
    }

    /// Discard working changes by reloading the server's view.
    func revertWorking(client: OMLXClient) async {
        await load(modelID: modelID, client: client)
    }

    /// Suggest a unique default name for the Save-as popover.
    func suggestSaveAsName() -> String {
        let base: String
        if case .working(let basedOn) = activeProfileState, let basedOn {
            base = "\(basedOn.name)-copy"
        } else {
            base = "profile-1"
        }
        let taken = Set(
            templates.map(\.name) + profiles.map(\.name)
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
        do {
            _ = try await client.createModelProfile(
                id: modelID,
                body: CreateProfileRequest(
                    name: template.name,
                    displayName: template.displayName,
                    description: template.description,
                    sourceTemplate: template.name,
                    settings: template.settings
                )
            )
            self.profiles = (try? await client.listModelProfiles(id: modelID).profiles) ?? []
        } catch {
            self.lastError = error.omlxDescription
        }
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

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
        case enableThinking, thinkingBudgetEnabled, thinkingBudgetTokens
        case limitToolResults, toolResultLimitTokens
        case forceSampling, isPinned, isFavorite
        case trustRemoteCode
        case reasoningParser
        case chatTemplateKwargs
        case turboquantKvEnabled, turboquantKvBits
        case indexCacheEnabled, indexCacheFreq
        case specprefillEnabled, specprefillDraftModel, specprefillKeepPct, specprefillThreshold
        case dflashEnabled, dflashDraftModel, dflashMaxCtx
        case dflashDraftQuantEnabled, dflashDraftQuantWeightBits
        case dflashDraftQuantActivationBits, dflashDraftQuantGroupSize
        case dflashVerifyMode, dflashDraftWindowSize, dflashDraftSinkSize
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
            ("embed", String(localized: "settings.model_type.embed",
                             defaultValue: "Embedding",
                             comment: "Model type option label for embedding models")),
            ("rerank", String(localized: "settings.model_type.rerank",
                              defaultValue: "Reranker",
                              comment: "Model type option label for reranker models")),
            ("audio-stt", String(localized: "settings.model_type.audio_stt",
                                 defaultValue: "Audio STT",
                                 comment: "Model type option label for speech-to-text models")),
            ("audio-tts", String(localized: "settings.model_type.audio_tts",
                                 defaultValue: "Audio TTS",
                                 comment: "Model type option label for text-to-speech models")),
            ("audio-sts", String(localized: "settings.model_type.audio_sts",
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
    var dflashDraftSinkSize: String = ""
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

    private func isDiffusionUnsupportedField(_ field: Field) -> Bool {
        switch field {
        case .topP, .topK, .minP, .repetitionPenalty, .presencePenalty:
            return true
        case .enableThinking, .thinkingBudgetEnabled, .thinkingBudgetTokens:
            return true
        case .limitToolResults, .toolResultLimitTokens:
            return true
        case .forceSampling, .reasoningParser:
            return true
        case .turboquantKvEnabled, .turboquantKvBits:
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
        case .dflashVerifyMode, .dflashDraftWindowSize, .dflashDraftSinkSize:
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
                self.dflashDraftSinkSize = s?.dflashDraftSinkSize.map(String.init) ?? ""
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

    /// MTP can't co-exist with DFlash or TurboQuant KV. The toggle uses
    /// this to disable itself and surface the conflict reason.
    var mtpConflictReason: String? {
        if dflashEnabled {
            return String(localized: "settings.mtp.conflict.dflash",
                          defaultValue: "Disable DFlash before enabling MTP.",
                          comment: "Tooltip / sublabel shown when MTP can't be enabled because DFlash is on")
        }
        if turboquantKvEnabled {
            return String(localized: "settings.mtp.conflict.turboquant",
                          defaultValue: "Disable TurboQuant KV before enabling MTP.",
                          comment: "Tooltip / sublabel shown when MTP can't be enabled because TurboQuant KV is on")
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
        return nil
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
            putBool(ProfileSettingsKey.turboquantKvEnabled, turboquantKvEnabled)
            if turboquantKvEnabled, let bits = Double(turboquantKvBits) {
                out[ProfileSettingsKey.turboquantKvBits] = AnyCodable(bits)
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

    /// SpecPrefill keep-pct dropdown is declared with string options like
    /// "0.2"; `String(0.2)` happens to print as `"0.2"` on Darwin but
    /// `"0.20"` would not match. Format defensively so the dropdown shows
    /// the saved value highlighted.
    fileprivate static func formatPct(_ v: Double) -> String {
        // Always 1-2 decimals to match the option values.
        let rounded = (v * 100).rounded() / 100
        if rounded == rounded.rounded() { return String(format: "%.1f", rounded) }
        return String(format: "%.2f", rounded)
    }
}

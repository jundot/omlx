// PR 8 — per-model settings drilled into from ModelsScreen via the chevron.
//
// Sections (segmented at the top):
//   • Profiles  — list per-model profiles + create / delete / apply,
//                  list templates (read-only) + apply a template as a profile
//   • Basic     — alias, model type, context window, max tokens, sampling
//                  defaults (temperature, top_p, top_k, min_p,
//                  repetition_penalty, presence_penalty), TTL
//   • Advanced  — enable_thinking, thinking budget, limit tool result tokens,
//                  force sampling, pin in memory
//
// Aliases (the design's 4th tab) is omitted: server has no /api/aliases
// endpoint and `model_alias` is singular. Keeping the surface honest.
//
// Saves on every committed edit (Popup change / TextField submit / Toggle
// flip), no explicit Save button — same UX as ServerScreen. The design's
// Save / Cancel / Load Defaults buttons live as a top-right toolbar that
// only does navigation back to Models.

import SwiftUI

struct ModelSettingsScreen: View {
    let modelID: String

    @Environment(AppServices.self) private var services
    @State private var vm = ModelSettingsScreenVM()

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Header(model: vm.model)

            SectionPicker(selection: $vm.section)

            switch vm.section {
            case .profiles:
                ProfilesTab(
                    vm: vm,
                    presetStore: services.presetBundle,
                    client: services.client,
                    serverDefaults: vm.serverDefaultSampling,
                    // Deep-link to the Server tab's Default Profile
                    // section. Setting the anchor *before* the section
                    // means ContentScaffold's `.task(id:)` sees both
                    // pieces in one go and scrolls without a noop pass.
                    onEditServer: {
                        services.requestedServerAnchor = .defaultProfile
                        services.requestedSection = .server
                    }
                )
            case .basic:
                BasicTab(vm: vm, client: services.client)
            case .advanced:
                AdvancedTab(vm: vm, client: services.client)
            }

            if let error = vm.lastError {
                Text(error)
                    .font(.omlxText(11))
                    .foregroundStyle(.red)
                    .padding(.horizontal, 18)
                    .padding(.top, 8)
            }
        }
        .toolbar {
            ToolbarItem(placement: .navigation) {
                backButton
            }
        }
        .task(id: modelID) { await vm.load(modelID: modelID, client: services.client) }
    }

    @ViewBuilder
    private var backButton: some View {
        Button {
            services.modelDetailID = nil
        } label: {
            Label(String(localized: "settings.header.back_to_models",
                         defaultValue: "Back to Models",
                         comment: "Back button label at the top of the per-model settings screen"),
                  systemImage: "chevron.left")
                .labelStyle(.iconOnly)
        }
    }
}

// MARK: - Header

private struct Header: View {
    let model: ModelDTO?
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        HStack(spacing: 12) {
            Squircle(systemSymbol: "cpu", size: 44, gradient: SquircleGradient.models)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 4) {
                    Text(model?.displayTitle ?? "—")
                        .font(.omlxText(17, weight: .semibold))
                        .foregroundStyle(theme.text)
                        .lineLimit(1)
                        .truncationMode(.tail)
                    if let id = model?.id {
                        CopyIconButton(value: id)
                    }
                }
                if let m = model {
                    Text("\(m.id) · \(m.estimatedSizeFormatted ?? formatBytes(m.estimatedSize))")
                        .font(.omlxMono(11))
                        .foregroundStyle(theme.textSecondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }
            .layoutPriority(1)
            Spacer()
        }
        .padding(.horizontal, 14)
        .padding(.bottom, 10)
    }
}

// MARK: - Section picker

private struct SectionPicker: View {
    @Binding var selection: ModelSettingsScreenVM.Section

    var body: some View {
        HStack {
            Segmented(
                selection: $selection,
                options: ModelSettingsScreenVM.Section.allCases.map {
                    ($0, $0.label)
                }
            )
            Spacer()
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 6)
    }
}

// MARK: - Profiles tab

private struct ProfilesTab: View {
    var vm: ModelSettingsScreenVM
    /// Source of `.preset` chips — the shipped JSON bundle, refreshable
    /// from omlx.ai via `POST /api/presets/refresh`. Replaces the legacy
    /// `vm.templates.filter { isBuiltin }` source after Phase 1 retired
    /// the server-side builtin templates.
    let presetStore: PresetBundleStore
    let client: OMLXClient
    /// Optional binding to a Server-Defaults DTO surfaced read-only at
    /// the bottom of the tab. Lives on the parent (a `@State`-
    /// owned VM) so Phase 3's Server screen and this tab share state.
    var serverDefaults: GlobalSettingsDTO.SamplingDTO?
    /// Action handler for "Edit on Server →" link in the Server
    /// Defaults section. Lifted by the parent so we don't introduce a
    /// hard dep on AppServices from inside this view.
    var onEditServer: () -> Void

    /// Currently previewed chip (overrides the active-state detail card).
    @State private var preview: ActiveProfileState.NamedProfileRef? = nil
    /// Save-as popover state. Non-nil → popover visible. Pre-set + switchable
    /// scope per chat2.md decisions.
    @State private var saveAsName: String = ""
    @State private var saveAsScope: ProfileScope = .global
    @State private var saveAsOpen: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Active state banner — three variants (working / named / defaults).
            ActiveProfileBanner(
                state: vm.activeProfileState,
                isSlim: false,
                onUpdateBasedOn: {
                    if case .working(let basedOn) = vm.activeProfileState, let basedOn {
                        Task {
                            await vm.updateProfileWithWorking(
                                scope: basedOn.scope, name: basedOn.name, client: client
                            )
                        }
                    }
                },
                onSaveAsNew: { openSaveAs(scope: .global) },
                onRevert: {
                    Task { await vm.revertWorking(client: client) }
                }
            )

            if saveAsOpen {
                SaveAsPopover(
                    name: $saveAsName,
                    scope: $saveAsScope,
                    onCommit: {
                        Task {
                            await vm.saveWorkingAs(
                                scope: saveAsScope, name: saveAsName, client: client
                            )
                            saveAsOpen = false
                        }
                    },
                    onCancel: { saveAsOpen = false }
                )
            }

            ProfileGroup(
                scope: .preset,
                label: String(localized: "settings.profiles.preset.label",
                              defaultValue: "Preset Profiles",
                              comment: "Section label above the bundled preset profiles chip group"),
                names: presetStore.entries.map(\.name),
                activeName: vm.activeProfileState.activeName(in: .preset),
                basedOnName: vm.activeProfileState.basedOnName(in: .preset),
                previewName: preview?.scope == .preset ? preview?.name : nil,
                canSaveCurrent: false,
                onSelect: { previewChip(scope: .preset, name: $0) },
                onSaveCurrent: { },
                onRefresh: {
                    Task { await presetStore.refresh(client: client) }
                },
                isRefreshing: presetStore.isRefreshing
            )

            ProfileGroup(
                scope: .global,
                label: String(localized: "settings.profiles.global.label",
                              defaultValue: "Global Profiles",
                              comment: "Section label above the user-defined global profile templates chip group"),
                names: vm.templates.filter { $0.templateScope == .global }.map(\.name),
                activeName: vm.activeProfileState.activeName(in: .global),
                basedOnName: vm.activeProfileState.basedOnName(in: .global),
                previewName: preview?.scope == .global ? preview?.name : nil,
                canSaveCurrent: vm.profileDirty,
                onSelect: { previewChip(scope: .global, name: $0) },
                onSaveCurrent: { openSaveAs(scope: .global) },
                onRename: { original, renamed in
                    Task { await vm.renameTemplate(from: original, to: renamed, client: client) }
                }
            )

            ProfileGroup(
                scope: .model,
                label: String(localized: "settings.profiles.model.label",
                              defaultValue: "Model Profiles · \(vm.model?.id ?? vm.modelID)",
                              comment: "Section label for the per-model profile chip group; placeholder is the model id"),
                names: vm.profiles
                    .filter { $0.sourceTemplate == nil }
                    .map(\.name),
                activeName: vm.activeProfileState.activeName(in: .model),
                basedOnName: vm.activeProfileState.basedOnName(in: .model),
                previewName: preview?.scope == .model ? preview?.name : nil,
                canSaveCurrent: vm.profileDirty,
                onSelect: { previewChip(scope: .model, name: $0) },
                onSaveCurrent: { openSaveAs(scope: .model) },
                onRename: { original, renamed in
                    Task { await vm.renameModelProfile(from: original, to: renamed, client: client) }
                }
            )

            detailCard

            SectionHeader(
                String(localized: "settings.profiles.server_defaults.title",
                       defaultValue: "Server Defaults",
                       comment: "Section header above the read-only Server Defaults card"),
                subtitle: String(localized: "settings.profiles.server_defaults.subtitle",
                                 defaultValue: "Used when no profile is set, or when a profile leaves a field empty",
                                 comment: "Subtitle explaining the role of the Server Defaults profile")
            ) {
                Button(String(localized: "settings.profiles.edit_on_server",
                              defaultValue: "Edit on Server →",
                              comment: "Plain link button that deep-links to the Server screen's Default Profile section")) {
                    onEditServer()
                }
                    .buttonStyle(.omlx(.plain, size: .small))
            }
            ProfileDetailCard(
                name: String(localized: "settings.profiles.server_default.name",
                             defaultValue: "Server Default Profile",
                             comment: "Display name of the synthesized 'Server Default' profile card"),
                scope: nil,
                settings: serverDefaultsAsDict(serverDefaults),
                isActive: false,
                isWorking: false,
                basedOn: nil,
                isWorkingBase: false,
                compact: true,
                hasWorking: false
            )
        }
    }

    @ViewBuilder
    private var detailCard: some View {
        if let preview, let tpl = lookupSettings(scope: preview.scope, name: preview.name) {
            ProfileDetailCard(
                name: preview.name,
                scope: preview.scope,
                settings: tpl,
                isActive: vm.activeProfileState.activeName(in: preview.scope) == preview.name,
                isWorking: false,
                basedOn: nil,
                isWorkingBase: vm.activeProfileState.basedOnName(in: preview.scope) == preview.name,
                compact: false,
                hasWorking: vm.profileDirty,
                onApply: {
                    Task {
                        if preview.scope == .preset,
                           let entry = presetStore.entries
                                .first(where: { $0.name == preview.name }) {
                            await vm.applyPreset(entry, client: client)
                        } else {
                            await vm.applyChip(
                                scope: preview.scope, name: preview.name, client: client
                            )
                        }
                        self.preview = nil
                    }
                },
                onUpdateFromWorking: vm.profileDirty && preview.scope != .preset
                    ? {
                        Task {
                            await vm.updateProfileWithWorking(
                                scope: preview.scope, name: preview.name, client: client
                            )
                            self.preview = nil
                        }
                    }
                    : nil,
                onDelete: preview.scope == .preset ? nil : {
                    Task {
                        await deleteChip(scope: preview.scope, name: preview.name)
                        self.preview = nil
                    }
                },
                onClosePreview: { self.preview = nil },
                exposeAsModel: modelProfile(named: preview.name)?.exposeAsModel ?? false,
                exposedModelId: modelProfile(named: preview.name)?.modelId,
                hasEngineFields: modelProfile(named: preview.name)?.hasEngineFields ?? false,
                onToggleExpose: preview.scope == .model
                    ? { exposed in
                        Task {
                            await vm.setExposeAsModel(
                                name: preview.name, exposed: exposed, client: client
                            )
                        }
                    }
                    : nil
            )
        } else {
            // No preview → show the active state's detail.
            switch vm.activeProfileState {
            case .working(let basedOn):
                ProfileDetailCard(
                    name: String(localized: "settings.profiles.working.name",
                                 defaultValue: "Working profile",
                                 comment: "Display name for the in-progress (unsaved) working profile detail card"),
                    scope: basedOn?.scope,
                    settings: vm.currentSettingsDict(),
                    isActive: true,
                    isWorking: true,
                    basedOn: basedOn,
                    isWorkingBase: false,
                    compact: false,
                    hasWorking: true
                )
            case .named(let scope, let name):
                let settings = lookupSettings(scope: scope, name: name) ?? [:]
                ProfileDetailCard(
                    name: name,
                    scope: scope,
                    settings: settings,
                    isActive: true,
                    isWorking: false,
                    basedOn: nil,
                    isWorkingBase: false,
                    compact: false,
                    hasWorking: false,
                    exposeAsModel: modelProfile(named: name)?.exposeAsModel ?? false,
                    exposedModelId: modelProfile(named: name)?.modelId,
                    hasEngineFields: modelProfile(named: name)?.hasEngineFields ?? false,
                    onToggleExpose: scope == .model
                        ? { exposed in
                            Task {
                                await vm.setExposeAsModel(
                                    name: name, exposed: exposed, client: client
                                )
                            }
                        }
                        : nil
                )
            case .defaults:
                ProfileDetailCard(
                    name: String(localized: "settings.profiles.no_profile.name",
                                 defaultValue: "No profile",
                                 comment: "Display name shown in the profile detail card when no profile is active"),
                    scope: nil,
                    settings: serverDefaultsAsDict(serverDefaults),
                    isActive: true,
                    isWorking: false,
                    basedOn: nil,
                    isWorkingBase: false,
                    compact: false,
                    hasWorking: false
                )
            }
        }
    }

    /// Per-model profile DTO lookup — source of the expose-as-model state
    /// and the derived model ID shown on the detail card.
    private func modelProfile(named name: String) -> ProfileDTO? {
        vm.profiles.first { $0.name == name }
    }

    private func previewChip(scope: ProfileScope, name: String) {
        // Toggle off when re-clicking the same chip.
        if preview?.scope == scope && preview?.name == name {
            preview = nil
        } else {
            preview = .init(scope: scope, name: name)
        }
    }

    private func openSaveAs(scope: ProfileScope) {
        saveAsScope = scope
        saveAsName = vm.suggestSaveAsName()
        saveAsOpen = true
    }

    private func deleteChip(scope: ProfileScope, name: String) async {
        do {
            switch scope {
            case .global:
                _ = try await client.deleteProfileTemplate(name: name)
            case .model:
                _ = try await client.deleteModelProfile(id: vm.modelID, name: name)
            case .preset:
                return
            }
            await vm.load(modelID: vm.modelID, client: client)
        } catch {
            // Surfaces via the screen's lastError banner — set on the VM.
            await MainActor.run { vm.lastError = error.omlxDescription }
        }
    }

    private func lookupSettings(scope: ProfileScope, name: String) -> [String: AnyCodable]? {
        switch scope {
        case .preset:
            return presetStore.entries.first(where: { $0.name == name })?.settings
        case .global:
            return vm.templates.first(where: { $0.name == name })?.settings
        case .model:
            return vm.profiles.first(where: { $0.name == name })?.settings
        }
    }

}

/// Translate the server's typed SamplingDTO into the loose dict the
/// ProfileDetailCard renders against. Keys match `ProfileSettingsKey`.
private func serverDefaultsAsDict(_ s: GlobalSettingsDTO.SamplingDTO?) -> [String: AnyCodable] {
    guard let s else { return [:] }
    return [
        ProfileSettingsKey.maxContextWindow:  AnyCodable(s.maxContextWindow),
        ProfileSettingsKey.maxTokens:         AnyCodable(s.maxTokens),
        ProfileSettingsKey.temperature:       AnyCodable(s.temperature),
        ProfileSettingsKey.topP:              AnyCodable(s.topP),
        ProfileSettingsKey.topK:              AnyCodable(s.topK),
        ProfileSettingsKey.repetitionPenalty: AnyCodable(s.repetitionPenalty),
    ]
}

private extension ActiveProfileState {
    /// Name of the active profile if it lives in the given scope, else nil.
    func activeName(in scope: ProfileScope) -> String? {
        if case .named(let s, let n) = self, s == scope { return n }
        return nil
    }

    /// Name of the "based on" reference if it lives in the given scope.
    func basedOnName(in scope: ProfileScope) -> String? {
        if case .working(let basedOn) = self, let basedOn, basedOn.scope == scope {
            return basedOn.name
        }
        return nil
    }
}

// ProfileChips / ChipView / FlowHStack / FlowLayout were the v1 layout
// of the Profiles tab. Replaced by ProfileGroup + ProfileViews.FlowLayout
// when the working-profile redesign landed.

// MARK: - Basic tab

private struct BasicTab: View {
    @Bindable var vm: ModelSettingsScreenVM
    let client: OMLXClient

    var body: some View {
        BasicEditBanner(vm: vm, client: client)
        SectionHeader(String(localized: "settings.basic.section",
                             defaultValue: "Basic Settings",
                             comment: "Section header above the Basic tab fields"))

        // Per-model fields (alias / modelType / TTL) auto-save on commit.
        // Profile-eligible fields (sampling, penalties) write to the
        // working profile instead — surfaced via the banner above.
        ListGroup {
            Row(label: String(localized: "settings.basic.alias.label",
                              defaultValue: "Model Alias",
                              comment: "Row label for the model alias field"),
                sublabel: String(localized: "settings.basic.alias.sub",
                                 defaultValue: "Falls back to the model id",
                                 comment: "Sublabel for the model alias field")) {
                TextInput(text: $vm.alias, placeholder: vm.modelID, mono: true, width: 220)
                    .onSubmit { Task { await vm.save(.alias, client: client) } }
            }
            Row(label: String(localized: "settings.basic.model_type.label",
                              defaultValue: "Model Type",
                              comment: "Row label for the model type override popup")) {
                Popup(
                    selection: vm.bind($vm.modelTypeOverride, save: { Task { await vm.save(.modelType, client: client) } }),
                    width: 170,
                    options: ModelSettingsScreenVM.modelTypeOptions
                )
            }
            Row(label: String(localized: "settings.basic.context_window.label",
                              defaultValue: "Context Window",
                              comment: "Row label for the context window field"),
                sublabel: String(localized: "settings.basic.context_window.sub",
                                 defaultValue: "Maximum tokens per request",
                                 comment: "Sublabel for the context window field")) {
                TextInput(text: vm.bindProfile($vm.contextLength), mono: true, suffix: "tk", width: 110)
            }
            Row(label: String(localized: "settings.basic.max_tokens.label",
                              defaultValue: "Max Tokens",
                              comment: "Row label for the max generated tokens field"),
                sublabel: String(localized: "settings.basic.max_tokens.sub",
                                 defaultValue: "Cap on generated tokens (empty = default)",
                                 comment: "Sublabel for the max generated tokens field")) {
                TextInput(text: vm.bindProfile($vm.maxTokens),
                          placeholder: String(localized: "settings.basic.max_tokens.placeholder",
                                              defaultValue: "Default",
                                              comment: "Placeholder shown when Max Tokens is empty (server default applies)"),
                          mono: true, width: 110)
            }
            Row(label: String(localized: "settings.basic.temperature.label",
                              defaultValue: "Temperature",
                              comment: "Row label for the sampling temperature field"),
                sublabel: String(localized: "settings.basic.temperature.sub",
                                 defaultValue: "Sampling randomness (≥ 0). 0 = deterministic.",
                                 comment: "Sublabel describing the temperature field range")) {
                TextInput(text: vm.bindProfile($vm.temperature), placeholder: "0.7", mono: true, width: 90)
            }
            if !vm.isDiffusionModel {
                Row(label: String(localized: "settings.basic.top_p.label",
                                  defaultValue: "Top P",
                                  comment: "Row label for the top-p nucleus sampling field"),
                    sublabel: String(localized: "settings.basic.top_p.sub",
                                     defaultValue: "Nucleus sampling cutoff (0 < p ≤ 1).",
                                     comment: "Sublabel describing the top-p valid range")) {
                    TextInput(text: vm.bindProfile($vm.topP), mono: true, width: 90)
                }
                Row(label: String(localized: "settings.basic.top_k.label",
                                  defaultValue: "Top K",
                                  comment: "Row label for the top-k sampling field"),
                    sublabel: String(localized: "settings.basic.top_k.sub",
                                     defaultValue: "Limit candidates to top K (positive integer).",
                                     comment: "Sublabel describing the top-k field")) {
                    TextInput(text: vm.bindProfile($vm.topK), mono: true, width: 90)
                }
                Row(label: String(localized: "settings.basic.min_p.label",
                                  defaultValue: "Min P",
                                  comment: "Row label for the min-p sampling field"),
                    sublabel: String(localized: "settings.basic.min_p.sub",
                                     defaultValue: "Minimum probability floor (0 ≤ p ≤ 1).",
                                     comment: "Sublabel describing the min-p field range")) {
                    TextInput(text: vm.bindProfile($vm.minP), mono: true, width: 90)
                }
                Row(label: String(localized: "settings.basic.repetition_penalty.label",
                                  defaultValue: "Repetition Penalty",
                                  comment: "Row label for the repetition-penalty field"),
                    sublabel: String(localized: "settings.basic.repetition_penalty.sub",
                                     defaultValue: "Penalize repeated tokens (−2 to 2).",
                                     comment: "Sublabel describing repetition-penalty range")) {
                    TextInput(text: vm.bindProfile($vm.repetitionPenalty), mono: true, width: 90)
                }
                Row(label: String(localized: "settings.basic.presence_penalty.label",
                                  defaultValue: "Presence Penalty",
                                  comment: "Row label for the presence-penalty field"),
                    sublabel: String(localized: "settings.basic.presence_penalty.sub",
                                     defaultValue: "Penalize tokens already present (−2 to 2).",
                                     comment: "Sublabel describing presence-penalty range")) {
                    TextInput(text: vm.bindProfile($vm.presencePenalty), mono: true, width: 90)
                }
            }
            Row(
                label: String(localized: "settings.basic.ttl.label",
                              defaultValue: "TTL",
                              comment: "Row label for the idle-unload TTL field"),
                sublabel: String(localized: "settings.basic.ttl.sub",
                                 defaultValue: "Seconds before idle unload (empty = no TTL)",
                                 comment: "Sublabel for the idle-unload TTL field"),
                isLast: true
            ) {
                TextInput(text: $vm.ttlSeconds,
                          placeholder: String(localized: "settings.basic.ttl.placeholder",
                                              defaultValue: "No TTL",
                                              comment: "Placeholder shown when no TTL is configured"),
                          mono: true, suffix: "s", width: 110)
                    .onSubmit { Task { await vm.save(.ttl, client: client) } }
            }
        }
    }
}

/// Slim ActiveProfileBanner used above Basic / Advanced editors so the user
/// can save without bouncing back to the Profiles tab. Renders nothing in
/// the `named` (clean) state — no banner clutter when there's nothing to
/// do.
private struct BasicEditBanner: View {
    var vm: ModelSettingsScreenVM
    let client: OMLXClient

    @State private var saveAsScope: ProfileScope = .global
    @State private var saveAsName: String = ""
    @State private var saveAsOpen: Bool = false

    var body: some View {
        switch vm.activeProfileState {
        case .named:
            EmptyView()
        default:
            VStack(alignment: .leading, spacing: 0) {
                ActiveProfileBanner(
                    state: vm.activeProfileState,
                    isSlim: true,
                    onUpdateBasedOn: {
                        if case .working(let basedOn) = vm.activeProfileState, let basedOn {
                            Task {
                                await vm.updateProfileWithWorking(
                                    scope: basedOn.scope, name: basedOn.name, client: client
                                )
                            }
                        }
                    },
                    onSaveAsNew: {
                        saveAsScope = .global
                        saveAsName = vm.suggestSaveAsName()
                        saveAsOpen = true
                    },
                    onRevert: {
                        Task { await vm.revertWorking(client: client) }
                    }
                )
                if saveAsOpen {
                    SaveAsPopover(
                        name: $saveAsName,
                        scope: $saveAsScope,
                        onCommit: {
                            Task {
                                await vm.saveWorkingAs(
                                    scope: saveAsScope, name: saveAsName, client: client
                                )
                                saveAsOpen = false
                            }
                        },
                        onCancel: { saveAsOpen = false }
                    )
                }
            }
        }
    }
}

// MARK: - Advanced tab

private struct AdvancedTab: View {
    @Bindable var vm: ModelSettingsScreenVM
    let client: OMLXClient

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        BasicEditBanner(vm: vm, client: client)
        SectionHeader(String(localized: "settings.advanced.section",
                             defaultValue: "Advanced Settings",
                             comment: "Section header above the Advanced tab fields"))

        // Profile-eligible toggles use `bindProfile` — flipping them flips
        // the working-dirty flag. `isPinned` and `trustRemoteCode` stay
        // per-model (server excludes them from profiles) and auto-save.
        ListGroup {
            if !vm.isDiffusionModel {
                Row(label: String(localized: "settings.advanced.enable_thinking.label",
                                  defaultValue: "Enable Thinking",
                                  comment: "Row label for the enable-thinking toggle"),
                    sublabel: String(localized: "settings.advanced.enable_thinking.sub",
                                     defaultValue: "Enable reasoning/thinking mode for this model",
                                     comment: "Sublabel for the enable-thinking toggle")) {
                    Toggle("", isOn: vm.bindProfile($vm.enableThinking))
                        .labelsHidden().toggleStyle(.switch)
                }
                Row(label: String(localized: "settings.advanced.thinking_budget.label",
                                  defaultValue: "Thinking Budget",
                                  comment: "Row label for the thinking budget field"),
                    sublabel: String(localized: "settings.advanced.thinking_budget.sub",
                                     defaultValue: "Limit thinking tokens for reasoning models. Forces end of thinking when exceeded.",
                                     comment: "Sublabel for the thinking budget field")) {
                    HStack(spacing: 8) {
                        if vm.thinkingBudgetEnabled {
                            TextInput(text: vm.bindProfile($vm.thinkingBudgetTokens),
                                      mono: true, suffix: "tk", width: 110)
                        }
                        Toggle("", isOn: vm.bindProfile($vm.thinkingBudgetEnabled))
                            .labelsHidden().toggleStyle(.switch)
                    }
                }
                Row(label: String(localized: "settings.advanced.tool_result_limit.label",
                                  defaultValue: "Limit Tool Result Tokens",
                                  comment: "Row label for the tool-result token limit field"),
                    sublabel: String(localized: "settings.advanced.tool_result_limit.sub",
                                     defaultValue: "Truncate large tool results (e.g. file reads) to a token limit",
                                     comment: "Sublabel for the tool-result token limit field")) {
                    HStack(spacing: 8) {
                        if vm.limitToolResults {
                            TextInput(text: vm.bindProfile($vm.toolResultLimitTokens),
                                      placeholder: "4096",
                                      mono: true, suffix: "tk", width: 110)
                        }
                        Toggle("", isOn: vm.bindProfile($vm.limitToolResults))
                            .labelsHidden().toggleStyle(.switch)
                    }
                }
                Row(label: String(localized: "settings.advanced.force_sampling.label",
                                  defaultValue: "Force Sampling",
                                  comment: "Row label for the force-sampling toggle"),
                    sublabel: String(localized: "settings.advanced.force_sampling.sub",
                                     defaultValue: "Override request sampling parameters with configured values",
                                     comment: "Sublabel for the force-sampling toggle")) {
                    Toggle("", isOn: vm.bindProfile($vm.forceSampling))
                        .labelsHidden().toggleStyle(.switch)
                }
                Row(label: String(localized: "settings.advanced.reasoning_parser.label",
                                  defaultValue: "Reasoning Parser",
                                  comment: "Row label for the reasoning-parser override field"),
                    sublabel: String(localized: "settings.advanced.reasoning_parser.sub",
                                     defaultValue: "Override the chain-of-thought parser. Leave empty to use the model's default.",
                                     comment: "Sublabel for the reasoning-parser override field")) {
                    TextInput(text: vm.bindProfile($vm.reasoningParser),
                              placeholder: "auto", mono: true, width: 150)
                }
            }
            Row(label: String(localized: "settings.advanced.pin_memory.label",
                              defaultValue: "Pin in memory",
                              comment: "Row label for the pin-in-memory toggle"),
                sublabel: String(localized: "settings.advanced.pin_memory.sub",
                                 defaultValue: "Keep this model resident between requests",
                                 comment: "Sublabel for the pin-in-memory toggle")) {
                Toggle("", isOn: vm.bind($vm.isPinned, save: {
                    Task { await vm.save(.isPinned, client: client) }
                }))
                .labelsHidden().toggleStyle(.switch)
            }
            // Security-sensitive row — flagged red to match the HTML
            // editor's visual treatment. HF custom-code execution gives
            // the model author the ability to run arbitrary Python in
            // the server process; never propagated via profiles.
            Row(label: String(localized: "settings.advanced.trust_remote_code.label",
                              defaultValue: "Trust Remote Code",
                              comment: "Row label for the security-sensitive trust-remote-code toggle"),
                sublabel: String(localized: "settings.advanced.trust_remote_code.sub",
                                 defaultValue: "Execute HuggingFace custom model code. Only enable for models you trust. Per-model only — never inherited from profiles.",
                                 comment: "Sublabel describing the security implications of trust-remote-code"),
                isLast: true) {
                Toggle("", isOn: vm.bind($vm.trustRemoteCode, save: {
                    Task { await vm.save(.trustRemoteCode, client: client) }
                }))
                .labelsHidden().toggleStyle(.switch)
                .tint(theme.redDot)
            }
        }

        SectionHeader(
            String(localized: "settings.advanced.chat_template.section",
                   defaultValue: "Chat Template Kwargs",
                   comment: "Section header above the chat-template kwargs editor"),
            subtitle: String(localized: "settings.advanced.chat_template.subtitle",
                             defaultValue: "Forwarded to the model's chat template. Toggle Force to override per-request values.",
                             comment: "Subtitle for the chat-template kwargs section")
        )
        ChatTemplateKwargsEditor(vm: vm, client: client)

        if !vm.isDiffusionModel {
            SectionHeader(
                String(localized: "settings.advanced.experimental.section",
                       defaultValue: "Experimental",
                       comment: "Section header above the Experimental settings group"),
                subtitle: String(localized: "settings.advanced.experimental.subtitle",
                                 defaultValue: "Speculative decoding, KV-cache quantization, and other research features.",
                                 comment: "Subtitle for the Experimental settings section")
            )
            ExperimentalSection(vm: vm, client: client)
        }
    }
}

// MARK: - Chat-template kwargs editor

private struct ChatTemplateKwargsEditor: View {
    var vm: ModelSettingsScreenVM
    let client: OMLXClient

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        ListGroup {
            FreeRow {
                HStack {
                    Text(vm.chatTemplateEntries.isEmpty
                         ? String(localized: "settings.advanced.chat_template.empty",
                                  defaultValue: "No chat-template kwargs.",
                                  comment: "Placeholder text shown when no chat-template kwargs are configured")
                         : String(localized: "settings.advanced.chat_template.count",
                                  defaultValue: "kwargs: \(vm.chatTemplateEntries.count)",
                                  comment: "Count summary in the chat-template editor; placeholder is the entry count"))
                        .font(.omlxText(12))
                        .foregroundStyle(theme.textSecondary)
                    Spacer()
                    addMenu
                }
            }
            ForEach(vm.chatTemplateEntries) { entry in
                let isLast = entry.id == vm.chatTemplateEntries.last?.id
                FreeRow(isLast: isLast) {
                    EntryEditor(
                        vm: vm,
                        client: client,
                        entryID: entry.id
                    )
                }
            }
        }
    }

    @ViewBuilder
    private var addMenu: some View {
        Menu {
            // `enable_thinking` and `reasoning_effort` are server-side
            // singletons — once added, the menu hides them so the user
            // can't push duplicate keys into `chat_template_kwargs`.
            if !vm.isDiffusionModel,
               !vm.chatTemplateEntries.contains(where: { $0.kind == .enableThinking }) {
                Button("enable_thinking") {
                    vm.addKwarg(.enableThinking)
                }
            }
            if !vm.isDiffusionModel,
               !vm.chatTemplateEntries.contains(where: { $0.kind == .reasoningEffort }) {
                Button("reasoning_effort") {
                    vm.addKwarg(.reasoningEffort)
                }
            }
            Button(String(localized: "settings.advanced.chat_template.add_custom",
                          defaultValue: "custom…",
                          comment: "Menu item for adding a custom (free-form key/value) chat-template kwarg")) {
                vm.addKwarg(.custom)
            }
        } label: {
            Label(String(localized: "settings.advanced.chat_template.add_kwarg",
                         defaultValue: "Add kwarg",
                         comment: "Plus-button label for adding a chat-template kwarg row"),
                  systemImage: "plus")
                .labelStyle(.titleAndIcon)
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
    }
}

private struct EntryEditor: View {
    var vm: ModelSettingsScreenVM
    let client: OMLXClient
    let entryID: UUID

    @Environment(\.omlxTheme) private var theme

    private var entry: ChatTemplateKwargEntry {
        vm.chatTemplateEntries.first { $0.id == entryID } ?? ChatTemplateKwargEntry(kind: .custom, value: "")
    }

    private var binding: Binding<ChatTemplateKwargEntry> {
        Binding(
            get: { vm.chatTemplateEntries.first { $0.id == entryID } ?? ChatTemplateKwargEntry(kind: .custom, value: "") },
            set: { newValue in
                if let idx = vm.chatTemplateEntries.firstIndex(where: { $0.id == entryID }) {
                    vm.chatTemplateEntries[idx] = newValue
                }
            }
        )
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text(typeLabel)
                    .font(.omlxText(11, weight: .semibold))
                    .foregroundStyle(theme.textSecondary)
                Spacer()
                Button {
                    vm.removeKwarg(id: entryID)
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(theme.textSecondary)
                        .frame(width: 22, height: 22)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help(String(localized: "settings.advanced.chat_template.remove",
                             defaultValue: "Remove kwarg",
                             comment: "Tooltip on the trash/xmark button that deletes a chat-template kwarg row"))
            }
            valueRow
        }
    }

    private var typeLabel: String {
        // These are eyebrow labels rendered uppercase above each editor.
        // Keeping the localization keys aligned with display text rather
        // than the server kwarg key.
        switch entry.kind {
        case .enableThinking:
            return String(localized: "settings.advanced.chat_template.type.enable_thinking",
                          defaultValue: "ENABLE_THINKING",
                          comment: "Eyebrow label above the enable_thinking kwarg editor")
        case .reasoningEffort:
            return String(localized: "settings.advanced.chat_template.type.reasoning_effort",
                          defaultValue: "REASONING_EFFORT",
                          comment: "Eyebrow label above the reasoning_effort kwarg editor")
        case .custom:
            return String(localized: "settings.advanced.chat_template.type.custom",
                          defaultValue: "CUSTOM",
                          comment: "Eyebrow label above a custom (free-form) chat-template kwarg editor")
        }
    }

    @ViewBuilder
    private var valueRow: some View {
        switch entry.kind {
        case .enableThinking:
            HStack(spacing: 8) {
                Popup(
                    selection: vm.bindProfile(binding.value),
                    width: 130,
                    options: [("true", "true"), ("false", "false")]
                )
                forceCheckbox
            }
        case .reasoningEffort:
            HStack(spacing: 8) {
                Popup(
                    selection: vm.bindProfile(binding.value),
                    width: 130,
                    options: [("low", "low"), ("medium", "medium"), ("high", "high")]
                )
                forceCheckbox
            }
        case .custom:
            VStack(alignment: .leading, spacing: 6) {
                TextInput(text: vm.bindProfile(binding.customKey),
                          placeholder: String(localized: "settings.advanced.chat_template.key_placeholder",
                                              defaultValue: "key",
                                              comment: "Placeholder for the custom kwarg key field"),
                          mono: true)
                HStack(spacing: 8) {
                    TextInput(text: vm.bindProfile(binding.value),
                              placeholder: String(localized: "settings.advanced.chat_template.value_placeholder",
                                                  defaultValue: "value",
                                                  comment: "Placeholder for the custom kwarg value field"),
                              mono: true)
                    forceCheckbox
                }
            }
        }
    }

    private var forceCheckbox: some View {
        Toggle(isOn: vm.bindProfile(binding.force)) {
            Text(String(localized: "settings.advanced.chat_template.force",
                        defaultValue: "Force",
                        comment: "Checkbox label for forcing a chat-template kwarg via forced_ct_kwargs"))
                .font(.omlxText(11))
                .foregroundStyle(theme.textSecondary)
        }
        .toggleStyle(.checkbox)
        .help(String(localized: "settings.advanced.chat_template.force.help",
                     defaultValue: "Add this key to forced_ct_kwargs so the request body can't override it.",
                     comment: "Tooltip explaining the Force checkbox"))
    }
}

// MARK: - Experimental section

private struct ExperimentalSection: View {
    @Bindable var vm: ModelSettingsScreenVM
    let client: OMLXClient

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        // All experimental fields are profile-eligible (universal or
        // model-specific). Edits write to the working profile via
        // bindProfile and surface in the Active banner above.
        ListGroup {
            // TurboQuant KV
            Row(label: String(localized: "settings.experimental.turboquant.label",
                              defaultValue: "TurboQuant KV Cache",
                              comment: "Row label for the TurboQuant KV cache toggle"),
                sublabel: turboquantSublabel) {
                HStack(spacing: 8) {
                    if vm.turboquantKvEnabled {
                        Popup(
                            selection: vm.bindProfile($vm.turboquantKvBits),
                            width: 120,
                            options: ModelSettingsScreenVM.turboquantKvBitsOptions
                        )
                    }
                    Toggle("", isOn: vm.bindProfile($vm.turboquantKvEnabled))
                        .labelsHidden().toggleStyle(.switch)
                        .disabled(vm.vlmMtpEnabled)
                        .help(vm.vlmMtpEnabled ? vlmMtpOwnsSpeculativePathReason : "")
                }
            }

            // IndexCache (DSA-only — surface to the user that the row
            // only applies to models whose config matches the DSA set).
            if vm.isDSAConfigModel {
                Row(label: String(localized: "settings.experimental.indexcache.label",
                                  defaultValue: "IndexCache",
                                  comment: "Row label for the DSA IndexCache toggle"),
                    sublabel: String(localized: "settings.experimental.indexcache.sub",
                                     defaultValue: "Sparse attention index cache for DSA models. THUDM/IndexCache.",
                                     comment: "Sublabel describing the DSA IndexCache feature")) {
                    HStack(spacing: 8) {
                        if vm.indexCacheEnabled {
                            TextInput(text: vm.bindProfile($vm.indexCacheFreq),
                                      placeholder: "4", mono: true, width: 80)
                        }
                        Toggle("", isOn: vm.bindProfile($vm.indexCacheEnabled))
                            .labelsHidden().toggleStyle(.switch)
                    }
                }
            }

            // SpecPrefill
            Row(label: String(localized: "settings.experimental.specprefill.label",
                              defaultValue: "SpecPrefill",
                              comment: "Row label for the SpecPrefill toggle"),
                sublabel: specprefillSublabel) {
                Toggle("", isOn: vm.bindProfile($vm.specprefillEnabled))
                    .labelsHidden().toggleStyle(.switch)
                    .disabled(vm.vlmMtpEnabled)
                    .help(vm.vlmMtpEnabled ? vlmMtpOwnsSpeculativePathReason : "")
            }
            if vm.specprefillEnabled {
                Row(label: String(localized: "settings.experimental.specprefill.draft.label",
                                  defaultValue: "Draft Model",
                                  comment: "Row label for the SpecPrefill draft-model picker"),
                    sublabel: String(localized: "settings.experimental.specprefill.draft.sub",
                                     defaultValue: "Small model sharing tokenizer with target.",
                                     comment: "Sublabel for the SpecPrefill draft-model picker")) {
                    Popup(
                        selection: vm.bindProfile($vm.specprefillDraftModel),
                        width: 260,
                        options: vm.draftModelOptions()
                    )
                }
                Row(label: String(localized: "settings.experimental.specprefill.keep_rate.label",
                                  defaultValue: "Keep Rate",
                                  comment: "Row label for the SpecPrefill keep-rate dropdown")) {
                    Popup(
                        selection: vm.bindProfile($vm.specprefillKeepPct),
                        width: 320,
                        options: ModelSettingsScreenVM.specprefillKeepPctOptions
                    )
                }
                Row(label: String(localized: "settings.experimental.specprefill.threshold.label",
                                  defaultValue: "Threshold",
                                  comment: "Row label for the SpecPrefill threshold field"),
                    sublabel: String(localized: "settings.experimental.specprefill.threshold.sub",
                                     defaultValue: "Min prompt tokens to trigger (shorter prompts use full prefill).",
                                     comment: "Sublabel for the SpecPrefill threshold field")) {
                    TextInput(text: vm.bindProfile($vm.specprefillThreshold),
                              placeholder: "8192", mono: true, suffix: "tk", width: 110)
                }
            }

            // DFlash
            Row(label: String(localized: "settings.experimental.dflash.label",
                              defaultValue: "DFlash",
                              comment: "Row label for the DFlash toggle"),
                sublabel: dflashSublabel) {
                Toggle("", isOn: vm.bindProfile($vm.dflashEnabled))
                    .labelsHidden().toggleStyle(.switch)
                    .disabled(dflashToggleDisabled)
                    .help(dflashHelp)
            }
            if vm.dflashEnabled {
                Row(label: String(localized: "settings.experimental.dflash.draft.label",
                                  defaultValue: "DFlash Draft Model",
                                  comment: "Row label for the DFlash draft-model picker")) {
                    Popup(
                        selection: vm.bindProfile($vm.dflashDraftModel),
                        width: 260,
                        options: vm.draftModelOptions()
                    )
                }
                Row(label: String(localized: "settings.experimental.dflash.draft_quant.label",
                                  defaultValue: "Draft Quantization",
                                  comment: "Row label for the DFlash draft quantization toggle"),
                    sublabel: String(localized: "settings.experimental.dflash.draft_quant.sub",
                                     defaultValue: "Enable quantization for the draft model (weight, activation bits & group size).",
                                     comment: "Sublabel for the DFlash draft quantization toggle")) {
                    Toggle("", isOn: vm.bindProfile($vm.dflashDraftQuantEnabled))
                        .labelsHidden().toggleStyle(.switch)
                }
                if vm.dflashDraftQuantEnabled {
                    Row(label: String(localized: "settings.experimental.dflash.draft_quant_weight.label",
                                      defaultValue: "Weight Bits",
                                      comment: "Row label for the DFlash draft quantization weight bits picker")) {
                        Popup(
                            selection: vm.bindProfile($vm.dflashDraftQuantWeightBits),
                            width: 110,
                            options: ModelSettingsScreenVM.dflashDraftQuantWeightBitsOptions
                        )
                    }
                    Row(label: String(localized: "settings.experimental.dflash.draft_quant_activation.label",
                                      defaultValue: "Activation Bits",
                                      comment: "Row label for the DFlash draft quantization activation bits picker")) {
                        Popup(
                            selection: vm.bindProfile($vm.dflashDraftQuantActivationBits),
                            width: 110,
                            options: ModelSettingsScreenVM.dflashDraftQuantActivationBitsOptions
                        )
                    }
                    Row(label: String(localized: "settings.experimental.dflash.draft_quant_group.label",
                                      defaultValue: "Group Size",
                                      comment: "Row label for the DFlash draft quantization group size picker")) {
                        Popup(
                            selection: vm.bindProfile($vm.dflashDraftQuantGroupSize),
                            width: 110,
                            options: ModelSettingsScreenVM.dflashDraftQuantGroupSizeOptions
                        )
                    }
                }
                Row(label: String(localized: "settings.experimental.dflash.max_ctx.label",
                                  defaultValue: "Max Context (fallback)",
                                  comment: "Row label for the DFlash max-context fallback field"),
                    sublabel: String(localized: "settings.experimental.dflash.max_ctx.sub",
                                     defaultValue: "Prompts at or above this token count switch to BatchedEngine. Empty = unlimited.",
                                     comment: "Sublabel describing the DFlash max-context fallback")) {
                    TextInput(text: vm.bindProfile($vm.dflashMaxCtx),
                              placeholder: String(localized: "settings.experimental.dflash.max_ctx.placeholder",
                                                  defaultValue: "unlimited",
                                                  comment: "Placeholder shown when DFlash max-context is unset (no cap)"),
                              mono: true, suffix: "tk", width: 130)
                }
                Row(label: String(localized: "settings.experimental.dflash.verify_mode.label",
                                  defaultValue: "Verify Mode",
                                  comment: "Row label for the DFlash verifier algorithm picker"),
                    sublabel: String(localized: "settings.experimental.dflash.verify_mode.sub",
                                     defaultValue: "Verifier algorithm. \"adaptive\" shrinks block size when acceptance drops; \"off\" disables speculative verify.",
                                     comment: "Sublabel for the DFlash verify mode picker")) {
                    Popup(
                        selection: vm.bindProfile($vm.dflashVerifyMode),
                        width: 140,
                        options: ModelSettingsScreenVM.dflashVerifyModeOptions
                    )
                }
                Row(label: String(localized: "settings.experimental.dflash.window_size.label",
                                  defaultValue: "Draft Window Size",
                                  comment: "Row label for the DFlash draft sliding-attention window size field"),
                    sublabel: String(localized: "settings.experimental.dflash.window_size.sub",
                                     defaultValue: "Draft model sliding-attention window. Empty = dflash default (1024).",
                                     comment: "Sublabel for the DFlash draft window size field")) {
                    TextInput(text: vm.bindProfile($vm.dflashDraftWindowSize),
                              placeholder: "1024", mono: true, width: 110)
                }
                Row(label: String(localized: "settings.experimental.dflash.sink_size.label",
                                  defaultValue: "Draft Sink Size",
                                  comment: "Row label for the DFlash attention-sink tokens field"),
                    sublabel: String(localized: "settings.experimental.dflash.sink_size.sub",
                                     defaultValue: "Attention-sink tokens always kept in the window. Empty = dflash default (64).",
                                     comment: "Sublabel for the DFlash draft sink size field")) {
                    TextInput(text: vm.bindProfile($vm.dflashDraftSinkSize),
                              placeholder: "64", mono: true, width: 110)
                }
                Row(label: String(localized: "settings.experimental.dflash.mem_cache.label",
                                  defaultValue: "DFlash in-memory cache",
                                  comment: "Row label for the DFlash L1 in-memory cache toggle"),
                    sublabel: String(localized: "settings.experimental.dflash.mem_cache.sub",
                                     defaultValue: "DFlash L1 prefix snapshot cache in RAM.",
                                     comment: "Sublabel for the DFlash L1 in-memory cache toggle")) {
                    HStack(spacing: 8) {
                        if vm.dflashInMemoryCache {
                            TextInput(text: vm.bindProfile($vm.dflashInMemoryCacheGib),
                                      placeholder: "8", mono: true, suffix: "GiB", width: 110)
                        }
                        Toggle("", isOn: vm.bindProfile($vm.dflashInMemoryCache))
                            .labelsHidden().toggleStyle(.switch)
                    }
                }
                if vm.dflashInMemoryCache {
                    Row(label: String(localized: "settings.experimental.dflash.mem_cache_entries.label",
                                      defaultValue: "Cache Entries",
                                      comment: "Row label for the DFlash L1 in-memory cache max entries field"),
                        sublabel: String(localized: "settings.experimental.dflash.mem_cache_entries.sub",
                                         defaultValue: "Maximum prefix snapshots kept in RAM. Each entry stores KV + draft GDN state.",
                                         comment: "Sublabel for the DFlash L1 cache max entries field")) {
                        TextInput(text: vm.bindProfile($vm.dflashInMemoryCacheMaxEntries),
                                  placeholder: "4", mono: true, width: 110)
                    }
                }
                Row(label: String(localized: "settings.experimental.dflash.ssd_cache.label",
                                  defaultValue: "DFlash SSD cache",
                                  comment: "Row label for the DFlash L2 SSD cache toggle"),
                    sublabel: dflashSsdSublabel) {
                    Toggle("", isOn: vm.bindProfile($vm.dflashSsdCache))
                        .labelsHidden().toggleStyle(.switch)
                        .disabled(!(vm.model?.dflashSsdCacheAvailable ?? false) || !vm.dflashInMemoryCache)
                }
                if vm.dflashSsdCache && (vm.model?.dflashSsdCacheAvailable ?? false) {
                    Row(label: String(localized: "settings.experimental.dflash.ssd_cache_size.label",
                                      defaultValue: "SSD Cache Size",
                                      comment: "Row label for the DFlash L2 SSD cache disk budget field"),
                        sublabel: String(localized: "settings.experimental.dflash.ssd_cache_size.sub",
                                         defaultValue: "Disk budget for L2 spill; oldest entries are evicted when exceeded.",
                                         comment: "Sublabel for the DFlash SSD cache size field")) {
                        TextInput(text: vm.bindProfile($vm.dflashSsdCacheGib),
                                  placeholder: "20", mono: true, suffix: "GiB", width: 110)
                    }
                }
            }

            // Native MTP
            Row(label: String(localized: "settings.experimental.mtp.label",
                              defaultValue: "Native MTP",
                              comment: "Row label for the Native MTP toggle"),
                sublabel: mtpSublabel) {
                Toggle("", isOn: vm.bindProfile($vm.mtpEnabled))
                    .labelsHidden().toggleStyle(.switch)
                    .disabled(mtpToggleDisabled)
                    .help(vm.mtpConflictReason ?? vm.model?.mtpCompatibilityReason ?? "")
            }

            // VLM MTP — last row of the experimental group. Reveals the
            // draft-model picker and block-size field when enabled.
            Row(label: String(localized: "settings.experimental.vlm_mtp.label",
                              defaultValue: "VLM MTP",
                              comment: "Row label for the VLM MTP toggle"),
                sublabel: vlmMtpSublabel,
                isLast: !vm.vlmMtpEnabled) {
                Toggle("", isOn: vm.bindProfile($vm.vlmMtpEnabled))
                    .labelsHidden().toggleStyle(.switch)
                    .disabled(vlmMtpToggleDisabled)
                    .help(vm.vlmMtpConflictReason ?? "")
            }
            if vm.vlmMtpEnabled {
                Row(label: String(localized: "settings.experimental.vlm_mtp.draft.label",
                                  defaultValue: "VLM Draft Model",
                                  comment: "Row label for the VLM MTP draft-model picker"),
                    sublabel: String(localized: "settings.experimental.vlm_mtp.draft.sub",
                                     defaultValue: "Assistant drafter sharing the target's tokenizer.",
                                     comment: "Sublabel for the VLM MTP draft-model picker")) {
                    Popup(
                        selection: vm.bindProfile($vm.vlmMtpDraftModel),
                        width: 260,
                        options: vm.vlmMtpDraftModelOptions()
                    )
                }
                Row(label: String(localized: "settings.experimental.vlm_mtp.block_size.label",
                                  defaultValue: "Draft Block Size",
                                  comment: "Row label for the VLM MTP draft block-size field"),
                    sublabel: String(localized: "settings.experimental.vlm_mtp.block_size.sub",
                                     defaultValue: "Tokens drafted per round. Empty uses the mlx-vlm default.",
                                     comment: "Sublabel for the VLM MTP draft block-size field"),
                    isLast: true) {
                    TextInput(text: vm.bindProfile($vm.vlmMtpDraftBlockSize),
                              placeholder: "4", mono: true, width: 80)
                }
            }
        }
    }

    private var vlmMtpOwnsSpeculativePathReason: String {
        String(localized: "settings.speculative.conflict.vlm_mtp",
               defaultValue: "Disable VLM MTP before enabling this feature.",
               comment: "Tooltip / sublabel shown when another speculative feature can't be enabled because VLM MTP is on")
    }

    private var turboquantSublabel: String {
        if vm.vlmMtpEnabled { return vlmMtpOwnsSpeculativePathReason }
        return String(localized: "settings.experimental.turboquant.sub",
                      defaultValue: "Quantize the KV cache during prefill. Saves memory at a small quality cost.",
                      comment: "Sublabel describing TurboQuant KV cache")
    }

    private var specprefillSublabel: String {
        if vm.vlmMtpEnabled { return vlmMtpOwnsSpeculativePathReason }
        return String(localized: "settings.experimental.specprefill.sub",
                      defaultValue: "Attention-based sparse prefill for MoE/hybrid models.",
                      comment: "Sublabel describing SpecPrefill")
    }

    private var dflashToggleDisabled: Bool {
        !(vm.model?.dflashCompatible ?? true) || vm.vlmMtpEnabled
    }

    private var dflashHelp: String {
        if let reason = vm.model?.dflashCompatibilityReason,
           !(vm.model?.dflashCompatible ?? true) {
            return reason
        }
        return vm.vlmMtpEnabled ? vlmMtpOwnsSpeculativePathReason : ""
    }

    private var dflashSublabel: String {
        if let reason = vm.model?.dflashCompatibilityReason,
           !(vm.model?.dflashCompatible ?? true) {
            return reason
        }
        if vm.vlmMtpEnabled { return vlmMtpOwnsSpeculativePathReason }
        return String(localized: "settings.experimental.dflash.sub",
                      defaultValue: "Block-diffusion speculative decoding. Single-stream only (requests run one at a time).",
                      comment: "Default sublabel for the DFlash toggle (used when the model is compatible)")
    }

    private var dflashSsdSublabel: String {
        if !(vm.model?.dflashSsdCacheAvailable ?? false) {
            return String(localized: "settings.experimental.dflash.ssd_cache.sub.unavailable",
                          defaultValue: "Enable the global paged SSD cache directory first.",
                          comment: "Sublabel for the DFlash SSD cache row when the global SSD cache directory isn't configured")
        }
        if !vm.dflashInMemoryCache {
            return String(localized: "settings.experimental.dflash.ssd_cache.sub.needs_l1",
                          defaultValue: "Requires the in-memory cache to be enabled.",
                          comment: "Sublabel for the DFlash SSD cache row when the L1 in-memory cache is off")
        }
        return String(localized: "settings.experimental.dflash.ssd_cache.sub",
                      defaultValue: "L2 spill of evicted L1 entries to disk.",
                      comment: "Default sublabel for the DFlash SSD cache toggle")
    }

    private var mtpToggleDisabled: Bool {
        let compatible = vm.model?.mtpCompatible ?? true
        if !compatible && !vm.mtpEnabled { return true }
        if vm.mtpConflictReason != nil { return true }
        return false
    }

    private var mtpSublabel: String {
        if let reason = vm.mtpConflictReason { return reason }
        if let reason = vm.model?.mtpCompatibilityReason,
           !(vm.model?.mtpCompatible ?? true) {
            return reason
        }
        return String(localized: "settings.experimental.mtp.sub",
                      defaultValue: "Multi-token prediction. Speeds generation when the model supports it.",
                      comment: "Default sublabel for the Native MTP toggle")
    }

    private var vlmMtpToggleDisabled: Bool {
        vm.vlmMtpConflictReason != nil
    }

    private var vlmMtpSublabel: String {
        if let reason = vm.vlmMtpConflictReason { return reason }
        return String(localized: "settings.experimental.vlm_mtp.sub",
                      defaultValue: "Multi-token prediction for vision-language models via an assistant drafter.",
                      comment: "Default sublabel for the VLM MTP toggle")
    }
}

// MARK: - Sampling validators
//
// Empty input is always valid and maps to nil — the server treats nil as
// "unset, fall back to model default". A non-empty value that fails to
// parse or falls outside the documented range is rejected before the
// patch is sent, so a slipped keystroke can't silently overwrite the
// server with an out-of-band value.

struct SamplingValidationError: Error, Equatable {
    let message: String
}

enum SamplingValidator {
    static func temperature(_ raw: String) -> Result<Double?, SamplingValidationError> {
        let label = String(localized: "settings.validator.temperature.name",
                           defaultValue: "Temperature",
                           comment: "Field name embedded in validation errors for temperature")
        return parseDouble(raw, label: label) { v in
            v >= 0 ? nil : String(localized: "settings.validator.temperature.range",
                                  defaultValue: "Temperature must be ≥ 0.",
                                  comment: "Validation error when temperature is below the allowed range")
        }
    }

    static func topP(_ raw: String) -> Result<Double?, SamplingValidationError> {
        let label = String(localized: "settings.validator.top_p.name",
                           defaultValue: "Top P",
                           comment: "Field name embedded in validation errors for top-p")
        return parseDouble(raw, label: label) { v in
            (v > 0 && v <= 1) ? nil : String(localized: "settings.validator.top_p.range",
                                             defaultValue: "Top P must be in (0, 1].",
                                             comment: "Validation error when top-p falls outside the allowed range")
        }
    }

    static func minP(_ raw: String) -> Result<Double?, SamplingValidationError> {
        let label = String(localized: "settings.validator.min_p.name",
                           defaultValue: "Min P",
                           comment: "Field name embedded in validation errors for min-p")
        return parseDouble(raw, label: label) { v in
            (v >= 0 && v <= 1) ? nil : String(localized: "settings.validator.min_p.range",
                                              defaultValue: "Min P must be in [0, 1].",
                                              comment: "Validation error when min-p falls outside the allowed range")
        }
    }

    static func topK(_ raw: String) -> Result<Int?, SamplingValidationError> {
        let t = raw.trimmingCharacters(in: .whitespaces)
        if t.isEmpty { return .success(nil) }
        guard let v = Int(t) else {
            return .failure(.init(message: String(localized: "settings.validator.top_k.integer",
                                                  defaultValue: "Top K must be an integer.",
                                                  comment: "Validation error when top-k isn't an integer")))
        }
        guard v >= 1 else {
            return .failure(.init(message: String(localized: "settings.validator.top_k.positive",
                                                  defaultValue: "Top K must be a positive integer.",
                                                  comment: "Validation error when top-k isn't positive")))
        }
        return .success(v)
    }

    static func penalty(_ raw: String, name: String) -> Result<Double?, SamplingValidationError> {
        parseDouble(raw, label: name) { v in
            (v >= -2 && v <= 2) ? nil : String(localized: "settings.validator.penalty.range",
                                               defaultValue: "\(name) must be in [-2, 2].",
                                               comment: "Validation error when a penalty field is outside [-2,2]; placeholder is the field name")
        }
    }

    private static func parseDouble(
        _ raw: String,
        label: String,
        check: (Double) -> String?
    ) -> Result<Double?, SamplingValidationError> {
        let t = raw.trimmingCharacters(in: .whitespaces)
        if t.isEmpty { return .success(nil) }
        guard let v = Double(t) else {
            return .failure(.init(message: String(localized: "settings.validator.must_be_number",
                                                  defaultValue: "\(label) must be a number.",
                                                  comment: "Validation error when a sampling field isn't a number; placeholder is the field name")))
        }
        if let msg = check(v) { return .failure(.init(message: msg)) }
        return .success(v)
    }
}

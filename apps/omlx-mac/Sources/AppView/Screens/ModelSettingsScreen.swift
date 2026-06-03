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

    @EnvironmentObject private var services: AppServices
    @StateObject private var vm = ModelSettingsScreenVM()

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Header(model: vm.model, onBack: { services.modelDetailID = nil })

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
        .task(id: modelID) { await vm.load(modelID: modelID, client: services.client) }
    }
}

// MARK: - Header

private struct Header: View {
    let model: ModelDTO?
    let onBack: () -> Void

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        HStack(spacing: 12) {
            Squircle(systemSymbol: "cpu", size: 44, gradient: SquircleGradient.models)
            VStack(alignment: .leading, spacing: 2) {
                Text(model?.settings?.displayName ?? model?.id ?? "—")
                    .font(.omlxText(17, weight: .semibold))
                    .foregroundStyle(theme.text)
                    .lineLimit(1)
                    .truncationMode(.tail)
                if let m = model {
                    Text("\(m.id) · \(m.estimatedSizeFormatted ?? formatBytes(m.estimatedSize))")
                        .font(.omlxMono(11))
                        .foregroundStyle(theme.textSecondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }
            Spacer(minLength: 8)
            Button {
                onBack()
            } label: {
                Label(String(localized: "settings.header.back_to_models",
                             defaultValue: "Back to Models",
                             comment: "Back button label at the top of the per-model settings screen"),
                      systemImage: "chevron.left")
                    .labelStyle(.titleAndIcon)
            }
            .buttonStyle(.omlx(.plain, size: .small))
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
            .frame(width: 320)
            Spacer()
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 6)
    }
}

// MARK: - Profiles tab

private struct ProfilesTab: View {
    @ObservedObject var vm: ModelSettingsScreenVM
    /// Source of `.preset` chips — the shipped JSON bundle, refreshable
    /// from omlx.ai via `POST /api/presets/refresh`. Replaces the legacy
    /// `vm.templates.filter { isBuiltin }` source after Phase 1 retired
    /// the server-side builtin templates.
    @ObservedObject var presetStore: PresetBundleStore
    let client: OMLXClient
    /// Optional binding to a Server-Defaults DTO surfaced read-only at
    /// the bottom of the tab. Lives on the parent (a `@StateObject`-
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
                onClosePreview: { self.preview = nil }
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
                    hasWorking: false
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
    @ObservedObject var vm: ModelSettingsScreenVM
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
    @ObservedObject var vm: ModelSettingsScreenVM
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
    @ObservedObject var vm: ModelSettingsScreenVM
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

// MARK: - Chat-template kwargs editor

private struct ChatTemplateKwargsEditor: View {
    @ObservedObject var vm: ModelSettingsScreenVM
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
                                  defaultValue: "\(vm.chatTemplateEntries.count) kwarg\(vm.chatTemplateEntries.count == 1 ? "" : "s")",
                                  comment: "Count summary in the chat-template editor; placeholders are the entry count and an optional plural 's'"))
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
            if !vm.chatTemplateEntries.contains(where: { $0.kind == .enableThinking }) {
                Button("enable_thinking") {
                    vm.addKwarg(.enableThinking)
                }
            }
            if !vm.chatTemplateEntries.contains(where: { $0.kind == .reasoningEffort }) {
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
    @ObservedObject var vm: ModelSettingsScreenVM
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
    @ObservedObject var vm: ModelSettingsScreenVM
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
                sublabel: String(localized: "settings.experimental.turboquant.sub",
                                 defaultValue: "Quantize the KV cache during prefill. Saves memory at a small quality cost.",
                                 comment: "Sublabel describing TurboQuant KV cache")) {
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
                sublabel: String(localized: "settings.experimental.specprefill.sub",
                                 defaultValue: "Attention-based sparse prefill for MoE/hybrid models.",
                                 comment: "Sublabel describing SpecPrefill")) {
                Toggle("", isOn: vm.bindProfile($vm.specprefillEnabled))
                    .labelsHidden().toggleStyle(.switch)
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
                    .disabled(!(vm.model?.dflashCompatible ?? true))
                    .help(vm.model?.dflashCompatibilityReason ?? "")
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
                                  comment: "Row label for the DFlash draft-quantization picker")) {
                    Popup(
                        selection: vm.bindProfile($vm.dflashDraftQuantBits),
                        width: 160,
                        options: ModelSettingsScreenVM.dflashDraftQuantOptions
                    )
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
                Row(label: String(localized: "settings.experimental.dflash.ssd_cache.label",
                                  defaultValue: "DFlash SSD cache",
                                  comment: "Row label for the DFlash L2 SSD cache toggle"),
                    sublabel: dflashSsdSublabel) {
                    Toggle("", isOn: vm.bindProfile($vm.dflashSsdCache))
                        .labelsHidden().toggleStyle(.switch)
                        .disabled(!(vm.model?.dflashSsdCacheAvailable ?? false) || !vm.dflashInMemoryCache)
                }
            }

            // Native MTP — last row of the experimental group.
            Row(label: String(localized: "settings.experimental.mtp.label",
                              defaultValue: "Native MTP",
                              comment: "Row label for the Native MTP toggle"),
                sublabel: mtpSublabel,
                isLast: true) {
                Toggle("", isOn: vm.bindProfile($vm.mtpEnabled))
                    .labelsHidden().toggleStyle(.switch)
                    .disabled(mtpToggleDisabled)
                    .help(vm.mtpConflictReason ?? vm.model?.mtpCompatibilityReason ?? "")
            }
        }
    }

    private var dflashSublabel: String {
        if let reason = vm.model?.dflashCompatibilityReason,
           !(vm.model?.dflashCompatible ?? true) {
            return reason
        }
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
}

// MARK: - View model

@MainActor
final class ModelSettingsScreenVM: ObservableObject {
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
        case forceSampling, isPinned
        case trustRemoteCode
        case reasoningParser
        case chatTemplateKwargs
        case turboquantKvEnabled, turboquantKvBits
        case indexCacheEnabled, indexCacheFreq
        case specprefillEnabled, specprefillDraftModel, specprefillKeepPct, specprefillThreshold
        case dflashEnabled, dflashDraftModel, dflashDraftQuantBits, dflashMaxCtx
        case dflashInMemoryCache, dflashInMemoryCacheGib, dflashSsdCache
        case mtpEnabled
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

    static var dflashDraftQuantOptions: [(String, String)] {
        [
            ("", String(localized: "settings.dflash.quant.bf16",
                        defaultValue: "bf16 (default)",
                        comment: "DFlash draft quantization option for bf16 (no quantization)")),
            ("4", String(localized: "settings.dflash.quant.4bit",
                         defaultValue: "4-bit",
                         comment: "DFlash draft quantization option")),
            ("8", String(localized: "settings.dflash.quant.8bit",
                         defaultValue: "8-bit",
                         comment: "DFlash draft quantization option")),
        ]
    }

    /// `config_model_type` values that surface IndexCache in the HTML
    /// admin. Mirrored from `dashboard.js:5-7` (`DSA_MODEL_TYPES`).
    static let dsaConfigModelTypes: Set<String> = [
        "deepseek_v32", "glm_moe_dsa",
    ]

    @Published var section: Section = .basic

    @Published var model: ModelDTO?
    /// Snapshot of every other model on the server, used to populate the
    /// SpecPrefill / DFlash draft-model dropdowns. Reloaded with `load()`.
    @Published var allModels: [ModelDTO] = []
    @Published var modelID: String = ""
    @Published var lastError: String?

    // Basic
    @Published var alias: String = ""
    @Published var modelTypeOverride: String = ""
    @Published var contextLength: String = ""
    @Published var maxTokens: String = ""
    @Published var temperature: String = ""
    @Published var topP: String = ""
    @Published var topK: String = ""
    @Published var minP: String = ""
    @Published var repetitionPenalty: String = ""
    @Published var presencePenalty: String = ""
    @Published var ttlSeconds: String = ""

    // Advanced
    @Published var enableThinking: Bool = true
    @Published var thinkingBudgetEnabled: Bool = false
    @Published var thinkingBudgetTokens: String = "8192"
    @Published var limitToolResults: Bool = false
    /// Token cap when `limitToolResults` is on. Defaults to the HTML
    /// admin's seeded value so the first save after enabling sends a
    /// sensible number instead of zero (which the server interprets as
    /// "disabled").
    @Published var toolResultLimitTokens: String = "4096"
    @Published var forceSampling: Bool = false
    @Published var isPinned: Bool = false

    // Security
    @Published var trustRemoteCode: Bool = false

    // Reasoning parser (free-form override; empty = auto)
    @Published var reasoningParser: String = ""

    // Chat-template kwargs — entries are the editor's view of the
    // (chat_template_kwargs, forced_ct_kwargs) server pair.
    @Published var chatTemplateEntries: [ChatTemplateKwargEntry] = []

    // Experimental: TurboQuant KV
    @Published var turboquantKvEnabled: Bool = false
    @Published var turboquantKvBits: String = "4"

    // Experimental: IndexCache (DSA-only)
    @Published var indexCacheEnabled: Bool = false
    @Published var indexCacheFreq: String = "4"

    // Experimental: SpecPrefill
    @Published var specprefillEnabled: Bool = false
    @Published var specprefillDraftModel: String = ""
    @Published var specprefillKeepPct: String = "0.2"
    @Published var specprefillThreshold: String = "8192"

    // Experimental: DFlash
    @Published var dflashEnabled: Bool = false
    @Published var dflashDraftModel: String = ""
    /// "" (bf16 default), "4", or "8".
    @Published var dflashDraftQuantBits: String = ""
    @Published var dflashMaxCtx: String = ""
    @Published var dflashInMemoryCache: Bool = false
    @Published var dflashInMemoryCacheGib: String = "8"
    @Published var dflashSsdCache: Bool = false

    // Experimental: native MTP
    @Published var mtpEnabled: Bool = false

    // Profiles
    @Published var profiles: [ProfileDTO] = []
    @Published var templates: [ProfileDTO] = []
    @Published var activeProfileName: String?
    /// Server's `GlobalSettings.sampling` snapshot, loaded alongside the
    /// per-model settings so the Profiles tab's "Server Defaults" card
    /// can render without a second round-trip.
    @Published var serverDefaultSampling: GlobalSettingsDTO.SamplingDTO?
    /// Display scope for the active profile (derived from `source_template`).
    @Published var activeProfileScope: ProfileScope = .model
    /// True when one or more profile-eligible fields have been edited
    /// since the last load / apply / save. Flips the screen into the
    /// "Working profile" state. Per-model fields (alias / modelType /
    /// ttl / isPinned / trustRemoteCode) auto-save and never set this.
    @Published var profileDirty: Bool = false

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
                if let s = m.settings {
                    self.alias = s.modelAlias ?? ""
                    self.modelTypeOverride = s.modelTypeOverride ?? ""
                    self.contextLength = s.maxContextWindow.map(String.init) ?? ""
                    self.maxTokens = s.maxTokens.map(String.init) ?? ""
                    self.temperature = s.temperature.map { String($0) } ?? ""
                    self.topP = s.topP.map { String($0) } ?? ""
                    self.topK = s.topK.map(String.init) ?? ""
                    self.minP = s.minP.map { String($0) } ?? ""
                    self.repetitionPenalty = s.repetitionPenalty.map { String($0) } ?? ""
                    self.presencePenalty = s.presencePenalty.map { String($0) } ?? ""
                    self.ttlSeconds = s.ttlSeconds.map(String.init) ?? ""
                    self.enableThinking = s.enableThinking ?? true
                    self.thinkingBudgetEnabled = s.thinkingBudgetEnabled ?? false
                    self.thinkingBudgetTokens = s.thinkingBudgetTokens.map(String.init) ?? "8192"
                    self.limitToolResults = (s.maxToolResultTokens ?? 0) > 0
                    if let n = s.maxToolResultTokens, n > 0 {
                        self.toolResultLimitTokens = String(n)
                    }
                    self.forceSampling = s.forceSampling ?? false
                    self.isPinned = s.isPinned ?? false
                    self.trustRemoteCode = s.trustRemoteCode ?? false
                    self.reasoningParser = s.reasoningParser ?? ""
                    self.chatTemplateEntries = ChatTemplateKwargsCodec.decode(
                        kwargs: s.chatTemplateKwargs,
                        forced: s.forcedCtKwargs
                    )
                    self.turboquantKvEnabled = s.turboquantKvEnabled ?? false
                    self.turboquantKvBits = s.turboquantKvBits.map { Self.formatBits($0) } ?? "4"
                    self.indexCacheEnabled = s.indexCacheFreq != nil
                    self.indexCacheFreq = s.indexCacheFreq.map(String.init) ?? "4"
                    self.specprefillEnabled = s.specprefillEnabled ?? false
                    self.specprefillDraftModel = s.specprefillDraftModel ?? ""
                    self.specprefillKeepPct = s.specprefillKeepPct.map { Self.formatPct($0) } ?? "0.2"
                    self.specprefillThreshold = s.specprefillThreshold.map(String.init) ?? "8192"
                    self.dflashEnabled = s.dflashEnabled ?? false
                    self.dflashDraftModel = s.dflashDraftModel ?? ""
                    self.dflashDraftQuantBits = s.dflashDraftQuantBits.map(String.init) ?? ""
                    self.dflashMaxCtx = s.dflashMaxCtx.map(String.init) ?? ""
                    self.dflashInMemoryCache = s.dflashInMemoryCache ?? false
                    self.dflashInMemoryCacheGib = DflashByteSize.bytesToGib(s.dflashInMemoryCacheMaxBytes)
                        .map(String.init) ?? "8"
                    self.dflashSsdCache = s.dflashSsdCache ?? false
                    self.mtpEnabled = s.mtpEnabled ?? false
                    self.activeProfileName = s.activeProfileName
                }
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
        case .trustRemoteCode:         patch.trustRemoteCode = trustRemoteCode
        case .reasoningParser:
            patch.reasoningParser = reasoningParser.isEmpty ? nil : reasoningParser
        case .chatTemplateKwargs:
            let pair = ChatTemplateKwargsCodec.encode(chatTemplateEntries)
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
        case .dflashDraftQuantBits:    patch.dflashDraftQuantBits = Int(dflashDraftQuantBits)
        case .dflashMaxCtx:            patch.dflashMaxCtx = Int(dflashMaxCtx)
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
        case .dflashSsdCache:          patch.dflashSsdCache = dflashSsdCache
        case .mtpEnabled:              patch.mtpEnabled = mtpEnabled
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
        return nil
    }

    // MARK: - Working profile dict assembly

    /// Snapshot the current profile-eligible field values into the
    /// loose `settings` dict the server stores on profiles + templates.
    /// Keys are snake_case (the server's wire shape). Empty / unparseable
    /// fields are dropped — the server treats absent keys as "use defaults".
    func currentSettingsDict() -> [String: AnyCodable] {
        var out: [String: AnyCodable] = [:]

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
        putDouble(ProfileSettingsKey.topP, topP)
        putInt(ProfileSettingsKey.topK, topK)
        putDouble(ProfileSettingsKey.minP, minP)
        putDouble(ProfileSettingsKey.repetitionPenalty, repetitionPenalty)
        putDouble(ProfileSettingsKey.presencePenalty, presencePenalty)

        // Universal — thinking / tool / reasoning
        putBool(ProfileSettingsKey.enableThinking, enableThinking)
        putBool(ProfileSettingsKey.thinkingBudgetEnabled, thinkingBudgetEnabled)
        putInt(ProfileSettingsKey.thinkingBudgetTokens, thinkingBudgetTokens)
        putBool(ProfileSettingsKey.forceSampling, forceSampling)
        putString(ProfileSettingsKey.reasoningParser, reasoningParser)
        // Server uses 0 as the "disable" sentinel; encode that exactly.
        out[ProfileSettingsKey.maxToolResultTokens] = AnyCodable(
            limitToolResults ? (Int(toolResultLimitTokens) ?? 4096) : 0
        )

        // Universal — chat template kwargs. AnyCodable's encode walks a
        // [String: AnyCodable] / [AnyCodable] explicitly, so nest those
        // shapes rather than `Any` so the Sendable check is satisfied.
        let kwargs = ChatTemplateKwargsCodec.encode(chatTemplateEntries)
        if let dict = kwargs.kwargs {
            out[ProfileSettingsKey.chatTemplateKwargs] = AnyCodable(dict)
        }
        if let forced = kwargs.forced, !forced.isEmpty {
            out[ProfileSettingsKey.forcedCtKwargs] = AnyCodable(
                forced.map { AnyCodable($0) }
            )
        }

        // Model-specific — experimental
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
            putInt(ProfileSettingsKey.dflashDraftQuantBits, dflashDraftQuantBits)
            putInt(ProfileSettingsKey.dflashMaxCtx, dflashMaxCtx)
            putBool(ProfileSettingsKey.dflashInMemoryCache, dflashInMemoryCache)
            if let bytes = DflashByteSize.gibToBytes(Int(dflashInMemoryCacheGib)) {
                out[ProfileSettingsKey.dflashInMemoryCacheMaxBytes] = AnyCodable(Int(bytes))
            }
            putBool(ProfileSettingsKey.dflashSsdCache, dflashSsdCache)
        }
        putBool(ProfileSettingsKey.mtpEnabled, mtpEnabled)

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

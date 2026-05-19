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
                Label("Back to Models", systemImage: "chevron.left")
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
                label: "Preset Profiles",
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
                label: "Global Profiles",
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
                label: "Model Profiles · \(vm.model?.id ?? vm.modelID)",
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
                "Server Defaults",
                subtitle: "Used when no profile is set, or when a profile leaves a field empty"
            ) {
                Button("Edit on Server →") { onEditServer() }
                    .buttonStyle(.omlx(.plain, size: .small))
            }
            ProfileDetailCard(
                name: "Server Default Profile",
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
                    name: "Working profile",
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
                    name: "No profile",
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
            await MainActor.run { vm.lastError = describe(error) }
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

    private func describe(_ error: Error) -> String {
        if let omlx = error as? OMLXClientError { return String(describing: omlx) }
        return error.localizedDescription
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
        SectionHeader("Basic Settings")

        // Per-model fields (alias / modelType / TTL) auto-save on commit.
        // Profile-eligible fields (sampling, penalties) write to the
        // working profile instead — surfaced via the banner above.
        ListGroup {
            Row(label: "Model Alias", sublabel: "Falls back to the model id") {
                TextInput(text: $vm.alias, placeholder: vm.modelID, mono: true, width: 220)
                    .onSubmit { Task { await vm.save(.alias, client: client) } }
            }
            Row(label: "Model Type") {
                Popup(
                    selection: vm.bind($vm.modelTypeOverride, save: { Task { await vm.save(.modelType, client: client) } }),
                    width: 170,
                    options: ModelSettingsScreenVM.modelTypeOptions
                )
            }
            Row(label: "Context Window", sublabel: "Maximum tokens per request") {
                TextInput(text: vm.bindProfile($vm.contextLength), mono: true, suffix: "tk", width: 110)
            }
            Row(label: "Max Tokens", sublabel: "Cap on generated tokens (empty = default)") {
                TextInput(text: vm.bindProfile($vm.maxTokens), placeholder: "Default", mono: true, width: 110)
            }
            Row(label: "Temperature",
                sublabel: "Sampling randomness (≥ 0). 0 = deterministic.") {
                TextInput(text: vm.bindProfile($vm.temperature), placeholder: "0.7", mono: true, width: 90)
            }
            Row(label: "Top P",
                sublabel: "Nucleus sampling cutoff (0 < p ≤ 1).") {
                TextInput(text: vm.bindProfile($vm.topP), mono: true, width: 90)
            }
            Row(label: "Top K",
                sublabel: "Limit candidates to top K (positive integer).") {
                TextInput(text: vm.bindProfile($vm.topK), mono: true, width: 90)
            }
            Row(label: "Min P",
                sublabel: "Minimum probability floor (0 ≤ p ≤ 1).") {
                TextInput(text: vm.bindProfile($vm.minP), mono: true, width: 90)
            }
            Row(label: "Repetition Penalty",
                sublabel: "Penalize repeated tokens (−2 to 2).") {
                TextInput(text: vm.bindProfile($vm.repetitionPenalty), mono: true, width: 90)
            }
            Row(label: "Presence Penalty",
                sublabel: "Penalize tokens already present (−2 to 2).") {
                TextInput(text: vm.bindProfile($vm.presencePenalty), mono: true, width: 90)
            }
            Row(
                label: "TTL",
                sublabel: "Seconds before idle unload (empty = no TTL)",
                isLast: true
            ) {
                TextInput(text: $vm.ttlSeconds, placeholder: "No TTL", mono: true, suffix: "s", width: 110)
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
        SectionHeader("Advanced Settings")

        // Profile-eligible toggles use `bindProfile` — flipping them flips
        // the working-dirty flag. `isPinned` and `trustRemoteCode` stay
        // per-model (server excludes them from profiles) and auto-save.
        ListGroup {
            Row(label: "Enable Thinking",
                sublabel: "Enable reasoning/thinking mode for this model") {
                Toggle("", isOn: vm.bindProfile($vm.enableThinking))
                    .labelsHidden().toggleStyle(.switch)
            }
            Row(label: "Thinking Budget",
                sublabel: "Limit thinking tokens for reasoning models. Forces end of thinking when exceeded.") {
                HStack(spacing: 8) {
                    if vm.thinkingBudgetEnabled {
                        TextInput(text: vm.bindProfile($vm.thinkingBudgetTokens),
                                  mono: true, suffix: "tk", width: 110)
                    }
                    Toggle("", isOn: vm.bindProfile($vm.thinkingBudgetEnabled))
                        .labelsHidden().toggleStyle(.switch)
                }
            }
            Row(label: "Limit Tool Result Tokens",
                sublabel: "Truncate large tool results (e.g. file reads) to a token limit") {
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
            Row(label: "Force Sampling",
                sublabel: "Override request sampling parameters with configured values") {
                Toggle("", isOn: vm.bindProfile($vm.forceSampling))
                    .labelsHidden().toggleStyle(.switch)
            }
            Row(label: "Reasoning Parser",
                sublabel: "Override the chain-of-thought parser. Leave empty to use the model's default.") {
                TextInput(text: vm.bindProfile($vm.reasoningParser),
                          placeholder: "auto", mono: true, width: 150)
            }
            Row(label: "Pin in memory",
                sublabel: "Keep this model resident between requests") {
                Toggle("", isOn: vm.bind($vm.isPinned, save: {
                    Task { await vm.save(.isPinned, client: client) }
                }))
                .labelsHidden().toggleStyle(.switch)
            }
            // Security-sensitive row — flagged red to match the HTML
            // editor's visual treatment. HF custom-code execution gives
            // the model author the ability to run arbitrary Python in
            // the server process; never propagated via profiles.
            Row(label: "Trust Remote Code",
                sublabel: "Execute HuggingFace custom model code. Only enable for models you trust. Per-model only — never inherited from profiles.",
                isLast: true) {
                Toggle("", isOn: vm.bind($vm.trustRemoteCode, save: {
                    Task { await vm.save(.trustRemoteCode, client: client) }
                }))
                .labelsHidden().toggleStyle(.switch)
                .tint(theme.redDot)
            }
        }

        SectionHeader(
            "Chat Template Kwargs",
            subtitle: "Forwarded to the model's chat template. Toggle Force to override per-request values."
        )
        ChatTemplateKwargsEditor(vm: vm, client: client)

        SectionHeader(
            "Experimental",
            subtitle: "Speculative decoding, KV-cache quantization, and other research features."
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
                         ? "No chat-template kwargs."
                         : "\(vm.chatTemplateEntries.count) kwarg\(vm.chatTemplateEntries.count == 1 ? "" : "s")")
                        .font(.omlxText(12))
                        .foregroundStyle(theme.textSecondary)
                    Spacer()
                    addMenu
                }
            }
            ForEach(Array(vm.chatTemplateEntries.enumerated()), id: \.element.id) { idx, entry in
                let isLast = idx == vm.chatTemplateEntries.count - 1
                FreeRow(isLast: isLast) {
                    EntryEditor(
                        vm: vm,
                        client: client,
                        index: idx,
                        entry: entry
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
            Button("custom…") {
                vm.addKwarg(.custom)
            }
        } label: {
            Label("Add kwarg", systemImage: "plus")
                .labelStyle(.titleAndIcon)
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
    }
}

private struct EntryEditor: View {
    @ObservedObject var vm: ModelSettingsScreenVM
    let client: OMLXClient
    let index: Int
    let entry: ChatTemplateKwargEntry

    @Environment(\.omlxTheme) private var theme

    private var binding: Binding<ChatTemplateKwargEntry> {
        Binding(
            get: { vm.chatTemplateEntries[index] },
            set: { vm.chatTemplateEntries[index] = $0 }
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
                    vm.removeKwarg(id: entry.id)
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(theme.textSecondary)
                        .frame(width: 22, height: 22)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help("Remove kwarg")
            }
            valueRow
        }
    }

    private var typeLabel: String {
        switch entry.kind {
        case .enableThinking:  return "ENABLE_THINKING"
        case .reasoningEffort: return "REASONING_EFFORT"
        case .custom:          return "CUSTOM"
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
                          placeholder: "key", mono: true)
                HStack(spacing: 8) {
                    TextInput(text: vm.bindProfile(binding.value),
                              placeholder: "value", mono: true)
                    forceCheckbox
                }
            }
        }
    }

    private var forceCheckbox: some View {
        Toggle(isOn: vm.bindProfile(binding.force)) {
            Text("Force")
                .font(.omlxText(11))
                .foregroundStyle(theme.textSecondary)
        }
        .toggleStyle(.checkbox)
        .help("Add this key to forced_ct_kwargs so the request body can't override it.")
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
            Row(label: "TurboQuant KV Cache",
                sublabel: "Quantize the KV cache during prefill. Saves memory at a small quality cost.") {
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
                Row(label: "IndexCache",
                    sublabel: "Sparse attention index cache for DSA models. THUDM/IndexCache.") {
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
            Row(label: "SpecPrefill",
                sublabel: "Attention-based sparse prefill for MoE/hybrid models.") {
                Toggle("", isOn: vm.bindProfile($vm.specprefillEnabled))
                    .labelsHidden().toggleStyle(.switch)
            }
            if vm.specprefillEnabled {
                Row(label: "Draft Model",
                    sublabel: "Small model sharing tokenizer with target.") {
                    Popup(
                        selection: vm.bindProfile($vm.specprefillDraftModel),
                        width: 260,
                        options: vm.draftModelOptions()
                    )
                }
                Row(label: "Keep Rate") {
                    Popup(
                        selection: vm.bindProfile($vm.specprefillKeepPct),
                        width: 320,
                        options: ModelSettingsScreenVM.specprefillKeepPctOptions
                    )
                }
                Row(label: "Threshold",
                    sublabel: "Min prompt tokens to trigger (shorter prompts use full prefill).") {
                    TextInput(text: vm.bindProfile($vm.specprefillThreshold),
                              placeholder: "8192", mono: true, suffix: "tk", width: 110)
                }
            }

            // DFlash
            Row(label: "DFlash",
                sublabel: dflashSublabel) {
                Toggle("", isOn: vm.bindProfile($vm.dflashEnabled))
                    .labelsHidden().toggleStyle(.switch)
                    .disabled(!(vm.model?.dflashCompatible ?? true))
                    .help(vm.model?.dflashCompatibilityReason ?? "")
            }
            if vm.dflashEnabled {
                Row(label: "DFlash Draft Model") {
                    Popup(
                        selection: vm.bindProfile($vm.dflashDraftModel),
                        width: 260,
                        options: vm.draftModelOptions()
                    )
                }
                Row(label: "Draft Quantization") {
                    Popup(
                        selection: vm.bindProfile($vm.dflashDraftQuantBits),
                        width: 160,
                        options: ModelSettingsScreenVM.dflashDraftQuantOptions
                    )
                }
                Row(label: "Max Context (fallback)",
                    sublabel: "Prompts at or above this token count switch to BatchedEngine. Empty = unlimited.") {
                    TextInput(text: vm.bindProfile($vm.dflashMaxCtx),
                              placeholder: "unlimited", mono: true, suffix: "tk", width: 130)
                }
                Row(label: "DFlash in-memory cache",
                    sublabel: "DFlash L1 prefix snapshot cache in RAM.") {
                    HStack(spacing: 8) {
                        if vm.dflashInMemoryCache {
                            TextInput(text: vm.bindProfile($vm.dflashInMemoryCacheGib),
                                      placeholder: "8", mono: true, suffix: "GiB", width: 110)
                        }
                        Toggle("", isOn: vm.bindProfile($vm.dflashInMemoryCache))
                            .labelsHidden().toggleStyle(.switch)
                    }
                }
                Row(label: "DFlash SSD cache",
                    sublabel: dflashSsdSublabel) {
                    Toggle("", isOn: vm.bindProfile($vm.dflashSsdCache))
                        .labelsHidden().toggleStyle(.switch)
                        .disabled(!(vm.model?.dflashSsdCacheAvailable ?? false) || !vm.dflashInMemoryCache)
                }
            }

            // Native MTP — last row of the experimental group.
            Row(label: "Native MTP",
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
        return "Block-diffusion speculative decoding. Single-stream only (requests run one at a time)."
    }

    private var dflashSsdSublabel: String {
        if !(vm.model?.dflashSsdCacheAvailable ?? false) {
            return "Enable the global paged SSD cache directory first."
        }
        if !vm.dflashInMemoryCache {
            return "Requires the in-memory cache to be enabled."
        }
        return "L2 spill of evicted L1 entries to disk."
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
        return "Multi-token prediction. Speeds generation when the model supports it."
    }
}

// MARK: - View model

@MainActor
final class ModelSettingsScreenVM: ObservableObject {
    enum Section: String, Hashable, CaseIterable, Sendable {
        case profiles, basic, advanced

        var label: String {
            switch self {
            case .profiles: return "Profiles"
            case .basic:    return "Basic"
            case .advanced: return "Advanced"
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

    static let modelTypeOptions: [(String, String)] = [
        ("",          "Auto-detect"),
        ("llm",       "LLM"),
        ("vlm",       "VLM"),
        ("embed",     "Embedding"),
        ("rerank",    "Reranker"),
        ("audio-stt", "Audio STT"),
        ("audio-tts", "Audio TTS"),
        ("audio-sts", "Audio STS"),
    ]

    static let turboquantKvBitsOptions: [(String, String)] = [
        ("2",   "2-bit"),
        ("2.5", "2.5-bit"),
        ("3",   "3-bit"),
        ("3.5", "3.5-bit"),
        ("4",   "4-bit"),
        ("6",   "6-bit"),
        ("8",   "8-bit"),
    ]

    /// Keep-pct labels mirror the HTML editor's tradeoff annotations
    /// so the user picks an approximate speedup, not a raw fraction.
    static let specprefillKeepPctOptions: [(String, String)] = [
        ("0.1",  "10% — Aggressive (~5-7x, some quality loss)"),
        ("0.2",  "20% — Balanced (~3x, recommended)"),
        ("0.25", "25% — Conservative+ (~2.5x)"),
        ("0.3",  "30% — Conservative (~2.2x)"),
        ("0.4",  "40% — Mild (~1.8x)"),
        ("0.5",  "50% — Minimal (~1.5x)"),
    ]

    static let dflashDraftQuantOptions: [(String, String)] = [
        ("",  "bf16 (default)"),
        ("4", "4-bit"),
        ("8", "8-bit"),
    ]

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
            self.lastError = describe(error)
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
            switch SamplingValidator.penalty(repetitionPenalty, name: "Repetition Penalty") {
            case .success(let v): patch.repetitionPenalty = v
            case .failure(let e): self.lastError = e.message; return
            }
        case .presencePenalty:
            switch SamplingValidator.penalty(presencePenalty, name: "Presence Penalty") {
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
            self.lastError = describe(error)
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
        var out: [(String, String)] = [("", "Select draft model…")]
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
        if dflashEnabled    { return "Disable DFlash before enabling MTP." }
        if turboquantKvEnabled { return "Disable TurboQuant KV before enabling MTP." }
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
            self.lastError = describe(error)
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
            self.lastError = describe(error)
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
            self.lastError = describe(error)
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
            self.lastError = describe(error)
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
            self.lastError = describe(error)
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
            self.lastError = describe(error)
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
            self.lastError = describe(error)
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
            self.lastError = describe(error)
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
            self.lastError = describe(error)
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
            self.lastError = describe(error)
        }
    }

    private func describe(_ error: Error) -> String {
        if let omlx = error as? OMLXClientError { return String(describing: omlx) }
        return error.localizedDescription
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
        parseDouble(raw, label: "Temperature") { v in
            v >= 0 ? nil : "Temperature must be ≥ 0."
        }
    }

    static func topP(_ raw: String) -> Result<Double?, SamplingValidationError> {
        parseDouble(raw, label: "Top P") { v in
            (v > 0 && v <= 1) ? nil : "Top P must be in (0, 1]."
        }
    }

    static func minP(_ raw: String) -> Result<Double?, SamplingValidationError> {
        parseDouble(raw, label: "Min P") { v in
            (v >= 0 && v <= 1) ? nil : "Min P must be in [0, 1]."
        }
    }

    static func topK(_ raw: String) -> Result<Int?, SamplingValidationError> {
        let t = raw.trimmingCharacters(in: .whitespaces)
        if t.isEmpty { return .success(nil) }
        guard let v = Int(t) else {
            return .failure(.init(message: "Top K must be an integer."))
        }
        guard v >= 1 else {
            return .failure(.init(message: "Top K must be a positive integer."))
        }
        return .success(v)
    }

    static func penalty(_ raw: String, name: String) -> Result<Double?, SamplingValidationError> {
        parseDouble(raw, label: name) { v in
            (v >= -2 && v <= 2) ? nil : "\(name) must be in [-2, 2]."
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
            return .failure(.init(message: "\(label) must be a number."))
        }
        if let msg = check(v) { return .failure(.init(message: msg)) }
        return .success(v)
    }
}

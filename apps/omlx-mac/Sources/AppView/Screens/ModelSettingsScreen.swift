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
                    client: services.client
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
    let client: OMLXClient

    @State private var newName: String = ""
    @State private var showCreate: Bool = false

    var body: some View {
        SectionHeader(
            "Per-model Profiles",
            subtitle: "Switch between saved bundles of settings for this model."
        )

        ListGroup {
            FreeRow {
                ProfileChips(
                    profiles: vm.profiles,
                    activeName: vm.activeProfileName,
                    onApply: { name in
                        Task {
                            await vm.applyProfile(name: name, client: client)
                        }
                    },
                    onDelete: { name in
                        Task {
                            await vm.deleteProfile(name: name, client: client)
                        }
                    }
                )
            }
            FreeRow(isLast: true) {
                if showCreate {
                    HStack(spacing: 8) {
                        TextInput(text: $newName, placeholder: "profile name", mono: true)
                            .frame(maxWidth: 220)
                        Button("Create") {
                            let trimmed = newName.trimmingCharacters(in: .whitespaces)
                            guard !trimmed.isEmpty else { return }
                            Task {
                                await vm.createProfile(name: trimmed, client: client)
                                newName = ""
                                showCreate = false
                            }
                        }
                        .buttonStyle(.omlx(.primary, size: .small))
                        Button("Cancel") {
                            newName = ""
                            showCreate = false
                        }
                        .buttonStyle(.omlx(.plain, size: .small))
                    }
                } else {
                    Button {
                        showCreate = true
                    } label: {
                        Label("New profile", systemImage: "plus")
                            .labelStyle(.titleAndIcon)
                    }
                    .buttonStyle(.omlx(.normal, size: .small))
                }
            }
        }

        SectionHeader(
            "Templates",
            subtitle: "Globally-shared bundles. Apply seeds a new per-model profile."
        )

        ListGroup {
            if vm.templates.isEmpty {
                FreeRow(isLast: true) {
                    Text("No templates saved yet.")
                        .font(.omlxText(12))
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .center)
                        .padding(.vertical, 12)
                }
            } else {
                ForEach(Array(vm.templates.enumerated()), id: \.element.id) { idx, tpl in
                    Row(label: tpl.displayName,
                        sublabel: tpl.description,
                        isLast: idx == vm.templates.count - 1) {
                        Button("Apply as profile") {
                            Task {
                                await vm.applyTemplate(template: tpl, client: client)
                            }
                        }
                        .buttonStyle(.omlx(.normal, size: .small))
                    }
                }
            }
        }
    }
}

private struct ProfileChips: View {
    let profiles: [ProfileDTO]
    let activeName: String?
    let onApply: (String) -> Void
    let onDelete: (String) -> Void

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        if profiles.isEmpty {
            Text("No profiles yet.")
                .font(.omlxText(12))
                .foregroundStyle(theme.textTertiary)
                .padding(.vertical, 4)
        } else {
            FlowHStack(spacing: 6) {
                ForEach(profiles) { profile in
                    ChipView(
                        label: profile.displayName,
                        isSelected: profile.name == activeName,
                        onTap: { onApply(profile.name) },
                        onDelete: profile.name == "default"
                            ? nil
                            : { onDelete(profile.name) }
                    )
                }
            }
        }
    }
}

private struct ChipView: View {
    let label: String
    let isSelected: Bool
    let onTap: () -> Void
    let onDelete: (() -> Void)?

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        HStack(spacing: 4) {
            Text(label)
                .font(.omlxText(12, weight: .medium))
                .foregroundStyle(isSelected ? theme.accentText : theme.text)
            if isSelected, let onDelete {
                Button {
                    onDelete()
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundStyle(.white)
                }
                .buttonStyle(.plain)
                .padding(.leading, 2)
            }
        }
        .padding(.horizontal, 10)
        .frame(height: 24)
        .background(
            Capsule()
                .fill(isSelected ? theme.accent : theme.codeBg)
        )
        .overlay(
            Capsule()
                .strokeBorder(isSelected ? Color.clear : theme.inputBorder, lineWidth: 0.5)
        )
        .contentShape(Capsule())
        .onTapGesture { onTap() }
    }
}

/// Minimal flow container — wraps chips when wider than the parent.
private struct FlowHStack<Content: View>: View {
    let spacing: CGFloat
    @ViewBuilder var content: () -> Content

    init(spacing: CGFloat = 6, @ViewBuilder content: @escaping () -> Content) {
        self.spacing = spacing
        self.content = content
    }

    var body: some View {
        FlowLayout(spacing: spacing) {
            content()
        }
    }
}

private struct FlowLayout: Layout {
    let spacing: CGFloat

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? .infinity
        var x: CGFloat = 0, y: CGFloat = 0, rowH: CGFloat = 0
        for sub in subviews {
            let s = sub.sizeThatFits(.unspecified)
            if x + s.width > width && x > 0 {
                x = 0; y += rowH + spacing; rowH = 0
            }
            x += s.width + spacing
            rowH = max(rowH, s.height)
        }
        return CGSize(
            width: min(width, max(x, 0)),
            height: y + rowH
        )
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var x = bounds.minX, y = bounds.minY, rowH: CGFloat = 0
        for sub in subviews {
            let s = sub.sizeThatFits(.unspecified)
            if x + s.width > bounds.maxX && x > bounds.minX {
                x = bounds.minX; y += rowH + spacing; rowH = 0
            }
            sub.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(s))
            x += s.width + spacing
            rowH = max(rowH, s.height)
        }
    }
}

// MARK: - Basic tab

private struct BasicTab: View {
    @ObservedObject var vm: ModelSettingsScreenVM
    let client: OMLXClient

    var body: some View {
        SectionHeader("Basic Settings")

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
                TextInput(text: $vm.contextLength, mono: true, suffix: "tk", width: 110)
                    .onSubmit { Task { await vm.save(.contextLength, client: client) } }
            }
            Row(label: "Max Tokens", sublabel: "Cap on generated tokens (empty = default)") {
                TextInput(text: $vm.maxTokens, placeholder: "Default", mono: true, width: 110)
                    .onSubmit { Task { await vm.save(.maxTokens, client: client) } }
            }
            Row(label: "Temperature") {
                HStack(spacing: 8) {
                    Slider(
                        value: vm.bind($vm.temperature, save: {
                            Task { await vm.save(.temperature, client: client) }
                        }),
                        in: 0...2,
                        step: 0.05
                    )
                    .frame(width: 140)
                    Text(String(format: "%.2f", vm.temperature))
                        .font(.omlxMono(11))
                        .foregroundStyle(.secondary)
                        .frame(width: 36, alignment: .trailing)
                }
            }
            Row(label: "Top P") {
                TextInput(text: $vm.topP, mono: true, width: 90)
                    .onSubmit { Task { await vm.save(.topP, client: client) } }
            }
            Row(label: "Top K") {
                TextInput(text: $vm.topK, mono: true, width: 90)
                    .onSubmit { Task { await vm.save(.topK, client: client) } }
            }
            Row(label: "Min P") {
                TextInput(text: $vm.minP, mono: true, width: 90)
                    .onSubmit { Task { await vm.save(.minP, client: client) } }
            }
            Row(label: "Repetition Penalty") {
                TextInput(text: $vm.repetitionPenalty, mono: true, width: 90)
                    .onSubmit { Task { await vm.save(.repetitionPenalty, client: client) } }
            }
            Row(label: "Presence Penalty") {
                TextInput(text: $vm.presencePenalty, mono: true, width: 90)
                    .onSubmit { Task { await vm.save(.presencePenalty, client: client) } }
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

// MARK: - Advanced tab

private struct AdvancedTab: View {
    @ObservedObject var vm: ModelSettingsScreenVM
    let client: OMLXClient

    var body: some View {
        SectionHeader("Advanced Settings")

        ListGroup {
            Row(label: "Enable Thinking",
                sublabel: "Enable reasoning/thinking mode for this model") {
                Toggle("", isOn: vm.bind($vm.enableThinking, save: {
                    Task { await vm.save(.enableThinking, client: client) }
                }))
                .labelsHidden().toggleStyle(.switch)
            }
            Row(label: "Thinking Budget",
                sublabel: "Limit thinking tokens for reasoning models. Forces end of thinking when exceeded.") {
                HStack(spacing: 8) {
                    if vm.thinkingBudgetEnabled {
                        TextInput(text: $vm.thinkingBudgetTokens,
                                  mono: true, suffix: "tk", width: 110)
                            .onSubmit { Task { await vm.save(.thinkingBudgetTokens, client: client) } }
                    }
                    Toggle("", isOn: vm.bind($vm.thinkingBudgetEnabled, save: {
                        Task { await vm.save(.thinkingBudgetEnabled, client: client) }
                    }))
                    .labelsHidden().toggleStyle(.switch)
                }
            }
            Row(label: "Limit Tool Result Tokens",
                sublabel: "Truncate large tool results (e.g. file reads) to a token limit") {
                Toggle("", isOn: vm.bind($vm.limitToolResults, save: {
                    Task { await vm.save(.limitToolResults, client: client) }
                }))
                .labelsHidden().toggleStyle(.switch)
            }
            Row(label: "Force Sampling",
                sublabel: "Override request sampling parameters with configured values") {
                Toggle("", isOn: vm.bind($vm.forceSampling, save: {
                    Task { await vm.save(.forceSampling, client: client) }
                }))
                .labelsHidden().toggleStyle(.switch)
            }
            Row(label: "Pin in memory",
                sublabel: "Keep this model resident between requests",
                isLast: true) {
                Toggle("", isOn: vm.bind($vm.isPinned, save: {
                    Task { await vm.save(.isPinned, client: client) }
                }))
                .labelsHidden().toggleStyle(.switch)
            }
        }
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
        case limitToolResults, forceSampling, isPinned
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

    @Published var section: Section = .basic

    @Published var model: ModelDTO?
    @Published var modelID: String = ""
    @Published var lastError: String?

    // Basic
    @Published var alias: String = ""
    @Published var modelTypeOverride: String = ""
    @Published var contextLength: String = ""
    @Published var maxTokens: String = ""
    @Published var temperature: Double = 0
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
    @Published var forceSampling: Bool = false
    @Published var isPinned: Bool = false

    // Profiles
    @Published var profiles: [ProfileDTO] = []
    @Published var templates: [ProfileDTO] = []
    @Published var activeProfileName: String?

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

    func load(modelID: String, client: OMLXClient) async {
        self.modelID = modelID
        do {
            let models = try await client.listModels().models
            if let m = models.first(where: { $0.id == modelID }) {
                self.model = m
                if let s = m.settings {
                    self.alias = s.modelAlias ?? ""
                    self.modelTypeOverride = s.modelTypeOverride ?? ""
                    self.contextLength = s.maxContextWindow.map(String.init) ?? ""
                    self.maxTokens = s.maxTokens.map(String.init) ?? ""
                    self.temperature = s.temperature ?? 0.7
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
                    self.forceSampling = s.forceSampling ?? false
                    self.isPinned = s.isPinned ?? false
                    self.activeProfileName = s.activeProfileName ?? "default"
                }
            }
            self.profiles = (try? await client.listModelProfiles(id: modelID).profiles) ?? []
            self.templates = (try? await client.listProfileTemplates().templates) ?? []
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
        case .temperature:             patch.temperature = temperature
        case .topP:                    patch.topP = Double(topP)
        case .topK:                    patch.topK = Int(topK)
        case .minP:                    patch.minP = Double(minP)
        case .repetitionPenalty:       patch.repetitionPenalty = Double(repetitionPenalty)
        case .presencePenalty:         patch.presencePenalty = Double(presencePenalty)
        case .ttl:                     patch.ttlSeconds = Int(ttlSeconds)
        case .enableThinking:          patch.enableThinking = enableThinking
        case .thinkingBudgetEnabled:   patch.thinkingBudgetEnabled = thinkingBudgetEnabled
        case .thinkingBudgetTokens:    patch.thinkingBudgetTokens = Int(thinkingBudgetTokens)
        case .limitToolResults:        patch.maxToolResultTokens = limitToolResults ? 4096 : 0
        case .forceSampling:           patch.forceSampling = forceSampling
        case .isPinned:                patch.isPinned = isPinned
        }
        do {
            _ = try await client.updateModelSettings(id: modelID, patch: patch)
            self.lastError = nil
        } catch {
            self.lastError = describe(error)
        }
    }

    // MARK: - Profile actions

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
}

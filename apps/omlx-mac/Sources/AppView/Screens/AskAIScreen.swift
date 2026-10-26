// Ask AI settings. Everything here is app-side state (`ask-ai.json` under
// Application Support), so edits apply at once without an Apply button.
// Sections:
//   - Ask AI - feature switch, global hotkey, conversation reuse window,
//     GPU power limit.
//   - Default - model and prompt for questions without a slash command.
//   - Presets - `/command` entries with their own model and system prompt.

import SwiftUI

struct AskAIScreen: View {
    @Environment(AppServices.self) private var services
    /// Nil until the model list loads, so saved models are not marked as
    /// missing while the server is still starting.
    @State private var models: [String]?
    @State private var modelsUnavailable = false
    /// Nil until the server answers; every model reports the same host value.
    @State private var gpuPowerLimitSupported: Bool?

    var body: some View {
        let askAI = services.askAI
        VStack(alignment: .leading, spacing: 0) {
            GeneralSection(controller: askAI, store: askAI.store,
                           gpuPowerLimitSupported: gpuPowerLimitSupported)
            if askAI.store.prefs.enabled {
                DefaultSection(store: askAI.store, models: models)
                PresetsSection(store: askAI.store, models: models)
                if modelsUnavailable {
                    HintLine(text: String(
                        localized: "askai.models.unavailable",
                        defaultValue: "Start the server to pick from installed models.",
                        comment: "Hint when the model list could not be loaded"
                    ))
                    .padding(.horizontal, 28)
                    .padding(.top, 6)
                }
            }
        }
        // Reload when the server comes up after the screen opened.
        .task(id: [askAI.store.prefs.enabled, services.serverState.isRunningLike]) {
            guard askAI.store.prefs.enabled else { return }
            await loadModels()
        }
    }

    /// Chat-capable models only; embedding, audio, and virtual entries
    /// cannot answer a chat request.
    private func loadModels() async {
        do {
            let all = try await services.client.listModels().models
            gpuPowerLimitSupported = all.first?.gpuPowerLimitSupported
            models = all.filter { model in
                let type = model.modelType ?? ""
                return (type.isEmpty || type == "llm" || type == "vlm") && model.virtual != true
            }.map(\.id)
            modelsUnavailable = false
        } catch {
            models = nil
            modelsUnavailable = true
        }
    }

    /// Popup options with a placeholder first. A saved model missing from
    /// the list stays selectable so the value is not lost. It is marked as
    /// not installed only when the list actually loaded.
    static func modelOptions(_ models: [String]?, current: String) -> [(String, String)] {
        let names = models ?? []
        var out: [(String, String)] = [
            ("", String(localized: "askai.model.select_placeholder",
                        defaultValue: "Select model…",
                        comment: "Placeholder option at the top of the Ask AI model picker")),
        ]
        if !current.isEmpty, !names.contains(current) {
            out.append((current, models != nil
                ? String(localized: "askai.model.missing",
                         defaultValue: "\(current) (not installed)",
                         comment: "Model picker option for a saved model that is no longer found")
                : current))
        }
        out += names.map { ($0, $0) }
        return out
    }
}

// MARK: - Ask AI

private struct GeneralSection: View {
    let controller: AskAIController
    @Bindable var store: AskAIStore
    let gpuPowerLimitSupported: Bool?
    @State private var gpuPowerLimitText = ""

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        SectionHeader(String(localized: "askai.section.general",
                             defaultValue: "Ask AI",
                             comment: "Section header for the Ask AI switch and hotkey"))

        ListGroup {
            Row(
                label: String(localized: "askai.row.enabled",
                              defaultValue: "Enable Ask AI",
                              comment: "Row label for the Ask AI feature switch"),
                sublabel: String(localized: "askai.row.enabled.sub",
                                 defaultValue: "Open a question bar from any app with a global shortcut.",
                                 comment: "Sublabel for the Ask AI feature switch"),
                isLast: !store.prefs.enabled
            ) {
                RowSwitch(isOn: $store.prefs.enabled)
            }
            if store.prefs.enabled {
                Row(
                    label: String(localized: "askai.row.hotkey",
                                  defaultValue: "Hotkey",
                                  comment: "Row label for the Ask AI global shortcut"),
                    sublabel: String(localized: "askai.row.hotkey.sub",
                                     defaultValue: "Press it again or Esc to close the bar.",
                                     comment: "Sublabel for the Ask AI global shortcut")
                ) {
                    HotKeyRecorder(combo: $store.prefs.hotKey) { recording in
                        controller.setHotKeySuspended(recording)
                    }
                }
                if let error = controller.hotKeyError {
                    FreeRow {
                        Label(error, systemImage: "exclamationmark.triangle.fill")
                            .font(.omlxText(DesignTokens.FontSize.callout))
                            .foregroundStyle(theme.redDot)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                Row(
                    label: String(localized: "askai.row.new_conversation",
                                  defaultValue: "Start New Conversation After",
                                  comment: "Row label for the idle time that resets the Ask AI conversation"),
                    sublabel: String(localized: "askai.row.new_conversation.sub",
                                     defaultValue: "Reopening the bar sooner keeps the last conversation for follow-up questions.",
                                     comment: "Sublabel for the Ask AI conversation reset time")
                ) {
                    Popup(selection: $store.prefs.newConversationAfter, width: .controlCompact, options: [
                        (0, String(localized: "askai.new_conversation.always", defaultValue: "Every Time",
                                   comment: "Ask AI option: always start a new conversation")),
                        (120, String(localized: "askai.new_conversation.2m", defaultValue: "2 Minutes",
                                     comment: "Ask AI option: new conversation after 2 idle minutes")),
                        (300, String(localized: "askai.new_conversation.5m", defaultValue: "5 Minutes",
                                     comment: "Ask AI option: new conversation after 5 idle minutes")),
                        (900, String(localized: "askai.new_conversation.15m", defaultValue: "15 Minutes",
                                     comment: "Ask AI option: new conversation after 15 idle minutes")),
                    ])
                }
                Row(
                    label: String(localized: "askai.row.gpu_power_limit",
                                  defaultValue: "GPU Power Limit",
                                  comment: "Row label for the GPU power limit while Ask AI answers"),
                    sublabel: gpuPowerLimitSupported == false
                        ? GPUPowerGuide.unsupportedText
                        : String(localized: "askai.row.gpu_power_limit.sub",
                                 defaultValue: "Caps GPU power for the whole Mac while Ask AI answers, to save battery (2 to 1000 W). If the model also has a limit, the lower one applies.",
                                 comment: "Sublabel for the Ask AI GPU power limit"),
                    isLast: true
                ) {
                    HStack(spacing: 8) {
                        if store.prefs.gpuPowerLimitEnabled {
                            // Text state so a cleared field shows the range placeholder.
                            TextInput(text: $gpuPowerLimitText,
                                      placeholder: "2-1000", mono: true, isNumeric: true,
                                      range: 2...1000, step: 1, suffix: "W",
                                      width: .controlCompact)
                                .onAppear { gpuPowerLimitText = String(store.prefs.gpuPowerLimitWatts) }
                                .onChange(of: gpuPowerLimitText) { _, text in
                                    if let watts = Int(text) { store.prefs.gpuPowerLimitWatts = watts }
                                }
                        }
                        GPUPowerGuideButton()
                        RowSwitch(isOn: $store.prefs.gpuPowerLimitEnabled)
                            .disabled(gpuPowerLimitSupported == false && !store.prefs.gpuPowerLimitEnabled)
                    }
                }
            }
        }
    }
}

// MARK: - Default

private struct DefaultSection: View {
    @Bindable var store: AskAIStore
    let models: [String]?

    var body: some View {
        SectionHeader(
            String(localized: "askai.section.default",
                   defaultValue: "Default",
                   comment: "Section header for the Ask AI default model and prompt"),
            subtitle: String(localized: "askai.section.default.sub",
                             defaultValue: "Used when a question does not start with a preset command.",
                             comment: "Subtitle for the Ask AI default section")
        )

        ListGroup {
            Row(label: AskAIScreenStrings.model) {
                Popup(selection: $store.prefs.defaultModel, width: .controlWide,
                      options: AskAIScreen.modelOptions(models, current: store.prefs.defaultModel))
            }
            Row(label: AskAIScreenStrings.thinking, sublabel: AskAIScreenStrings.thinkingSub) {
                RowSwitch(isOn: $store.prefs.defaultAllowThinking)
            }
            Row(label: AskAIScreenStrings.autoCopy, sublabel: AskAIScreenStrings.autoCopySub) {
                RowSwitch(isOn: $store.prefs.defaultAutoCopy)
            }
            FreeRow(isLast: true) {
                PromptField(text: $store.prefs.defaultSystemPrompt,
                            placeholder: String(localized: "askai.prompt.default_placeholder",
                                                defaultValue: "Optional. For example: Answer briefly.",
                                                comment: "Placeholder for the Ask AI default system prompt"))
            }
        }
    }
}

// MARK: - Presets

private struct PresetsSection: View {
    @Bindable var store: AskAIStore
    let models: [String]?

    /// Id of the preset being edited; not in the list yet for a new one.
    /// The editor binds to the non-optional `draft`, so clearing this id
    /// never leaves a binding pointing at nil.
    @State private var editingID: UUID?
    @State private var draft = AskAIPreset()
    @State private var editError: String?
    @State private var pendingDelete: AskAIPreset?

    @Environment(\.omlxTheme) private var theme

    private var isCreating: Bool {
        guard let editingID else { return false }
        return !store.prefs.presets.contains { $0.id == editingID }
    }

    var body: some View {
        SectionHeader(
            String(localized: "askai.section.presets",
                   defaultValue: "Presets",
                   comment: "Section header for the Ask AI slash-command presets"),
            subtitle: String(localized: "askai.section.presets.sub",
                             defaultValue: "Type /command in the Ask AI bar to use a preset's model and system prompt.",
                             comment: "Subtitle for the Ask AI presets section")
        ) {
            Button {
                editError = nil
                draft = AskAIPreset()
                editingID = draft.id
            } label: {
                Label(String(localized: "askai.presets.new",
                             defaultValue: "New",
                             comment: "Button that adds an Ask AI preset"),
                      systemImage: "plus")
                    .labelStyle(.titleAndIcon)
            }
            .buttonStyle(.omlx(.normal, size: .small))
            .disabled(editingID != nil)
        }

        ListGroup {
            if isCreating {
                editor(isLast: store.prefs.presets.isEmpty)
            }
            if store.prefs.presets.isEmpty && !isCreating {
                FreeRow(isLast: true) {
                    Text(String(localized: "askai.presets.empty",
                                defaultValue: "No presets yet.",
                                comment: "Empty state of the Ask AI presets list"))
                        .font(.omlxText(DesignTokens.FontSize.callout))
                        .foregroundStyle(theme.textTertiary)
                        .frame(maxWidth: .infinity, alignment: .center)
                        .padding(.vertical, 14)
                }
            }
            ForEach(Array(store.prefs.presets.enumerated()), id: \.element.id) { index, preset in
                let isLast = index == store.prefs.presets.count - 1
                if editingID == preset.id {
                    editor(isLast: isLast)
                } else {
                    PresetRow(preset: preset, isLast: isLast, isLocked: editingID != nil) {
                        editError = nil
                        draft = preset
                        editingID = preset.id
                    } onDelete: {
                        pendingDelete = preset
                    }
                }
            }
        }
        .confirmationDialog(
            deleteTitle,
            isPresented: Binding(get: { pendingDelete != nil }, set: { if !$0 { pendingDelete = nil } })
        ) {
            Button(String(localized: "askai.presets.delete.confirm",
                          defaultValue: "Delete",
                          comment: "Confirm button for deleting an Ask AI preset"),
                   role: .destructive) {
                if let id = pendingDelete?.id {
                    store.prefs.presets.removeAll { $0.id == id }
                }
                pendingDelete = nil
            }
        }
    }

    private var deleteTitle: String {
        let command = pendingDelete?.command ?? ""
        return String(localized: "askai.presets.delete.title",
                      defaultValue: "Delete /\(command)?",
                      comment: "Confirmation title before deleting an Ask AI preset; the argument is the command")
    }

    private func editor(isLast: Bool) -> some View {
        PresetEditor(preset: $draft, models: models, error: editError, isLast: isLast) {
            editingID = nil
            editError = nil
        } onSave: {
            save()
        }
    }

    private func save() {
        var preset = draft
        var command = preset.command.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if command.hasPrefix("/") { command.removeFirst() }
        preset.command = command
        if let problem = AskAIPresetValidation.check(preset, among: store.prefs.presets) {
            draft = preset
            editError = problem.message
            return
        }
        if let index = store.prefs.presets.firstIndex(where: { $0.id == preset.id }) {
            store.prefs.presets[index] = preset
        } else {
            store.prefs.presets.append(preset)
        }
        editingID = nil
        editError = nil
    }
}

private struct PresetRow: View {
    let preset: AskAIPreset
    let isLast: Bool
    let isLocked: Bool
    let onEdit: () -> Void
    let onDelete: () -> Void

    @Environment(\.omlxTheme) private var theme

    private var detail: String {
        var parts = [preset.model.isEmpty ? AskAIScreenStrings.noModel : preset.model]
        if preset.allowThinking { parts.append(AskAIScreenStrings.thinking) }
        if preset.autoCopy { parts.append(AskAIScreenStrings.autoCopy) }
        return parts.joined(separator: " · ")
    }

    var body: some View {
        FreeRow(isLast: isLast) {
            HStack(spacing: 10) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("/\(preset.command)")
                        .font(.omlxMono(DesignTokens.FontSize.body, weight: .medium))
                        .foregroundStyle(theme.text)
                    Text(detail)
                        .font(.omlxText(DesignTokens.FontSize.subheadline))
                        .foregroundStyle(theme.textSecondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    if !preset.promptPreview.isEmpty {
                        Text(preset.promptPreview)
                            .font(.omlxText(DesignTokens.FontSize.subheadline))
                            .foregroundStyle(theme.textTertiary)
                            .lineLimit(1)
                    }
                }
                Spacer(minLength: 8)
                Button(action: onEdit) {
                    Image(systemName: "pencil")
                        .font(.system(size: 12))
                }
                .buttonStyle(.omlx(.plain, size: .small))
                .help(String(localized: "askai.presets.edit",
                             defaultValue: "Edit",
                             comment: "Tooltip on the edit button of an Ask AI preset"))
                Button(action: onDelete) {
                    Image(systemName: "trash")
                        .font(.system(size: 12))
                        .foregroundStyle(theme.redDot)
                }
                .buttonStyle(.omlx(.plain, size: .small))
                .help(String(localized: "askai.presets.delete",
                             defaultValue: "Delete",
                             comment: "Tooltip on the delete button of an Ask AI preset"))
            }
            .disabled(isLocked)
        }
    }
}

private struct PresetEditor: View {
    @Binding var preset: AskAIPreset
    let models: [String]?
    let error: String?
    let isLast: Bool
    let onCancel: () -> Void
    let onSave: () -> Void

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        Row(label: String(localized: "askai.preset.command",
                          defaultValue: "Command",
                          comment: "Row label for an Ask AI preset command name"),
            sublabel: String(localized: "askai.preset.command.sub",
                             defaultValue: "Letters in any language, digits, - and _. No spaces.",
                             comment: "Sublabel for an Ask AI preset command name")) {
            HStack(spacing: 4) {
                Text(verbatim: "/")
                    .font(.omlxMono(DesignTokens.FontSize.body, weight: .medium))
                    .foregroundStyle(theme.textSecondary)
                TextInput(text: $preset.command,
                          placeholder: String(localized: "askai.preset.command.placeholder",
                                              defaultValue: "translate",
                                              comment: "Placeholder for an Ask AI preset command name"),
                          mono: true, width: .controlMedium)
            }
        }
        Row(label: AskAIScreenStrings.model) {
            Popup(selection: $preset.model, width: .controlWide,
                  options: AskAIScreen.modelOptions(models, current: preset.model))
        }
        Row(label: AskAIScreenStrings.thinking, sublabel: AskAIScreenStrings.thinkingSub) {
            RowSwitch(isOn: $preset.allowThinking)
        }
        Row(label: AskAIScreenStrings.autoCopy, sublabel: AskAIScreenStrings.autoCopySub) {
            RowSwitch(isOn: $preset.autoCopy)
        }
        FreeRow {
            PromptField(text: $preset.systemPrompt,
                        placeholder: String(localized: "askai.prompt.preset_placeholder",
                                            defaultValue: "For example: Translate the text into English in a concise, professional Slack tone. Output only the translation.",
                                            comment: "Placeholder for an Ask AI preset system prompt"))
        }
        FreeRow(isLast: isLast) {
            HStack(spacing: 6) {
                if let error {
                    Text(error)
                        .font(.omlxText(DesignTokens.FontSize.callout))
                        .foregroundStyle(theme.redDot)
                }
                Spacer()
                Button(String(localized: "common.cancel",
                              defaultValue: "Cancel",
                              comment: "Common cancel button label"), action: onCancel)
                    .buttonStyle(.omlx(.plain, size: .small))
                Button(String(localized: "common.save",
                              defaultValue: "Save",
                              comment: "Common save button label"), action: onSave)
                    .buttonStyle(.omlx(.primary, size: .small))
            }
        }
    }
}

/// System prompt editor: label above a multi-line field.
private struct PromptField: View {
    @Binding var text: String
    let placeholder: String

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(String(localized: "askai.row.system_prompt",
                        defaultValue: "System Prompt",
                        comment: "Label above the Ask AI system prompt editor"))
                .font(.omlxText(DesignTokens.FontSize.body, weight: .medium))
                .foregroundStyle(theme.text)
            TextEditor(text: $text)
                .font(.omlxText(DesignTokens.FontSize.callout))
                .scrollContentBackground(.hidden)
                .frame(height: 96)
                .padding(6)
                .background(theme.inputBg)
                .overlay(alignment: .topLeading) {
                    if text.isEmpty {
                        Text(placeholder)
                            .font(.omlxText(DesignTokens.FontSize.callout))
                            .foregroundStyle(theme.textTertiary)
                            .padding(.horizontal, 11)
                            .padding(.vertical, 6)
                            .allowsHitTesting(false)
                    }
                }
                .overlay(
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .strokeBorder(theme.inputBorder, lineWidth: 0.5)
                )
        }
    }
}

private enum AskAIScreenStrings {
    static var model: String {
        String(localized: "askai.row.model", defaultValue: "Model",
               comment: "Row label for the Ask AI model picker")
    }
    static var noModel: String {
        String(localized: "askai.presets.no_model", defaultValue: "No model",
               comment: "Shown in a preset row when no model is chosen")
    }
    static var thinking: String {
        String(localized: "askai.row.thinking", defaultValue: "Thinking",
               comment: "Row label for the Ask AI thinking switch")
    }
    static var thinkingSub: String {
        String(localized: "askai.row.thinking.sub",
               defaultValue: "Off asks reasoning models to answer without a thinking pass.",
               comment: "Sublabel for the Ask AI thinking switch")
    }
    static var autoCopy: String {
        String(localized: "askai.row.auto_copy", defaultValue: "Auto Copy",
               comment: "Row label for the Ask AI auto copy switch")
    }
    static var autoCopySub: String {
        String(localized: "askai.row.auto_copy.sub",
               defaultValue: "Copy the answer to the clipboard when it finishes.",
               comment: "Sublabel for the Ask AI auto copy switch")
    }
}

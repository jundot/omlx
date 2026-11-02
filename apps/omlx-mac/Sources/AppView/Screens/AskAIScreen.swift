// Ask AI settings. Everything here is app-side state (`ask-ai.json` under
// Application Support), so edits apply at once without an Apply button.
// Sections:
//   - Ask AI - switches for asking by text and by voice.
//   - Ask via Text - hotkey, model, GPU power limit; presets and advanced
//     options fold below.
//   - Ask via Voice - hotkey, voice model, speech speed; voice and system
//     prompt fold below.

import SwiftUI

/// Sub-page of the Ask AI screen; nil is its top page.
enum AskAIPage: Hashable, Sendable {
    case presets, textAdvanced, voiceAdvanced

    var title: String {
        switch self {
        case .presets: AskAIScreenStrings.presets
        case .textAdvanced, .voiceAdvanced: AskAIScreenStrings.advanced
        }
    }

    /// The section the page belongs to, shown under its title.
    var sectionTitle: String {
        switch self {
        case .presets, .textAdvanced: AskAIScreenStrings.textSection
        case .voiceAdvanced: AskAIScreenStrings.voiceSection
        }
    }
}

struct AskAIScreen: View {
    @Environment(AppServices.self) private var services
    @Environment(\.omlxTheme) private var theme
    /// Nil until the model list loads, so saved models are not marked as
    /// missing while the server is still starting.
    @State private var models: [String]?
    /// Models with a realtime voice backend, for Ask via Voice.
    @State private var voiceModels: [String]?
    /// Models that read images, for the screen capture tool.
    @State private var visionModels: Set<String>?
    @State private var modelsUnavailable = false
    /// Nil until the server answers; every model reports the same host value.
    @State private var gpuPowerLimitSupported: Bool?
    @State private var microphone = MicrophoneAccess.state
    @State private var showsMicrophoneGuide = false

    var body: some View {
        let askAI = services.askAI
        let prefs = askAI.store.prefs
        VStack(alignment: .leading, spacing: 0) {
            switch services.askAIPage {
            case nil:
                ModesSection(store: askAI.store, microphone: microphone,
                             showsMicrophoneGuide: $showsMicrophoneGuide)
                if prefs.enabled {
                    TextSection(controller: askAI, store: askAI.store, models: models,
                                gpuPowerLimitSupported: gpuPowerLimitSupported)
                }
                if prefs.talk.enabled {
                    VoiceSection(controller: askAI, store: askAI.store, models: voiceModels,
                                 gpuPowerLimitSupported: gpuPowerLimitSupported,
                                 microphone: microphone, showsMicrophoneGuide: $showsMicrophoneGuide)
                }
                if modelsUnavailable, prefs.enabled || prefs.talk.enabled {
                    HintLine(text: String(
                        localized: "askai.models.unavailable",
                        defaultValue: "Start the server to pick from installed models.",
                        comment: "Hint when the model list could not be loaded"
                    ))
                    .padding(.horizontal, 28)
                    .padding(.top, 6)
                }
            case .presets:
                PresetsSection(store: askAI.store, models: models, visionModels: visionModels)
            case .textAdvanced:
                TextAdvancedSection(store: askAI.store, visionModels: visionModels)
            case .voiceAdvanced:
                VoiceAdvancedSection(store: askAI.store)
            }
        }
        .toolbar {
            if services.askAIPage != nil {
                ToolbarItem(placement: .navigation) {
                    Button {
                        services.askAIPage = nil
                    } label: {
                        Label(String(localized: "askai.page.back",
                                     defaultValue: "Back to Ask AI",
                                     comment: "Back button label on an Ask AI settings sub-page"),
                              systemImage: "chevron.left")
                            .labelStyle(.iconOnly)
                    }
                }
            }
        }
        .sheet(isPresented: $showsMicrophoneGuide) {
            MicrophoneGuideSheet(microphone: $microphone)
                .environment(\.omlxTheme, theme)
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            microphone = MicrophoneAccess.state
        }
        // Reload when the server comes up after the screen opened.
        .task(id: [prefs.enabled, prefs.talk.enabled, services.serverState.isRunningLike]) {
            guard prefs.enabled || prefs.talk.enabled else { return }
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
            voiceModels = all.filter { $0.realtimeVoice == true && $0.virtual != true }.map(\.id)
            visionModels = Set(all.filter { $0.modelType == "vlm" }.map(\.id))
            modelsUnavailable = false
        } catch {
            models = nil
            voiceModels = nil
            visionModels = nil
            modelsUnavailable = true
        }
    }

    /// False only when the list loaded and `model` is not a vision model.
    static func readsImages(_ model: String, in visionModels: Set<String>?) -> Bool {
        guard let visionModels, !model.isEmpty else { return true }
        return visionModels.contains(model)
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

private struct ModesSection: View {
    @Bindable var store: AskAIStore
    let microphone: MicrophoneAccess.State
    @Binding var showsMicrophoneGuide: Bool

    private var voiceEnabled: Binding<Bool> {
        Binding {
            store.prefs.talk.enabled
        } set: { on in
            store.prefs.talk.enabled = on
            if on, microphone != .authorized { showsMicrophoneGuide = true }
        }
    }

    var body: some View {
        SectionHeader(String(localized: "askai.section.general",
                             defaultValue: "Ask AI",
                             comment: "Section header for the Ask AI text and voice switches"))

        ListGroup {
            Row(
                label: String(localized: "askai.row.enabled",
                              defaultValue: "Enable Text",
                              comment: "Row label for the switch that enables asking Ask AI by text"),
                sublabel: String(localized: "askai.row.enabled.sub",
                                 defaultValue: "Type a question from any app with a global shortcut.",
                                 comment: "Sublabel for the Ask AI text switch")
            ) {
                RowSwitch(isOn: $store.prefs.enabled)
            }
            Row(
                label: String(localized: "talkai.row.enabled",
                              defaultValue: "Enable Voice",
                              comment: "Row label for the switch that enables asking Ask AI by voice"),
                sublabel: String(localized: "talkai.row.enabled.sub",
                                 defaultValue: "Talk with a voice model from any app with a global shortcut.",
                                 comment: "Sublabel for the Ask AI voice switch"),
                isLast: true
            ) {
                RowSwitch(isOn: voiceEnabled)
            }
        }
    }
}

// MARK: - Ask via Text

private struct TextSection: View {
    let controller: AskAIController
    @Bindable var store: AskAIStore
    let models: [String]?
    let gpuPowerLimitSupported: Bool?

    @Environment(AppServices.self) private var services
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        SectionHeader(AskAIScreenStrings.textSection)

        ListGroup {
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
                label: AskAIScreenStrings.model,
                sublabel: String(localized: "askai.row.model.sub",
                                 defaultValue: "Used when a question does not start with a preset command.",
                                 comment: "Sublabel for the Ask AI default model picker")
            ) {
                Popup(selection: $store.prefs.defaultModel, width: .controlWide,
                      options: AskAIScreen.modelOptions(models, current: store.prefs.defaultModel))
            }
            GPUPowerLimitRow(
                isOn: $store.prefs.gpuPowerLimitEnabled, watts: $store.prefs.gpuPowerLimitWatts,
                supported: gpuPowerLimitSupported,
                sublabel: String(localized: "askai.row.gpu_power_limit.sub",
                                 defaultValue: "Caps GPU power for the whole Mac while Ask AI answers, to save battery (2 to 1000 W). If the model also has a limit, the lower one applies.",
                                 comment: "Sublabel for the Ask AI GPU power limit")
            )
        }

        // System Settings style: rows that open a page.
        ListGroup {
            NavRow(label: AskAIScreenStrings.presets,
                   value: store.prefs.presets.isEmpty ? nil : String(store.prefs.presets.count),
                   action: { services.askAIPage = .presets }) {
                Squircle(systemSymbol: "command", gradient: SquircleGradient.models)
            }
            NavRow(label: AskAIScreenStrings.advanced, isLast: true,
                   action: { services.askAIPage = .textAdvanced }) {
                Squircle(systemSymbol: "gearshape.fill", gradient: SquircleGradient.server)
            }
        }
        .padding(.top, 14)
    }
}

private struct TextAdvancedSection: View {
    @Bindable var store: AskAIStore
    let visionModels: Set<String>?

    var body: some View {
        Group {
            ListGroup {
                Row(
                    label: String(localized: "askai.row.send_on_enter",
                                  defaultValue: "Send on Enter",
                                  comment: "Row label for the Ask AI Enter-to-send switch"),
                    sublabel: String(localized: "askai.row.send_on_enter.sub",
                                     defaultValue: "Enter sends the question and ⇧↩ adds a new line. ⌘↩ always sends.",
                                     comment: "Sublabel for the Ask AI Enter-to-send switch")
                ) {
                    RowSwitch(isOn: $store.prefs.sendOnEnter)
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
                                                    comment: "Placeholder for the Ask AI default system prompt"),
                                defaultText: AskAIPrefs.defaultPrompt)
                }
            }
            ListGroup {
                ToolRows(tools: $store.prefs.defaultTools,
                         modelReadsImages: AskAIScreen.readsImages(store.prefs.defaultModel, in: visionModels),
                         isLast: true)
            }
            .padding(.top, 14)
        }
    }
}

/// The tool calling switch and the tools it offers.
private struct ToolRows: View {
    @Binding var tools: AskAIToolPrefs
    let modelReadsImages: Bool
    var isLast = false

    @State private var screenGranted = ScreenCaptureAccess.isGranted
    @State private var showsScreenGuide = false
    @Environment(AppServices.self) private var services
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        Row(label: AskAIScreenStrings.toolCalling,
            sublabel: String(localized: "askai.tools.row.sub",
                             defaultValue: "Lets the model search the web, read links and look at your screen when it needs to.",
                             comment: "Sublabel for the Ask AI tool calling switch"),
            isLast: isLast && !tools.enabled) {
            RowSwitch(isOn: $tools.enabled)
        }
        if tools.enabled {
            FreeRow(isLast: isLast) {
                VStack(alignment: .leading, spacing: 10) {
                    toolToggle($tools.webSearch,
                               title: String(localized: "askai.tools.web_search", defaultValue: "Web Search",
                                             comment: "Checkbox for the Ask AI web search tool"),
                               detail: String(localized: "askai.tools.web_search.sub",
                                              defaultValue: "Uses the search provider set in Dashboard Settings > Integrations.",
                                              comment: "Description of the Ask AI web search tool"))
                    toolToggle($tools.readLinks,
                               title: String(localized: "askai.tools.read_links", defaultValue: "Read Links",
                                             comment: "Checkbox for the Ask AI tool that reads web pages"),
                               detail: String(localized: "askai.tools.read_links.sub",
                                              defaultValue: "Reads web pages, including links you put in a question.",
                                              comment: "Description of the Ask AI read links tool"))
                    toolToggle($tools.screenCapture,
                               title: String(localized: "askai.tools.screen", defaultValue: "Screen Capture",
                                             comment: "Checkbox for the Ask AI screen capture tool"),
                               detail: modelReadsImages
                                   ? String(localized: "askai.tools.screen.sub",
                                            defaultValue: "Takes a screenshot when you ask about something on your screen. oMLX windows are left out.",
                                            comment: "Description of the Ask AI screen capture tool")
                                   : String(localized: "askai.tools.screen.needs_vision",
                                            defaultValue: "Needs a vision (VLM) model.",
                                            comment: "Shown under the screen capture tool when the model cannot read images"))
                        .disabled(!modelReadsImages)
                    if tools.screenCapture && modelReadsImages && !screenGranted {
                        HStack(spacing: 10) {
                            Label(String(localized: "askai.tools.screen.warning",
                                         defaultValue: "Screen recording permission is off, so Ask AI can't see your screen.",
                                         comment: "Warning shown while oMLX has no screen recording permission"),
                                  systemImage: "exclamationmark.triangle.fill")
                                .font(.omlxText(DesignTokens.FontSize.callout))
                                .foregroundStyle(theme.amberDot)
                                .fixedSize(horizontal: false, vertical: true)
                            Spacer()
                            Button(AskAIScreenStrings.restartApp) {
                                services.relaunchApp()
                            }
                            .buttonStyle(.omlx(.normal, size: .small))
                            Button(String(localized: "askai.tools.screen.allow", defaultValue: "Allow…",
                                          comment: "Button that opens the screen recording permission guide")) {
                                showsScreenGuide = true
                            }
                            .buttonStyle(.omlx(.normal, size: .small))
                        }
                    }
                }
                .sheet(isPresented: $showsScreenGuide) {
                    ScreenCaptureGuideSheet()
                        .environment(\.omlxTheme, theme)
                }
                .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
                    screenGranted = ScreenCaptureAccess.isGranted
                }
            }
        }
    }

    private func toolToggle(_ isOn: Binding<Bool>, title: String, detail: String) -> some View {
        Toggle(isOn: isOn) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.omlxText(DesignTokens.FontSize.body, weight: .medium))
                    .foregroundStyle(theme.text)
                Text(detail)
                    .font(.omlxText(11.5))
                    .foregroundStyle(theme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .toggleStyle(.checkbox)
    }
}

/// macOS applies a new screen recording permission only after oMLX restarts.
/// The sheet closes before System Settings opens, because an open sheet
/// blocks its Quit & Reopen.
private struct ScreenCaptureGuideSheet: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(String(localized: "askai.screen.guide.title",
                        defaultValue: "Allow Screen Recording",
                        comment: "Title of the screen recording permission guide"))
                .font(.omlxText(DesignTokens.FontSize.title3, weight: .semibold))
                .foregroundStyle(theme.text)
            step(1, String(localized: "askai.screen.guide.step1",
                           defaultValue: "Click Open System Settings, then turn on oMLX in Screen & System Audio Recording.",
                           comment: "First step of the screen recording permission guide"))
            step(2, String(localized: "askai.screen.guide.step2",
                           defaultValue: "The permission applies after oMLX restarts. Choose Quit & Reopen when macOS asks, or click Restart oMLX next to the warning.",
                           comment: "Second step of the screen recording permission guide"))
            HStack {
                Spacer()
                Button(String(localized: "askai.screen.guide.later",
                              defaultValue: "Not Now",
                              comment: "Button that closes the screen recording permission guide")) {
                    dismiss()
                }
                .buttonStyle(.omlx(.normal, size: .small))
                Button(String(localized: "askai.screen.guide.open_settings",
                              defaultValue: "Open System Settings",
                              comment: "Button that opens the macOS screen recording privacy settings")) {
                    // Only the first request shows the macOS prompt; it also lists oMLX in System Settings.
                    ScreenCaptureAccess.request()
                    ScreenCaptureAccess.openSystemSettings()
                    dismiss()
                }
                .buttonStyle(.omlx(.primary, size: .small))
            }
            .padding(.top, 6)
        }
        .padding(22)
        .frame(width: 420)
    }

    private func step(_ number: Int, _ text: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Text("\(number)")
                .font(.omlxText(DesignTokens.FontSize.callout, weight: .semibold))
                .foregroundStyle(theme.accentText)
                .frame(width: 20, height: 20)
                .background(Circle().fill(theme.accent))
            Text(text)
                .font(.omlxText(DesignTokens.FontSize.body))
                .foregroundStyle(theme.text)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

/// GPU power cap switch with its watt field. The server applies the lower
/// of this and the model's own limit.
private struct GPUPowerLimitRow: View {
    @Binding var isOn: Bool
    @Binding var watts: Int
    let supported: Bool?
    let sublabel: String

    /// Text state so a cleared field shows the range placeholder.
    @State private var text = ""

    var body: some View {
        Row(
            label: String(localized: "askai.row.gpu_power_limit",
                          defaultValue: "GPU Power Limit",
                          comment: "Row label for the GPU power limit while Ask AI answers"),
            sublabel: supported == false ? GPUPowerGuide.unsupportedText : sublabel,
            isLast: true
        ) {
            HStack(spacing: 8) {
                if isOn {
                    TextInput(text: $text,
                              placeholder: "2-1000", mono: true, isNumeric: true,
                              range: 2...1000, step: 1, suffix: "W",
                              width: .controlCompact)
                        .onAppear { text = String(watts) }
                        .onChange(of: text) { _, text in
                            if let value = Int(text) { watts = value }
                        }
                }
                GPUPowerGuideButton()
                RowSwitch(isOn: $isOn)
                    .disabled(supported == false && !isOn)
            }
        }
    }
}

// MARK: - Ask via Voice

private struct VoiceSection: View {
    let controller: AskAIController
    @Bindable var store: AskAIStore
    let models: [String]?
    let gpuPowerLimitSupported: Bool?
    let microphone: MicrophoneAccess.State
    @Binding var showsMicrophoneGuide: Bool

    @Environment(AppServices.self) private var services
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        SectionHeader(
            AskAIScreenStrings.voiceSection,
            subtitle: String(localized: "talkai.section.sub",
                             defaultValue: "Start speaking at any time to interrupt the reply.",
                             comment: "Subtitle for the Ask AI voice settings")
        )

        ListGroup {
            if microphone != .authorized {
                warningRow(String(localized: "talkai.microphone.warning",
                                  defaultValue: "Microphone access is off, so Ask AI can't hear you.",
                                  comment: "Warning shown while oMLX has no microphone access"),
                           button: String(localized: "talkai.microphone.allow",
                                          defaultValue: "Allow…",
                                          comment: "Button that opens the microphone permission guide")) {
                    showsMicrophoneGuide = true
                }
            }
            if let models, models.isEmpty {
                warningRow(String(localized: "talkai.models.missing",
                                  defaultValue: "No voice model is installed. Ask via Voice needs a speech-to-speech model such as LFM3-Audio, LFM3-Omni or Qwen3-Omni.",
                                  comment: "Warning shown when no installed model can hold a voice conversation"),
                           button: String(localized: "talkai.models.open",
                                          defaultValue: "Open Downloads",
                                          comment: "Button that opens the Downloads screen to get a voice model")) {
                    services.requestedSection = .downloads
                }
            }
            Row(
                label: String(localized: "talkai.row.hotkey",
                              defaultValue: "Hotkey",
                              comment: "Row label for the Ask AI voice global shortcut"),
                sublabel: String(localized: "talkai.row.hotkey.sub",
                                 defaultValue: "Press it again or Esc to end the conversation. Press Space to finish your turn.",
                                 comment: "Sublabel for the Ask AI voice global shortcut")
            ) {
                HotKeyRecorder(combo: $store.prefs.talk.hotKey) { recording in
                    controller.setHotKeySuspended(recording)
                }
            }
            if let error = controller.talk.hotKeyError {
                FreeRow {
                    Label(error, systemImage: "exclamationmark.triangle.fill")
                        .font(.omlxText(DesignTokens.FontSize.callout))
                        .foregroundStyle(theme.redDot)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Row(
                label: AskAIScreenStrings.model,
                sublabel: String(localized: "talkai.row.model.sub",
                                 defaultValue: "Speech-to-speech models only (LFM3, Qwen3-Omni).",
                                 comment: "Sublabel for the Ask AI voice model picker")
            ) {
                Popup(selection: $store.prefs.talk.model, width: .controlWide,
                      options: AskAIScreen.modelOptions(models, current: store.prefs.talk.model))
            }
            Row(
                label: String(localized: "talkai.row.speed",
                              defaultValue: "Speech Speed",
                              comment: "Row label for the Ask AI voice reply playback speed"),
                sublabel: String(localized: "talkai.row.speed.sub",
                                 defaultValue: "Plays replies faster or slower without changing the pitch.",
                                 comment: "Sublabel for the Ask AI voice reply playback speed")
            ) {
                Popup(selection: $store.prefs.talk.speechSpeed, width: .controlCompact,
                      options: speedOptions)
            }
            GPUPowerLimitRow(
                isOn: $store.prefs.talk.gpuPowerLimitEnabled, watts: $store.prefs.talk.gpuPowerLimitWatts,
                supported: gpuPowerLimitSupported,
                sublabel: String(localized: "talkai.row.gpu_power_limit.sub",
                                 defaultValue: "Caps GPU power for the whole Mac during a voice conversation (2 to 1000 W). Low limits can make replies start late or stutter. If the model also has a limit, the lower one applies.",
                                 comment: "Sublabel for the Ask AI voice GPU power limit")
            )
        }
        // Voices belong to one model, so a choice never carries over.
        .onChange(of: store.prefs.talk.model) { store.prefs.talk.voice = "" }

        // System Settings style: a row that opens a page.
        ListGroup {
            NavRow(label: AskAIScreenStrings.advanced, isLast: true,
                   action: { services.askAIPage = .voiceAdvanced }) {
                Squircle(systemSymbol: "gearshape.fill", gradient: SquircleGradient.server)
            }
        }
        .padding(.top, 14)
    }

    private func warningRow(_ text: String, button: String, action: @escaping () -> Void) -> some View {
        FreeRow {
            HStack(spacing: 10) {
                Label(text, systemImage: "exclamationmark.triangle.fill")
                    .font(.omlxText(DesignTokens.FontSize.callout))
                    .foregroundStyle(theme.amberDot)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer()
                Button(button, action: action)
                    .buttonStyle(.omlx(.normal, size: .small))
            }
        }
    }

    private var speedOptions: [(Double, String)] {
        var speeds = TalkPrefs.speechSpeeds
        let current = store.prefs.talk.speechSpeed
        if !speeds.contains(current) { speeds = (speeds + [current]).sorted() }
        return speeds.map { ($0, "\($0.formatted(.number.precision(.fractionLength(1...2))))×") }
    }
}

private struct VoiceAdvancedSection: View {
    @Bindable var store: AskAIStore

    @Environment(AppServices.self) private var services
    /// Voices the selected model's realtime session can pick; empty hides the row.
    @State private var voices: [String] = []
    /// The selected model's default prompt; nil until the server reports it.
    @State private var modelDefault: String?
    /// What the selected model's realtime backend accepts; nil until known.
    @State private var capabilities: RealtimeVoiceCapabilities?

    var body: some View {
        let model = store.prefs.talk.model
        ListGroup {
            if !voices.isEmpty {
                Row(label: String(localized: "talkai.row.voice",
                                  defaultValue: "Voice",
                                  comment: "Row label for the Ask AI voice voice picker")) {
                    Popup(selection: $store.prefs.talk.voice, width: .controlCompact,
                          options: voiceOptions)
                }
            }
            if !model.isEmpty {
                FreeRow {
                    PromptField(text: modelPrompt(model),
                                placeholder: modelDefault == ""
                                    ? String(localized: "talkai.prompt.model.none",
                                             defaultValue: "Optional. This model has no recommended prompt.",
                                             comment: "Placeholder for the voice model prompt when the model has no default")
                                    : "",
                                defaultText: modelDefault,
                                label: String(localized: "talkai.prompt.model",
                                              defaultValue: "Model Prompt",
                                              comment: "Label above the prompt used only with the selected voice model"),
                                caption: String(localized: "talkai.prompt.model.sub",
                                                defaultValue: "Used only with \(model), before the general prompt. Reset to Default puts back the model's recommended prompt.",
                                                comment: "Explains the voice model prompt; the argument is the model name"))
                }
            }
            FreeRow(isLast: true) {
                PromptField(text: $store.prefs.talk.systemPrompt,
                            placeholder: String(localized: "talkai.prompt.placeholder",
                                                defaultValue: "Optional. For example: Keep answers short and friendly.",
                                                comment: "Placeholder for the Ask AI voice system prompt"),
                            defaultText: TalkPrefs.defaultPrompt,
                            label: String(localized: "talkai.prompt.general",
                                          defaultValue: "General Prompt",
                                          comment: "Label above the prompt used with every voice model"),
                            caption: String(localized: "talkai.prompt.general.sub",
                                            defaultValue: "Used with every voice model, after the model prompt.",
                                            comment: "Explains the voice general prompt"))
            }
        }
        .task(id: [model, String(services.serverState.isRunningLike)]) {
            await loadModelInfo()
        }

        if let capabilities {
            if capabilities.tools {
                ListGroup {
                    ToolRows(tools: $store.prefs.talk.tools, modelReadsImages: capabilities.inputImage,
                             isLast: true)
                }
                .padding(.top, 14)
            } else {
                HintLine(text: String(localized: "talkai.tools.unsupported",
                                      defaultValue: "This voice model can't call tools.",
                                      comment: "Hint under the voice settings when the selected model has no tool calling"))
                    .padding(.horizontal, 28)
                    .padding(.top, 8)
            }
        }
    }

    /// The model's own prompt until the user edits it; matching the default again forgets the edit.
    private func modelPrompt(_ model: String) -> Binding<String> {
        Binding(
            get: { store.prefs.talk.modelPrompts[model] ?? modelDefault ?? "" },
            set: { text in
                store.prefs.talk.modelPrompts[model] = text == modelDefault ? nil : text
            }
        )
    }

    private var voiceOptions: [(String, String)] {
        var options: [(String, String)] = [
            ("", String(localized: "talkai.voice.default",
                        defaultValue: "Default",
                        comment: "Voice picker option that uses the model's default voice")),
        ]
        let current = store.prefs.talk.voice
        if !current.isEmpty, !voices.contains(current) { options.append((current, current)) }
        options += voices.map { ($0, $0.capitalized) }
        return options
    }

    /// Voices and default prompt of the selected model's realtime session.
    private func loadModelInfo() async {
        let model = store.prefs.talk.model
        let config = services.config
        guard !model.isEmpty,
              var components = AppConfig.httpURL(host: config.host, port: config.port,
                                                 path: "/v1/audio/voices")
                  .flatMap({ URLComponents(url: $0, resolvingAgainstBaseURL: false) })
        else {
            voices = []
            modelDefault = nil
            capabilities = nil
            return
        }
        // Another model's default must not show while this one loads.
        modelDefault = nil
        capabilities = nil
        components.queryItems = [URLQueryItem(name: "model", value: model),
                                 URLQueryItem(name: "realtime", value: "true")]
        guard let url = components.url else { return }
        var request = URLRequest(url: url)
        if let key = config.apiKey, !key.isEmpty {
            request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        }
        struct Info: Decodable {
            struct Capabilities: Decodable {
                let inputImage: Bool
                let tools: Bool
            }
            let voices: [String]
            let instructions: String?
            let capabilities: Capabilities?
        }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        guard let (data, _) = try? await URLSession.shared.data(for: request),
              let decoded = try? decoder.decode(Info.self, from: data)
        else {
            // Another model's voices would be rejected by this one.
            voices = []
            modelDefault = nil
            return
        }
        voices = decoded.voices
        modelDefault = decoded.instructions
        capabilities = decoded.capabilities.map {
            RealtimeVoiceCapabilities(inputImage: $0.inputImage, tools: $0.tools)
        }
    }
}

/// Step-by-step microphone permission guide shown when Ask via Voice is
/// turned on without access.
private struct MicrophoneGuideSheet: View {
    @Binding var microphone: MicrophoneAccess.State
    @Environment(\.dismiss) private var dismiss
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(String(localized: "talkai.guide.title",
                        defaultValue: "Allow Microphone Access",
                        comment: "Title of the Ask AI voice microphone permission guide"))
                .font(.omlxText(DesignTokens.FontSize.title3, weight: .semibold))
                .foregroundStyle(theme.text)
            step(1, String(localized: "talkai.guide.step1",
                           defaultValue: "Click Continue, then choose Allow when macOS asks to use the microphone.",
                           comment: "First step of the microphone permission guide"))
            step(2, String(localized: "talkai.guide.step2",
                           defaultValue: "If you denied it before, open System Settings > Privacy & Security > Microphone and turn on oMLX.",
                           comment: "Second step of the microphone permission guide"))
            step(3, String(localized: "talkai.guide.step3",
                           defaultValue: "Choose a voice model below, then press the hotkey and start talking.",
                           comment: "Third step of the microphone permission guide"))
            HStack {
                Button(String(localized: "talkai.guide.open_settings",
                              defaultValue: "Open System Settings",
                              comment: "Button that opens the macOS microphone privacy settings")) {
                    MicrophoneAccess.openSystemSettings()
                }
                .buttonStyle(.omlx(.normal, size: .small))
                Spacer()
                Button(String(localized: "talkai.guide.later",
                              defaultValue: "Not Now",
                              comment: "Button that closes the microphone permission guide")) {
                    dismiss()
                }
                .buttonStyle(.omlx(.normal, size: .small))
                Button(String(localized: "talkai.guide.continue",
                              defaultValue: "Continue",
                              comment: "Button that asks macOS for microphone access")) {
                    Task {
                        _ = await MicrophoneAccess.request()
                        microphone = MicrophoneAccess.state
                        if microphone == .authorized { dismiss() }
                    }
                }
                .buttonStyle(.omlx(.primary, size: .small))
                .disabled(microphone == .denied)
            }
            .padding(.top, 6)
        }
        .padding(22)
        .frame(width: 420)
    }

    private func step(_ number: Int, _ text: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Text("\(number)")
                .font(.omlxText(DesignTokens.FontSize.callout, weight: .semibold))
                .foregroundStyle(theme.accentText)
                .frame(width: 20, height: 20)
                .background(Circle().fill(theme.accent))
            Text(text)
                .font(.omlxText(DesignTokens.FontSize.body))
                .foregroundStyle(theme.text)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

// MARK: - Presets

private struct PresetsSection: View {
    @Bindable var store: AskAIStore
    let models: [String]?
    let visionModels: Set<String>?

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
            AskAIScreenStrings.presets,
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

        presetList
    }

    private var presetList: some View {
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
        PresetEditor(preset: $draft, models: models, visionModels: visionModels, error: editError,
                     isLast: isLast) {
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
        if preset.tools.enabled { parts.append(AskAIScreenStrings.toolCalling) }
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
    let visionModels: Set<String>?
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
        ToolRows(tools: $preset.tools,
                 modelReadsImages: AskAIScreen.readsImages(preset.model, in: visionModels))
        FreeRow {
            PromptField(text: $preset.systemPrompt,
                        placeholder: String(localized: "askai.prompt.preset_placeholder",
                                            defaultValue: "For example: Translate the text into English in a concise, professional Slack tone. Output only the translation.",
                                            comment: "Placeholder for an Ask AI preset system prompt"),
                        caption: String(localized: "askai.prompt.preset.sub",
                                        defaultValue: "Used instead of the default system prompt while this preset runs, so add the date or other placeholders here if the preset needs them.",
                                        comment: "Explains that a preset prompt replaces the default system prompt"))
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
    /// Offered as Reset to Default while the text differs from it.
    var defaultText: String?
    var label = String(localized: "askai.row.system_prompt",
                       defaultValue: "System Prompt",
                       comment: "Label above the Ask AI system prompt editor")
    /// How the prompt is used, under the label.
    var caption: String?

    @Environment(\.omlxTheme) private var theme
    @State private var clipboardNeedsPermission = ClipboardAccess.needsPermission
    @State private var showsClipboardGuide = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(label)
                        .font(.omlxText(DesignTokens.FontSize.body, weight: .medium))
                        .foregroundStyle(theme.text)
                    if let caption {
                        Text(caption)
                            .font(.omlxText(11.5))
                            .foregroundStyle(theme.textSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                Spacer()
                if let defaultText, text != defaultText {
                    Button(String(localized: "askai.prompt.reset",
                                  defaultValue: "Reset to Default",
                                  comment: "Button that puts the default Ask AI system prompt back")) {
                        text = defaultText
                    }
                    .buttonStyle(.omlx(.normal, size: .small))
                }
            }
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
            if text.contains(PromptPlaceholder.clipboard.token), clipboardNeedsPermission {
                HStack(spacing: 10) {
                    Label(String(localized: "askai.clipboard.warning",
                                 defaultValue: "macOS asks before oMLX reads the clipboard for {clipboard}.",
                                 comment: "Warning shown while the clipboard placeholder needs a paste permission"),
                          systemImage: "exclamationmark.triangle.fill")
                        .font(.omlxText(DesignTokens.FontSize.callout))
                        .foregroundStyle(theme.amberDot)
                        .fixedSize(horizontal: false, vertical: true)
                    Spacer()
                    Button(String(localized: "askai.clipboard.how",
                                  defaultValue: "How to Allow…",
                                  comment: "Button that opens the clipboard permission guide")) {
                        showsClipboardGuide = true
                    }
                    .buttonStyle(.omlx(.normal, size: .small))
                }
            }
            PlaceholderList { token in
                let separator = text.isEmpty || text.last?.isWhitespace == true ? "" : " "
                text += separator + token
            }
        }
        .sheet(isPresented: $showsClipboardGuide) {
            ClipboardGuideSheet()
                .environment(\.omlxTheme, theme)
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            clipboardNeedsPermission = ClipboardAccess.needsPermission
        }
    }
}

/// The placeholders a prompt can use, with the value each would get now.
private struct PlaceholderList: View {
    let add: (String) -> Void

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        let context = PromptContext()
        VStack(alignment: .leading, spacing: 4) {
            Text(String(localized: "askai.placeholders.help",
                        defaultValue: "Placeholders are filled in when a conversation starts. Click one to add it.",
                        comment: "Help text above the list of Ask AI prompt placeholders"))
                .font(.omlxText(11.5))
                .foregroundStyle(theme.textSecondary)
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 2) {
                ForEach(PromptPlaceholder.allCases, id: \.self) { placeholder in
                    GridRow {
                        Button(placeholder.token) { add(placeholder.token) }
                            .buttonStyle(.plain)
                            .font(.omlxMono(11.5, weight: .medium))
                            .foregroundStyle(theme.accent)
                        Text(Self.example(placeholder, context))
                            .font(.omlxText(11.5))
                            .foregroundStyle(theme.textTertiary)
                            .lineLimit(1)
                    }
                }
            }
        }
        .padding(.top, 2)
    }

    /// Live values, except for those that depend on when and where Ask AI opens.
    /// The clipboard is never read here.
    private static func example(_ placeholder: PromptPlaceholder, _ context: PromptContext) -> String {
        switch placeholder {
        case .app:
            String(localized: "askai.placeholders.app",
                   defaultValue: "The app in front when Ask AI opens",
                   comment: "Description of the {app} prompt placeholder")
        case .model:
            String(localized: "askai.placeholders.model",
                   defaultValue: "The model that answers",
                   comment: "Description of the {model} prompt placeholder")
        case .clipboard:
            String(localized: "askai.placeholders.clipboard",
                   defaultValue: "Text on the clipboard, up to 4,000 characters",
                   comment: "Description of the {clipboard} prompt placeholder")
        default:
            PromptPlaceholders.value(of: placeholder, context: context)
        }
    }
}

/// Step-by-step guide for the paste permission that {clipboard} needs.
private struct ClipboardGuideSheet: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(String(localized: "askai.clipboard.guide.title",
                        defaultValue: "Allow Clipboard Access",
                        comment: "Title of the clipboard permission guide"))
                .font(.omlxText(DesignTokens.FontSize.title3, weight: .semibold))
                .foregroundStyle(theme.text)
            step(1, String(localized: "askai.clipboard.guide.step1",
                           defaultValue: "Click Test, then choose Allow when macOS asks to paste from another app.",
                           comment: "First step of the clipboard permission guide"))
            step(2, String(localized: "askai.clipboard.guide.step2",
                           defaultValue: "To stop the question on every conversation, open System Settings > Privacy & Security and choose Always Allow for oMLX.",
                           comment: "Second step of the clipboard permission guide"))
            HStack {
                Button(String(localized: "askai.clipboard.guide.open_settings",
                              defaultValue: "Open System Settings",
                              comment: "Button that opens the macOS Privacy & Security settings")) {
                    ClipboardAccess.openSystemSettings()
                }
                .buttonStyle(.omlx(.normal, size: .small))
                Spacer()
                Button(String(localized: "askai.clipboard.guide.test",
                              defaultValue: "Test",
                              comment: "Button that reads the clipboard once so macOS asks for permission")) {
                    _ = NSPasteboard.general.string(forType: .string)
                }
                .buttonStyle(.omlx(.normal, size: .small))
                Button(String(localized: "askai.clipboard.guide.done",
                              defaultValue: "Done",
                              comment: "Button that closes the clipboard permission guide")) {
                    dismiss()
                }
                .buttonStyle(.omlx(.primary, size: .small))
            }
            .padding(.top, 6)
        }
        .padding(22)
        .frame(width: 420)
    }

    private func step(_ number: Int, _ text: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Text("\(number)")
                .font(.omlxText(DesignTokens.FontSize.callout, weight: .semibold))
                .foregroundStyle(theme.accentText)
                .frame(width: 20, height: 20)
                .background(Circle().fill(theme.accent))
            Text(text)
                .font(.omlxText(DesignTokens.FontSize.body))
                .foregroundStyle(theme.text)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

private enum AskAIScreenStrings {
    static var advanced: String {
        String(localized: "askai.section.advanced", defaultValue: "Advanced Settings",
               comment: "Row and page title for the less used Ask AI text or voice settings")
    }
    static var presets: String {
        String(localized: "askai.section.presets", defaultValue: "Presets",
               comment: "Section header for the Ask AI slash-command presets")
    }
    static var textSection: String {
        String(localized: "askai.section.text", defaultValue: "Ask via Text",
               comment: "Section header for the Ask AI text bar settings")
    }
    static var voiceSection: String {
        String(localized: "talkai.section", defaultValue: "Ask via Voice",
               comment: "Section header for the Ask AI voice conversation settings")
    }
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
    static var restartApp: String {
        String(localized: "askai.restart_app", defaultValue: "Restart oMLX",
               comment: "Button that quits and reopens oMLX so a new permission applies")
    }
    static var toolCalling: String {
        String(localized: "askai.tools.row", defaultValue: "Tool Calling",
               comment: "Row label for the Ask AI tool calling switch")
    }
    static var autoCopySub: String {
        String(localized: "askai.row.auto_copy.sub",
               defaultValue: "Copy the answer to the clipboard when it finishes.",
               comment: "Sublabel for the Ask AI auto copy switch")
    }
}

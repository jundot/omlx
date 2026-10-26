// Owns the Ask AI feature at runtime: preferences, the global hotkey, the
// floating panel, and the current conversation. AppServices holds one
// instance; AppDelegate calls `start` outside unit tests.

import AppKit
import Foundation
import Observation

@MainActor
@Observable
final class AskAIController {
    struct Notice: Equatable {
        let text: String
        var offersSettings = false
    }

    let store: AskAIStore
    let conversation = AskAIConversation()

    var draft = "" {
        didSet {
            guard draft != oldValue else { return }
            suggestionsDismissed = false
            selectedSuggestion = 0
        }
    }
    var selectedSuggestion = 0
    private(set) var attachments: [AskAIImage] = []
    private(set) var notice: Notice?
    private(set) var hotKeyError: String?
    private(set) var showsCopied = false
    /// Panel size limits for the current screen and position, set by the
    /// panel controller.
    private(set) var panelWidth = CGFloat(AskAIPrefs.defaultPanelWidth)
    private(set) var maxPanelHeight: CGFloat = 600

    @ObservationIgnored let inputFocus = AskAIInputFocus()
    @ObservationIgnored private let hotKey = GlobalHotKey()
    @ObservationIgnored private var panel: AskAIPanelController?
    @ObservationIgnored private weak var services: AppServices?
    @ObservationIgnored private var openSettingsAction: (() -> Void)?
    @ObservationIgnored private var hotKeySuspended = false
    @ObservationIgnored private var copiedToken = 0
    /// Model id -> `model_type` from the admin model list, for the image check.
    @ObservationIgnored private var modelTypes: [String: String] = [:]
    @ObservationIgnored private var isSubmitting = false
    @ObservationIgnored private var lastClosedAt = Date.distantPast
    /// Pastes whose images are still being checked or encoded.
    @ObservationIgnored private var loadingAttachments = 0
    @ObservationIgnored private var submitAfterAttachments = false
    private var suggestionsDismissed = false

    init(store: AskAIStore = AskAIStore()) {
        self.store = store
    }

    func start(services: AppServices, openSettings: @escaping () -> Void) {
        self.services = services
        self.openSettingsAction = openSettings
        store.onChange = { [weak self] old, new in
            guard let self else { return }
            if old.enabled != new.enabled || old.hotKey != new.hotKey {
                self.applyHotKey()
            }
            if !new.enabled {
                self.hidePanel()
            }
        }
        applyHotKey()
    }

    func shutdown() {
        store.flush()
        hotKey.invalidate()
    }

    /// The recorder releases the hotkey while it listens, so pressing the
    /// current combo reaches the recorder instead of opening the panel.
    func setHotKeySuspended(_ suspended: Bool) {
        hotKeySuspended = suspended
        applyHotKey()
    }

    private func applyHotKey() {
        guard services != nil, store.prefs.enabled, !hotKeySuspended else {
            hotKey.unregister()
            if !store.prefs.enabled { hotKeyError = nil }
            return
        }
        let combo = store.prefs.hotKey
        let status = hotKey.register(combo) { [weak self] in
            self?.togglePanel()
        }
        hotKeyError = status == noErr ? nil : String(
            localized: "askai.hotkey.error.in_use",
            defaultValue: "\(combo.displayString) is already used by macOS or another app. Choose a different shortcut.",
            comment: "Shown when the Ask AI hotkey cannot be registered; the argument is the shortcut"
        )
    }

    // MARK: - Panel

    var isPanelVisible: Bool { panel?.isVisible == true }

    func togglePanel() {
        if isPanelVisible {
            hidePanel()
        } else {
            showPanel()
        }
    }

    func showPanel() {
        guard store.prefs.enabled else { return }
        // Idle time counts from the last send, answer, or close.
        let lastUse = max(conversation.lastActivity, lastClosedAt)
        if !conversation.isStreaming,
           Date().timeIntervalSince(lastUse) > store.prefs.newConversationAfter {
            conversation.reset()
        }
        notice = nil
        let panel = self.panel ?? AskAIPanelController(controller: self)
        self.panel = panel
        panel.show()
        inputFocus.focus()
        // Model types only feed the image check and rarely change; a miss
        // refreshes on demand.
        if modelTypes.isEmpty {
            Task { await refreshModelTypes() }
        }
    }

    func hidePanel() {
        panel?.hide()
    }

    /// Called by the panel controller on every hide, whatever the cause.
    func panelDidHide() {
        lastClosedAt = Date()
    }

    func updatePanelLayout(width: CGFloat, maxHeight: CGFloat) {
        panelWidth = width
        maxPanelHeight = maxHeight
    }

    func savePanelPosition(_ topLeft: NSPoint?) {
        store.prefs.panelPosition = topLeft.map { AskAIPanelPosition(x: $0.x, y: $0.y) }
    }

    func savePanelWidth(_ width: CGFloat) {
        store.prefs.panelWidth = width
    }

    func openSettings() {
        hidePanel()
        openSettingsAction?()
    }

    // MARK: - Input

    var visibleSuggestions: [AskAIPreset] {
        guard !suggestionsDismissed else { return [] }
        return Array(AskAIInput.suggestions(for: draft, presets: store.prefs.presets).prefix(6))
    }

    /// Preset command and model the next send would use, for the footer.
    var pendingTarget: (command: String?, model: String)? {
        switch AskAIInput.parse(draft, presets: store.prefs.presets) {
        case .preset(let preset, _):
            return (preset.command, preset.model)
        case .unknownCommand:
            return nil
        case .plain:
            if let profile = conversation.profile {
                return (conversation.command, profile.model)
            }
            let model = store.prefs.defaultModel
            return model.isEmpty ? nil : (nil, model)
        }
    }

    func handle(_ command: AskAIInputCommand) {
        switch command {
        case .submit: submit()
        case .stop: conversation.stop()
        case .newConversation:
            conversation.reset()
            notice = nil
        case .copyLastAnswer: copyLastAnswer()
        case .escape: escape()
        case .suggestionUp: moveSuggestion(by: -1)
        case .suggestionDown: moveSuggestion(by: 1)
        case .acceptSuggestion: acceptSuggestion()
        }
    }

    /// What a submit resolved to before any async check.
    private struct PendingSend {
        let profile: AskAIProfile
        let command: String?
        let text: String
        let startsConversation: Bool
    }

    func submit() {
        guard !conversation.isStreaming, !isSubmitting else { return }
        if loadingAttachments > 0 {
            // Send once the pasted images are ready.
            submitAfterAttachments = true
            return
        }
        notice = nil
        let images = attachments
        let submittedDraft = draft
        guard let pending = resolveSend(hasImages: !images.isEmpty) else { return }
        guard !images.isEmpty else {
            start(pending, images: [], submittedDraft: submittedDraft)
            return
        }
        // The server drops image parts for text-only models without an
        // error, so refuse before sending, like the web chat does.
        isSubmitting = true
        Task {
            defer { isSubmitting = false }
            if let refusal = await imageRefusal(model: pending.profile.model) {
                notice = refusal
                return
            }
            start(pending, images: images, submittedDraft: submittedDraft)
        }
    }

    private func resolveSend(hasImages: Bool) -> PendingSend? {
        switch AskAIInput.parse(draft, presets: store.prefs.presets) {
        case .unknownCommand(let command):
            notice = Self.unknownPresetNotice(command)
            return nil
        case .preset(let preset, let body):
            guard !body.isEmpty || hasImages else {
                notice = Notice(text: String(
                    localized: "askai.notice.empty_body",
                    defaultValue: "Type the text to send after /\(preset.command).",
                    comment: "Shown when a slash command has no text after it"
                ))
                return nil
            }
            guard !preset.model.isEmpty else {
                notice = Self.missingModelNotice
                return nil
            }
            return PendingSend(profile: preset.profile, command: preset.command,
                               text: body, startsConversation: true)
        case .plain(let text):
            guard !text.isEmpty || hasImages else { return nil }
            if let profile = conversation.profile {
                return PendingSend(profile: currentProfile(command: conversation.command, fallback: profile),
                                   command: conversation.command, text: text, startsConversation: false)
            }
            let profile = store.prefs.defaultProfile
            guard !profile.model.isEmpty else {
                notice = Self.noDefaultModelNotice
                return nil
            }
            return PendingSend(profile: profile, command: nil, text: text, startsConversation: true)
        }
    }

    /// Latest settings for the conversation's preset, or the default profile
    /// for a plain conversation, so a toggle such as Auto Copy changed
    /// mid-conversation applies to the next follow-up.
    private func currentProfile(command: String?, fallback: AskAIProfile) -> AskAIProfile {
        let latest = if let command {
            store.prefs.presets.first { $0.command == command }?.profile
        } else {
            store.prefs.defaultProfile
        }
        guard let latest, !latest.model.isEmpty else { return fallback }
        return latest
    }

    private func start(_ pending: PendingSend, images: [AskAIImage], submittedDraft: String) {
        guard let config = services?.config,
              let url = AppConfig.httpURL(host: config.host, port: config.port,
                                          path: "/v1/chat/completions")
        else { return }
        if pending.startsConversation {
            conversation.begin(profile: pending.profile, command: pending.command)
        } else {
            conversation.updateProfile(pending.profile)
        }
        // A reset during the image check leaves nothing to follow up on.
        // Keep the input so the user can send again.
        guard conversation.send(pending.text, images: images,
                                gpuPowerLimitW: store.prefs.requestGPUPowerLimit,
                                using: ChatCompletionClient(url: url, apiKey: config.apiKey))
        else { return }
        if draft == submittedDraft { draft = "" }
        attachments.removeAll { images.contains($0) }
    }

    /// Nil for a vision model, otherwise the notice that explains why images
    /// cannot go to `model`.
    private func imageRefusal(model: String) async -> Notice? {
        if modelTypes[model] == nil {
            await refreshModelTypes()
        }
        switch modelTypes[model] {
        case .some("vlm"):
            return nil
        case .none:
            return Notice(text: String(
                localized: "askai.notice.model_type_unknown",
                defaultValue: "Couldn't check whether \(model) reads images. Make sure the server is running.",
                comment: "Shown when the model list cannot be loaded to check image support"
            ))
        default:
            return Notice(text: String(
                localized: "askai.notice.model_no_images",
                defaultValue: "\(model) can't read images. Use a vision (VLM) model for questions with images.",
                comment: "Shown when an image is pasted or sent but the target model is not a vision model"
            ), offersSettings: true)
        }
    }

    private func refreshModelTypes() async {
        guard let client = services?.client,
              let models = try? await client.listModels().models else { return }
        modelTypes = Dictionary(models.map { ($0.id, $0.modelType ?? "") },
                                uniquingKeysWith: { first, _ in first })
    }

    private static func unknownPresetNotice(_ command: String) -> Notice {
        Notice(
            text: String(localized: "askai.notice.unknown_preset",
                         defaultValue: "No preset named /\(command). Add it in Ask AI settings.",
                         comment: "Shown when the typed slash command matches no preset"),
            offersSettings: true
        )
    }

    private static var noDefaultModelNotice: Notice {
        Notice(
            text: String(localized: "askai.notice.no_default_model",
                         defaultValue: "Choose a default model in Ask AI settings.",
                         comment: "Shown when a plain question is sent with no default model"),
            offersSettings: true
        )
    }

    private static var missingModelNotice: Notice {
        Notice(text: AskAIPresetValidation.missingModel.message, offersSettings: true)
    }

    // MARK: - Attachments

    /// Handles a paste that carries images. Returns false to let the text
    /// view paste normally. Images attach only when the model the next send
    /// targets is a vision model.
    func attachImages(from pasteboard: NSPasteboard) -> Bool {
        let sources = AskAIImage.sources(on: pasteboard)
        guard !sources.isEmpty else { return false }
        notice = nil
        guard let model = imageTargetModel() else { return true }
        loadingAttachments += 1
        Task {
            defer { finishAttachmentLoad() }
            if let refusal = await imageRefusal(model: model) {
                notice = refusal
                submitAfterAttachments = false
                return
            }
            for source in sources {
                guard let image = await AskAIImage.make(from: source) else { continue }
                guard attachments.count < AskAIImage.maxCount else {
                    notice = Notice(text: String(
                        localized: "askai.notice.too_many_images",
                        defaultValue: "You can attach up to \(AskAIImage.maxCount) images.",
                        comment: "Shown when more images are pasted than the Ask AI limit"
                    ))
                    break
                }
                attachments.append(image)
            }
        }
        return true
    }

    /// Model the next send would go to. Nil after setting the notice that
    /// says why there is none.
    private func imageTargetModel() -> String? {
        switch AskAIInput.parse(draft, presets: store.prefs.presets) {
        case .unknownCommand(let command):
            notice = Self.unknownPresetNotice(command)
            return nil
        case .preset(let preset, _):
            guard !preset.model.isEmpty else {
                notice = Self.missingModelNotice
                return nil
            }
            return preset.model
        case .plain:
            let model = conversation.profile?.model ?? store.prefs.defaultModel
            guard !model.isEmpty else {
                notice = Self.noDefaultModelNotice
                return nil
            }
            return model
        }
    }

    private func finishAttachmentLoad() {
        loadingAttachments -= 1
        guard loadingAttachments == 0, submitAfterAttachments else { return }
        submitAfterAttachments = false
        submit()
    }

    func removeAttachment(_ id: UUID) {
        attachments.removeAll { $0.id == id }
    }

    private func escape() {
        if !visibleSuggestions.isEmpty {
            suggestionsDismissed = true
        } else if conversation.isStreaming {
            conversation.stop()
        } else {
            hidePanel()
        }
    }

    private func moveSuggestion(by delta: Int) {
        let count = visibleSuggestions.count
        guard count > 0 else { return }
        selectedSuggestion = (selectedSuggestion + delta + count) % count
    }

    func acceptSuggestion(at index: Int? = nil) {
        let list = visibleSuggestions
        guard !list.isEmpty else { return }
        let preset = list[min(index ?? selectedSuggestion, list.count - 1)]
        draft = "/\(preset.command) "
        inputFocus.focus()
    }

    private func copyLastAnswer() {
        guard let answer = conversation.lastAnswer?.trimmingCharacters(in: .whitespacesAndNewlines),
              !answer.isEmpty else { return }
        AskAIClipboard.copy(answer)
        copiedToken += 1
        let token = copiedToken
        showsCopied = true
        Task { [weak self] in
            try? await Task.sleep(for: .seconds(1.5))
            guard let self, self.copiedToken == token else { return }
            self.showsCopied = false
        }
    }
}

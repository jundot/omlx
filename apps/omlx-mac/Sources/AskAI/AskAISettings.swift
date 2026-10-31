// Ask AI preferences: the feature switch, global hotkey, default request
// profile, and slash-command presets. Stored as JSON under Application
// Support because nothing here belongs to the server's settings.json.

import Foundation
import Observation

/// Model, prompt, and output options used for one conversation.
struct AskAIProfile: Equatable, Sendable {
    var model: String
    var systemPrompt: String
    var allowThinking: Bool
    var autoCopy: Bool
    var tools = AskAIToolPrefs()
}

/// Tools the model may call while it answers. Off until tool calling is on.
struct AskAIToolPrefs: Codable, Equatable, Sendable {
    var enabled = false
    var webSearch = true
    var readLinks = true
    var screenCapture = false

    init() {}

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let fallback = AskAIToolPrefs()
        enabled = c.lenient(Bool.self, .enabled) ?? fallback.enabled
        webSearch = c.lenient(Bool.self, .webSearch) ?? fallback.webSearch
        readLinks = c.lenient(Bool.self, .readLinks) ?? fallback.readLinks
        screenCapture = c.lenient(Bool.self, .screenCapture) ?? fallback.screenCapture
    }
}

struct AskAIPreset: Codable, Equatable, Identifiable, Sendable {
    var id: UUID
    /// Command name without the leading slash.
    var command: String
    var model: String
    var systemPrompt: String
    var allowThinking: Bool
    var autoCopy: Bool
    var tools: AskAIToolPrefs

    init(
        id: UUID = UUID(),
        command: String = "",
        model: String = "",
        systemPrompt: String = "",
        allowThinking: Bool = false,
        autoCopy: Bool = false,
        tools: AskAIToolPrefs = AskAIToolPrefs()
    ) {
        self.id = id
        self.command = command
        self.model = model
        self.systemPrompt = systemPrompt
        self.allowThinking = allowThinking
        self.autoCopy = autoCopy
        self.tools = tools
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = c.lenient(UUID.self, .id) ?? UUID()
        command = c.lenient(String.self, .command) ?? ""
        model = c.lenient(String.self, .model) ?? ""
        systemPrompt = c.lenient(String.self, .systemPrompt) ?? ""
        allowThinking = c.lenient(Bool.self, .allowThinking) ?? false
        autoCopy = c.lenient(Bool.self, .autoCopy) ?? false
        tools = c.lenient(AskAIToolPrefs.self, .tools) ?? AskAIToolPrefs()
    }

    var profile: AskAIProfile {
        AskAIProfile(model: model, systemPrompt: systemPrompt,
                     allowThinking: allowThinking, autoCopy: autoCopy, tools: tools)
    }

    /// First non-blank line of the prompt, capped for one-line previews.
    /// Prompts can run to thousands of lines, so this never scans past it.
    var promptPreview: String {
        let start = systemPrompt.drop(while: \.isWhitespace)
        return String(start.prefix { !$0.isNewline }.prefix(120))
    }
}

/// Top-left corner of the Ask AI panel in screen points.
struct AskAIPanelPosition: Codable, Equatable, Sendable {
    var x: Double
    var y: Double
}

/// Ask AI voice mode: a spoken conversation with a voice model over /v1/realtime.
struct TalkPrefs: Codable, Equatable, Sendable {
    /// The voice models bring their own speaking style, so the general
    /// prompt only adds what they cannot know.
    static let defaultPrompt = "Current date and time: {weekday}, {datetime} ({timezone})."

    var enabled = false
    var hotKey = HotKeyCombo.optionShiftSpace
    var model = ""
    /// Sent with every voice model, after the model prompt.
    var systemPrompt = TalkPrefs.defaultPrompt
    /// Model prompts the user edited, by model id. A model without one uses
    /// the default the server reports for it.
    var modelPrompts: [String: String] = [:]
    /// Empty uses the model's default voice.
    var voice = ""
    /// Reply playback rate; pitch stays the same.
    var speechSpeed = 1.0
    /// System-wide GPU power cap for the whole conversation.
    var gpuPowerLimitEnabled = false
    var gpuPowerLimitWatts = 10
    /// Tools the voice model may call, when its backend can call tools.
    var tools = AskAIToolPrefs()

    init() {}

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let fallback = TalkPrefs()
        enabled = c.lenient(Bool.self, .enabled) ?? fallback.enabled
        hotKey = c.lenient(HotKeyCombo.self, .hotKey) ?? fallback.hotKey
        model = c.lenient(String.self, .model) ?? fallback.model
        systemPrompt = c.lenient(String.self, .systemPrompt) ?? fallback.systemPrompt
        modelPrompts = c.lenient([String: String].self, .modelPrompts) ?? fallback.modelPrompts
        voice = c.lenient(String.self, .voice) ?? fallback.voice
        speechSpeed = c.lenient(Double.self, .speechSpeed).map(TalkPrefs.clampedSpeed) ?? fallback.speechSpeed
        gpuPowerLimitEnabled = c.lenient(Bool.self, .gpuPowerLimitEnabled) ?? fallback.gpuPowerLimitEnabled
        gpuPowerLimitWatts = c.lenient(Int.self, .gpuPowerLimitWatts) ?? fallback.gpuPowerLimitWatts
        tools = c.lenient(AskAIToolPrefs.self, .tools) ?? fallback.tools
    }

    var requestGPUPowerLimit: Int? {
        AskAIPrefs.requestGPUPowerLimit(enabled: gpuPowerLimitEnabled, watts: gpuPowerLimitWatts)
    }

    /// The session instructions: the model prompt, then the general prompt.
    /// `modelDefault` is the prompt the server opened the session with.
    func instructions(modelDefault: String, context: PromptContext) -> String {
        [modelPrompts[model] ?? modelDefault, systemPrompt]
            .map { PromptPlaceholders.fill($0, context: context).trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: "\n\n")
    }

    static let speechSpeeds = [0.8, 0.9, 1.0, 1.1, 1.25, 1.5]

    static func clampedSpeed(_ speed: Double) -> Double {
        min(max(speed, 0.5), 2.0)
    }
}

struct AskAIPrefs: Codable, Equatable, Sendable {
    static let defaultPanelWidth: Double = 680
    static let defaultPrompt = """
        You are Ask AI, a quick assistant the user opens over any app on their Mac.
        Current date and time: {weekday}, {datetime} ({timezone}). The app in front: {app}.

        In user messages, "I/me/my" refer to the user and "you/your" refer to you; keep that perspective in your replies.
        Put the answer first and keep it short. Add detail only when it helps or the user asks for it.
        Use Markdown lists, tables, or code blocks only when they make the answer clearer.
        Reply in the language the user writes in unless they ask otherwise.
        If the request is unclear and none of your tools can settle it, ask one short question.
        """

    var enabled = false
    var hotKey = HotKeyCombo.optionSpace
    var defaultModel = ""
    var defaultSystemPrompt = AskAIPrefs.defaultPrompt
    var defaultAllowThinking = false
    var defaultAutoCopy = false
    var defaultTools = AskAIToolPrefs()
    /// Idle time after which reopening the panel starts a new conversation.
    var newConversationAfter: TimeInterval = 300
    var presets: [AskAIPreset] = []
    /// Set after the user drags the panel; nil uses the default spot.
    var panelPosition: AskAIPanelPosition?
    var panelWidth = AskAIPrefs.defaultPanelWidth
    /// System-wide GPU power cap while Ask AI answers. The server applies the
    /// lower of this and the model's own limit.
    var gpuPowerLimitEnabled = false
    var gpuPowerLimitWatts = 10
    var talk = TalkPrefs()
    /// False in settings saved before prompts had a default; their empty
    /// prompts get the default once, and a prompt cleared later stays empty.
    var promptDefaultsApplied = true
    /// Return sends and Shift-Return inserts the newline. Command-Return
    /// sends either way.
    var sendOnEnter = false

    init() {}

    /// A bad or missing field falls back on its own, so one stale value
    /// never resets the presets and prompts around it.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let fallback = AskAIPrefs()
        enabled = c.lenient(Bool.self, .enabled) ?? fallback.enabled
        hotKey = c.lenient(HotKeyCombo.self, .hotKey) ?? fallback.hotKey
        defaultModel = c.lenient(String.self, .defaultModel) ?? fallback.defaultModel
        defaultSystemPrompt = c.lenient(String.self, .defaultSystemPrompt) ?? fallback.defaultSystemPrompt
        defaultAllowThinking = c.lenient(Bool.self, .defaultAllowThinking) ?? fallback.defaultAllowThinking
        defaultAutoCopy = c.lenient(Bool.self, .defaultAutoCopy) ?? fallback.defaultAutoCopy
        defaultTools = c.lenient(AskAIToolPrefs.self, .defaultTools) ?? fallback.defaultTools
        newConversationAfter = c.lenient(TimeInterval.self, .newConversationAfter)
            ?? fallback.newConversationAfter
        presets = c.lenient([Lossy<AskAIPreset>].self, .presets)?.compactMap(\.value) ?? fallback.presets
        panelPosition = c.lenient(AskAIPanelPosition.self, .panelPosition)
        panelWidth = c.lenient(Double.self, .panelWidth) ?? fallback.panelWidth
        gpuPowerLimitEnabled = c.lenient(Bool.self, .gpuPowerLimitEnabled) ?? fallback.gpuPowerLimitEnabled
        gpuPowerLimitWatts = c.lenient(Int.self, .gpuPowerLimitWatts) ?? fallback.gpuPowerLimitWatts
        talk = c.lenient(TalkPrefs.self, .talk) ?? fallback.talk
        sendOnEnter = c.lenient(Bool.self, .sendOnEnter) ?? fallback.sendOnEnter
        promptDefaultsApplied = c.lenient(Bool.self, .promptDefaultsApplied) ?? false
        if !promptDefaultsApplied {
            if defaultSystemPrompt.isEmpty { defaultSystemPrompt = Self.defaultPrompt }
            if talk.systemPrompt.isEmpty { talk.systemPrompt = TalkPrefs.defaultPrompt }
            promptDefaultsApplied = true
        }
    }

    var requestGPUPowerLimit: Int? {
        Self.requestGPUPowerLimit(enabled: gpuPowerLimitEnabled, watts: gpuPowerLimitWatts)
    }

    /// The `gpu_power_limit_w` to send, clamped to what the server accepts.
    static func requestGPUPowerLimit(enabled: Bool, watts: Int) -> Int? {
        enabled ? min(max(watts, 2), 1000) : nil
    }

    var defaultProfile: AskAIProfile {
        AskAIProfile(model: defaultModel, systemPrompt: defaultSystemPrompt,
                     allowThinking: defaultAllowThinking, autoCopy: defaultAutoCopy,
                     tools: defaultTools)
    }
}

/// Decodes one array element, or nil when that element is malformed.
private struct Lossy<Value: Decodable>: Decodable {
    let value: Value?

    init(from decoder: Decoder) throws {
        value = try? Value(from: decoder)
    }
}

private extension KeyedDecodingContainer {
    /// The value for `key`, or nil when it is missing or has the wrong shape.
    func lenient<T: Decodable>(_ type: T.Type, _ key: Key) -> T? {
        (try? decodeIfPresent(type, forKey: key)) ?? nil
    }
}

// MARK: - Slash commands

/// What the panel should do with the typed text.
enum AskAIInput: Equatable {
    case plain(String)
    case preset(AskAIPreset, body: String)
    case unknownCommand(String)

    /// A leading `/name` token selects a preset. Text such as `/usr/bin ...`
    /// is plain because `/` is not a valid command character.
    static func parse(_ text: String, presets: [AskAIPreset]) -> AskAIInput {
        guard let (command, body) = splitCommand(text) else {
            return .plain(text.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        if let preset = presets.first(where: { $0.command == command }) {
            return .preset(preset, body: body)
        }
        return .unknownCommand(command)
    }

    /// Presets matching a command that is still being typed (no space yet).
    static func suggestions(for text: String, presets: [AskAIPreset]) -> [AskAIPreset] {
        guard text.hasPrefix("/"), !text.contains(where: \.isWhitespace) else { return [] }
        let typed = text.dropFirst().lowercased()
        guard typed.allSatisfy(isCommandCharacter) else { return [] }
        return presets
            .filter { $0.command.hasPrefix(typed) }
            .sorted { $0.command < $1.command }
    }

    private static func splitCommand(_ text: String) -> (String, String)? {
        guard text.hasPrefix("/") else { return nil }
        let afterSlash = text.dropFirst()
        let token = afterSlash.prefix { !$0.isWhitespace }
        guard !token.isEmpty, token.allSatisfy(isCommandCharacter) else { return nil }
        let body = afterSlash.dropFirst(token.count)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return (String(token).lowercased(), body)
    }

    /// Letters and digits of any script (e.g. `/번역`, `/翻訳`), `-`, and `_`.
    static func isCommandCharacter(_ ch: Character) -> Bool {
        ch.isLetter || ch.isNumber || ch == "-" || ch == "_"
    }
}

enum AskAIPresetValidation: Equatable {
    case emptyCommand
    case invalidCommand
    case duplicateCommand
    case missingModel

    static func check(_ preset: AskAIPreset, among presets: [AskAIPreset]) -> AskAIPresetValidation? {
        let command = preset.command
        if command.isEmpty { return .emptyCommand }
        guard let first = command.first, first.isLetter || first.isNumber,
              command.allSatisfy(AskAIInput.isCommandCharacter),
              command == command.lowercased()
        else { return .invalidCommand }
        if presets.contains(where: { $0.id != preset.id && $0.command == command }) {
            return .duplicateCommand
        }
        if preset.model.isEmpty { return .missingModel }
        return nil
    }

    var message: String {
        switch self {
        case .emptyCommand:
            return String(localized: "askai.preset.error.empty_command",
                          defaultValue: "Enter a command name.",
                          comment: "Validation error when a preset has no command name")
        case .invalidCommand:
            return String(localized: "askai.preset.error.invalid_command",
                          defaultValue: "Use letters, digits, - and _ only, in lowercase and without spaces.",
                          comment: "Validation error for a preset command with invalid characters")
        case .duplicateCommand:
            return String(localized: "askai.preset.error.duplicate_command",
                          defaultValue: "Another preset already uses this command.",
                          comment: "Validation error when two presets share a command")
        case .missingModel:
            return String(localized: "askai.preset.error.missing_model",
                          defaultValue: "Choose a model.",
                          comment: "Validation error when a preset has no model")
        }
    }
}

// MARK: - Store

@MainActor
@Observable
final class AskAIStore {
    var prefs: AskAIPrefs {
        didSet {
            guard prefs != oldValue else { return }
            scheduleSave()
            onChange?(oldValue, prefs)
        }
    }

    /// Called after every effective change, before the debounced save.
    @ObservationIgnored var onChange: ((AskAIPrefs, AskAIPrefs) -> Void)?

    @ObservationIgnored private let url: URL
    @ObservationIgnored private var saveTask: Task<Void, Never>?

    init(url: URL = AppConfig.appSupportURL().appendingPathComponent("ask-ai.json")) {
        self.url = url
        if let prefs = Self.read(from: url) {
            self.prefs = prefs
        } else {
            Self.backUpUnreadable(url)
            self.prefs = AskAIPrefs()
        }
    }

    /// Writes pending changes now. Called on termination.
    func flush() {
        guard saveTask != nil else { return }
        saveTask?.cancel()
        saveTask = nil
        write()
    }

    // Prompt editors bind straight to prefs, so coalesce keystrokes.
    private func scheduleSave() {
        saveTask?.cancel()
        saveTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(400))
            guard !Task.isCancelled, let self else { return }
            self.saveTask = nil
            self.write()
        }
    }

    private func write() {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(prefs) else { return }
        try? FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        try? data.write(to: url, options: [.atomic])
    }

    private static func read(from url: URL) -> AskAIPrefs? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(AskAIPrefs.self, from: data)
    }

    // The next save writes defaults over the file, so keep a copy of a file
    // that exists but is not valid JSON.
    private static func backUpUnreadable(_ url: URL) {
        let fm = FileManager.default
        guard fm.fileExists(atPath: url.path) else { return }
        let backup = url.appendingPathExtension("bak")
        try? fm.removeItem(at: backup)
        try? fm.copyItem(at: url, to: backup)
    }
}

// `{name}` tokens in Ask AI system prompts, filled in when a conversation
// starts. A prompt is filled once per conversation, so follow-up questions
// keep the same prompt text and the server can reuse its prefix cache.

import AppKit
import Foundation

enum PromptPlaceholder: String, CaseIterable, Sendable {
    case datetime, date, time, weekday, timezone, language, app, model, clipboard

    var token: String { "{\(rawValue)}" }
}

/// What a prompt is filled from. The clipboard is read only when the prompt
/// uses it, since macOS may ask the user before an app reads it.
struct PromptContext {
    var now = Date()
    var timeZone = TimeZone.current
    var language = Locale.preferredLanguages.first ?? Locale.current.identifier
    /// The app in front when Ask AI opened.
    var app: String?
    var model = ""
    var readClipboard: () -> String? = { NSPasteboard.general.string(forType: .string) }
}

enum PromptPlaceholders {
    static let clipboardLimit = 4000

    /// Replaces known tokens; other braces, such as JSON in a prompt, stay as written.
    static func fill(_ template: String, context: PromptContext) -> String {
        var out = template
        for placeholder in PromptPlaceholder.allCases where out.contains(placeholder.token) {
            out = out.replacingOccurrences(of: placeholder.token, with: value(of: placeholder, context: context))
        }
        return out
    }

    static func value(of placeholder: PromptPlaceholder, context: PromptContext) -> String {
        switch placeholder {
        case .datetime: format("yyyy-MM-dd HH:mm:ss", context)
        case .date: format("yyyy-MM-dd", context)
        case .time: format("HH:mm:ss", context)
        // English names read the same to every model.
        case .weekday: format("EEEE", context)
        case .timezone: "\(context.timeZone.identifier) (UTC\(offset(context)))"
        case .language: languageName(context.language)
        case .app: context.app ?? ""
        case .model: context.model
        case .clipboard: String((context.readClipboard() ?? "").prefix(clipboardLimit))
        }
    }

    private static func format(_ pattern: String, _ context: PromptContext) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = context.timeZone
        formatter.dateFormat = pattern
        return formatter.string(from: context.now)
    }

    private static func offset(_ context: PromptContext) -> String {
        let seconds = context.timeZone.secondsFromGMT(for: context.now)
        let minutes = abs(seconds) / 60
        return String(format: "%@%02d:%02d", seconds < 0 ? "-" : "+", minutes / 60, minutes % 60)
    }

    private static func languageName(_ identifier: String) -> String {
        let code = Locale(identifier: identifier).language.languageCode?.identifier ?? identifier
        let name = Locale(identifier: "en_US").localizedString(forLanguageCode: code) ?? code
        return "\(name) (\(identifier))"
    }
}

/// macOS asks before an app reads the general pasteboard on its own, which
/// `{clipboard}` does on every new conversation.
enum ClipboardAccess {
    /// True when reading the clipboard can show an alert or be refused.
    static var needsPermission: Bool {
        if #available(macOS 15.4, *) {
            return NSPasteboard.general.accessBehavior != .alwaysAllow
        }
        return false
    }

    static func openSystemSettings() {
        let link = "x-apple.systempreferences:com.apple.settings.PrivacySecurity.extension"
        guard let url = URL(string: link) else { return }
        NSWorkspace.shared.open(url)
    }
}

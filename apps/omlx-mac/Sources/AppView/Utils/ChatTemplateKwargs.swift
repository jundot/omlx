// PR 14 — chat-template kwargs editor model.
//
// The server stores two sibling fields on a model's settings:
//   chat_template_kwargs: dict[str, Any]    — key/value bag forwarded
//                                              to the chat template
//   forced_ct_kwargs:     list[str]         — keys whose value should
//                                              override anything the
//                                              client request also sets
//
// The HTML admin's editor (dashboard.js:990-1012, 1511-1521) renders this
// as a flat list of entries, each one of three kinds:
//
//   • enable_thinking   — bool, dropdown true/false
//   • reasoning_effort  — string, dropdown low/medium/high
//   • custom            — free-form key + value with light auto-coercion
//                         ("true"/"false" → Bool, numeric strings → Number,
//                         otherwise string)
//
// Each entry also carries a `force` checkbox that adds the entry's key
// to `forced_ct_kwargs`. The codec below is the Swift mirror of the
// JS logic so the round-trip is lossless.

import Foundation

enum ChatTemplateKwargEntryKind: Equatable, Sendable {
    case enableThinking
    case reasoningEffort
    case custom
}

struct ChatTemplateKwargEntry: Equatable, Identifiable, Sendable {
    let id: UUID
    var kind: ChatTemplateKwargEntryKind
    /// Used only when `kind == .custom`. The other kinds have a fixed
    /// server-side key (`enable_thinking`, `reasoning_effort`).
    var customKey: String
    /// String form (the editor renders all values as strings; the codec
    /// coerces on encode).
    var value: String
    /// When true, the key gets added to `forced_ct_kwargs` so this
    /// chat-template kwarg overrides whatever the request body sent.
    var force: Bool

    init(
        id: UUID = UUID(),
        kind: ChatTemplateKwargEntryKind,
        customKey: String = "",
        value: String,
        force: Bool = false
    ) {
        self.id = id
        self.kind = kind
        self.customKey = customKey
        self.value = value
        self.force = force
    }

    /// Resolved server-side key. Returns `nil` for a custom entry with
    /// an empty/whitespace-only key — those are silently dropped on
    /// encode, matching the JS check `e.key && e.key.trim()`.
    var resolvedKey: String? {
        switch kind {
        case .enableThinking:  return "enable_thinking"
        case .reasoningEffort: return "reasoning_effort"
        case .custom:
            let k = customKey.trimmingCharacters(in: .whitespaces)
            return k.isEmpty ? nil : k
        }
    }
}

/// Encode + decode bridge between the editor's flat entry list and the
/// server's `(chat_template_kwargs, forced_ct_kwargs)` pair.
enum ChatTemplateKwargsCodec {

    /// Build the server-facing pair. Returns `(nil, nil)` when the entry
    /// list is empty after dropping invalid custom entries — the patch
    /// builder uses that to know whether to send the fields at all.
    static func encode(
        _ entries: [ChatTemplateKwargEntry]
    ) -> (kwargs: [String: AnyCodable]?, forced: [String]?) {
        var dict: [String: AnyCodable] = [:]
        var forced: [String] = []

        for entry in entries {
            guard let key = entry.resolvedKey else { continue }
            let coerced = coerceValue(entry.value, for: entry.kind)
            dict[key] = coerced
            if entry.force { forced.append(key) }
        }

        return (
            kwargs: dict.isEmpty ? nil : dict,
            forced: forced.isEmpty ? nil : forced
        )
    }

    /// Reconstruct the editor's entries from a server-loaded pair. The
    /// HTML admin tracks insertion order via Object iteration; Swift
    /// dictionaries are unordered, so we render known kinds first
    /// (enable_thinking, reasoning_effort) and then the remaining
    /// keys sorted alphabetically for stability across reloads.
    static func decode(
        kwargs: [String: AnyCodable]?,
        forced: [String]?
    ) -> [ChatTemplateKwargEntry] {
        guard let kwargs, !kwargs.isEmpty else { return [] }
        let forcedSet = Set(forced ?? [])
        var out: [ChatTemplateKwargEntry] = []

        if let v = kwargs["enable_thinking"] {
            out.append(ChatTemplateKwargEntry(
                kind: .enableThinking,
                value: stringify(v),
                force: forcedSet.contains("enable_thinking")
            ))
        }
        if let v = kwargs["reasoning_effort"] {
            out.append(ChatTemplateKwargEntry(
                kind: .reasoningEffort,
                value: stringify(v),
                force: forcedSet.contains("reasoning_effort")
            ))
        }
        let known: Set<String> = ["enable_thinking", "reasoning_effort"]
        let extras = kwargs.keys.filter { !known.contains($0) }.sorted()
        for key in extras {
            guard let v = kwargs[key] else { continue }
            out.append(ChatTemplateKwargEntry(
                kind: .custom,
                customKey: key,
                value: stringify(v),
                force: forcedSet.contains(key)
            ))
        }
        return out
    }

    // MARK: - Helpers

    /// Render a `AnyCodable` back to the editor's string form. We avoid
    /// the JSONEncoder round-trip for primitives so the user's typed
    /// representation survives (e.g. `0.20` doesn't become `0.2`).
    private static func stringify(_ v: AnyCodable) -> String {
        switch v.value {
        case let s as String: return s
        case let b as Bool:   return b ? "true" : "false"
        case let i as Int:    return String(i)
        case let d as Double:
            // Keep trailing zeros stripped only if the value is integral
            // — `1.0` -> "1" reads cleaner in the field.
            if d.rounded() == d, d.isFinite, abs(d) < 1e15 {
                return String(Int64(d))
            }
            return String(d)
        default: return ""
        }
    }

    /// Mirror of dashboard.js's encode-time coercion:
    /// • enable_thinking: "true"/"false" → Bool
    /// • reasoning_effort: always a string
    /// • custom: bool literals first, then numeric, otherwise string
    private static func coerceValue(
        _ raw: String,
        for kind: ChatTemplateKwargEntryKind
    ) -> AnyCodable {
        switch kind {
        case .enableThinking:
            return AnyCodable(raw == "true")
        case .reasoningEffort:
            return AnyCodable(raw)
        case .custom:
            let t = raw.trimmingCharacters(in: .whitespaces)
            if t == "true"  { return AnyCodable(true) }
            if t == "false" { return AnyCodable(false) }
            if !t.isEmpty,
               let n = Double(t),
               !n.isNaN {
                // Prefer Int when the user's literal has no decimal/exponent.
                if !t.contains(".") && !t.contains("e") && !t.contains("E"),
                   let i = Int(t) {
                    return AnyCodable(i)
                }
                return AnyCodable(n)
            }
            return AnyCodable(raw)
        }
    }
}

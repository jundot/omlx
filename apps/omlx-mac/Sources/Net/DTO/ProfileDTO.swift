// PR 8 — Profiles + Templates DTO surface.
//
// ModelSettingsScreen's Profiles tab calls /admin/api/models/{id}/profiles
// (per-model bundles) and /admin/api/profile-templates (globally shared
// universal-field bundles). The settings dict carried inside each is a free-
// form bag that mirrors `ModelSettings`; for now we keep it as a typed
// `[String: AnyCodable]` so we don't have to round-trip every setting field
// in screen-state. Apply / save flows write to /api/models/{id}/settings.

import Foundation

struct ProfileListResponse: Codable, Sendable {
    let profiles: [ProfileDTO]
}

/// Three logical scopes the editor's chip groups split profiles into. The
/// distinction is purely client-side: built-in templates render as Preset
/// (read-only), user templates render as Global (editable, cross-model),
/// and per-model profiles render as Model (editable, this-model-only).
/// The server stores them in just two collections — `profile-templates`
/// (with an `is_builtin` flag) and `models/{id}/profiles`.
enum ProfileScope: String, Hashable, CaseIterable, Sendable {
    case preset, global, model
}

struct ProfileDTO: Codable, Equatable, Sendable, Identifiable {
    let name: String
    let displayName: String
    let description: String?
    /// Nullable: built-in global templates ship inside the bundle and have
    /// no per-instance creation moment, so the server sends `null` for both
    /// timestamps. User-created profiles and templates always have stamps.
    let createdAt: String?
    let updatedAt: String?
    let sourceTemplate: String?
    /// True for shipped built-in templates (read-only on the server side —
    /// save/update/delete with a built-in name returns 400). UI should
    /// render the Edit/Delete affordances disabled when this is true.
    let isBuiltin: Bool?
    /// Server-side `expose_as_model` — when true this per-model profile is
    /// also published as an independent model ID on /v1/models, overlaying
    /// its settings on the base model per request. Nil for templates and
    /// for servers that predate the feature.
    let exposeAsModel: Bool?
    /// Derived API model ID (`<base-model>:<profile-name>`) the exposed
    /// profile serves under. The server computes it from the profile name;
    /// sent on per-model profile responses only.
    let modelId: String?
    /// Server-derived: true when the profile overrides engine-construction
    /// fields, which the exposed-model overlay ignores. Drives the warning
    /// under the expose toggle — the server owns the field classification,
    /// so the client never mirrors the list.
    let hasEngineFields: Bool?
    /// Free-form. We re-shape on apply via `applyProfile()`, not by reading
    /// every setting individually.
    let settings: [String: AnyCodable]?

    var id: String { name }

    /// Splits a template list into preset/global by `isBuiltin`. Defaults
    /// to Global for legacy server responses where the field is missing —
    /// the server is the source of truth for builtin status, so an absent
    /// flag means "the server didn't say built-in, therefore user-managed".
    var templateScope: ProfileScope {
        (isBuiltin == true) ? .preset : .global
    }
}

struct CreateProfileRequest: Encodable, Sendable {
    let name: String
    let displayName: String
    let description: String?
    let sourceTemplate: String?
    let settings: [String: AnyCodable]?
    let alsoSaveAsTemplate: Bool
    let exposeAsModel: Bool

    enum CodingKeys: String, CodingKey {
        case name, displayName, description
        case sourceTemplate = "source_template"
        case settings
        case alsoSaveAsTemplate = "also_save_as_template"
        case exposeAsModel = "expose_as_model"
    }

    init(
        name: String,
        displayName: String,
        description: String? = nil,
        sourceTemplate: String? = nil,
        settings: [String: AnyCodable]? = nil,
        alsoSaveAsTemplate: Bool = false,
        exposeAsModel: Bool = false
    ) {
        self.name = name
        self.displayName = displayName
        self.description = description
        self.sourceTemplate = sourceTemplate
        self.settings = settings
        self.alsoSaveAsTemplate = alsoSaveAsTemplate
        self.exposeAsModel = exposeAsModel
    }
}

struct CreateProfileResponse: Decodable, Sendable {
    let profile: ProfileDTO
}

struct ApplyProfileResponse: Decodable, Sendable {
    let modelId: String
    let settings: [String: AnyCodable]
}

/// Body for PUT /admin/api/models/{id}/profiles/{name}. Mirrors the Python
/// `UpdateProfileRequest` — every field is optional so the server merges
/// instead of resetting. The Profiles tab's "Update <name>" action uses
/// just `settings` to push the working profile into an existing slot.
struct UpdateProfileRequest: Encodable, Sendable {
    let newName: String?
    let displayName: String?
    let description: String?
    let settings: [String: AnyCodable]?
    let sourceTemplate: String?
    let alsoSaveAsTemplate: Bool
    /// Nil is omitted from the body — the server treats an absent flag as
    /// "don't touch", so settings-only updates never clobber exposure
    /// state set elsewhere (e.g. the web dashboard).
    let exposeAsModel: Bool?

    enum CodingKeys: String, CodingKey {
        case newName = "new_name"
        case displayName = "display_name"
        case description
        case settings
        case sourceTemplate = "source_template"
        case alsoSaveAsTemplate = "also_save_as_template"
        case exposeAsModel = "expose_as_model"
    }

    init(
        newName: String? = nil,
        displayName: String? = nil,
        description: String? = nil,
        settings: [String: AnyCodable]? = nil,
        sourceTemplate: String? = nil,
        alsoSaveAsTemplate: Bool = false,
        exposeAsModel: Bool? = nil
    ) {
        self.newName = newName
        self.displayName = displayName
        self.description = description
        self.settings = settings
        self.sourceTemplate = sourceTemplate
        self.alsoSaveAsTemplate = alsoSaveAsTemplate
        self.exposeAsModel = exposeAsModel
    }
}

struct UpdateProfileResponse: Decodable, Sendable {
    let profile: ProfileDTO
}

/// Body for PUT /admin/api/profile-templates/{name}. Same merge semantics
/// as `UpdateProfileRequest`.
struct UpdateTemplateRequest: Encodable, Sendable {
    let newName: String?
    let displayName: String?
    let description: String?
    let settings: [String: AnyCodable]?

    enum CodingKeys: String, CodingKey {
        case newName = "new_name"
        case displayName = "display_name"
        case description
        case settings
    }

    init(
        newName: String? = nil,
        displayName: String? = nil,
        description: String? = nil,
        settings: [String: AnyCodable]? = nil
    ) {
        self.newName = newName
        self.displayName = displayName
        self.description = description
        self.settings = settings
    }
}

struct UpdateTemplateResponse: Decodable, Sendable {
    let template: ProfileDTO
}

struct DeleteResponse: Decodable, Sendable {
    let deleted: Bool
    let name: String?
}

// MARK: - Templates

struct TemplateListResponse: Codable, Sendable {
    let templates: [ProfileDTO]
}

struct CreateTemplateRequest: Encodable, Sendable {
    let name: String
    let displayName: String
    let description: String?
    let settings: [String: AnyCodable]?
}

struct CreateTemplateResponse: Decodable, Sendable {
    let template: ProfileDTO
}

// MARK: - AnyCodable bridge

/// Loose-typed JSON value used to round-trip the settings dict inside a
/// profile without enumerating every concrete type. Sufficient for PR 8's
/// needs (passing the dict back to the server unchanged).
struct AnyCodable: Codable, Equatable, Sendable {
    let value: any Sendable

    init(_ value: any Sendable) { self.value = value }

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() {
            self.value = NSNull()
        } else if let bool = try? c.decode(Bool.self) {
            self.value = bool
        } else if let int = try? c.decode(Int.self) {
            self.value = int
        } else if let double = try? c.decode(Double.self) {
            self.value = double
        } else if let string = try? c.decode(String.self) {
            self.value = string
        } else if let array = try? c.decode([AnyCodable].self) {
            self.value = array
        } else if let dict = try? c.decode([String: AnyCodable].self) {
            self.value = dict
        } else {
            throw DecodingError.dataCorruptedError(
                in: c, debugDescription: "Unsupported AnyCodable value"
            )
        }
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch value {
        case is NSNull:               try c.encodeNil()
        case let v as Bool:            try c.encode(v)
        case let v as Int:             try c.encode(v)
        case let v as Double:          try c.encode(v)
        case let v as String:          try c.encode(v)
        case let v as [AnyCodable]:    try c.encode(v)
        case let v as [String: AnyCodable]: try c.encode(v)
        default:
            throw EncodingError.invalidValue(
                value,
                EncodingError.Context(
                    codingPath: encoder.codingPath,
                    debugDescription: "Unsupported AnyCodable value"
                )
            )
        }
    }

    static func == (lhs: AnyCodable, rhs: AnyCodable) -> Bool {
        switch (lhs.value, rhs.value) {
        case (is NSNull, is NSNull): return true
        case let (l as Bool,   r as Bool):   return l == r
        case let (l as Int,    r as Int):    return l == r
        case let (l as Double, r as Double): return l == r
        case let (l as String, r as String): return l == r
        case let (l as [AnyCodable], r as [AnyCodable]): return l == r
        case let (l as [String: AnyCodable], r as [String: AnyCodable]): return l == r
        default: return false
        }
    }
}

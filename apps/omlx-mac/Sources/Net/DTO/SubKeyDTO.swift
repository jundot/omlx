// PR 9 — sub-key + setup-api-key payloads (Security screen).
//
// SubKey CRUD is via /admin/api/sub-keys (POST + DELETE). DELETE is a
// non-standard "DELETE with body" — the server reads `key` from the request
// body, not the URL. OMLXClient gets a tiny `deleteWithBody` overload to
// support that without leaking the path quirk into screen code.

import Foundation

struct SubKeyDTO: Codable, Equatable, Sendable, Identifiable {
    let key: String
    let name: String
    let createdAt: String

    var id: String { key }
}

struct CreateSubKeyRequest: Encodable, Sendable {
    let key: String
    let name: String
}

struct CreateSubKeyResponse: Decodable, Sendable {
    let success: Bool
    let subKey: SubKeyDTO?
}

struct DeleteSubKeyRequest: Encodable, Sendable {
    let key: String
}

struct SetupApiKeyRequest: Encodable, Sendable {
    let apiKey: String
    let apiKeyConfirm: String
}

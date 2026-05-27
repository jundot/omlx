// PR 7 — GET /admin/api/server-info. Used by ServerScreen to render the
// API Endpoints CodeChips and (later) by the Welcome wizard.

import Foundation

struct ServerInfoDTO: Codable, Equatable, Sendable {
    let host: String
    let port: Int
    let aliases: [String]
}

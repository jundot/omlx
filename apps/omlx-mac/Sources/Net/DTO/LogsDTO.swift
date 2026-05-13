// PR 7 — GET /admin/api/logs.

import Foundation

struct LogsDTO: Codable, Equatable, Sendable {
    let logs: String
    let totalLines: Int
    let logFile: String
    let availableFiles: [String]
}

/// Response shape for DELETE /admin/api/logs. Lists which rotated files
/// the server removed; `deletedCount` is the same as `deleted.count` but
/// kept distinct so the server can grow features (e.g. retain-summary)
/// without breaking the contract.
struct DeleteLogsResponse: Decodable, Sendable {
    let deleted: [String]
    let deletedCount: Int
}

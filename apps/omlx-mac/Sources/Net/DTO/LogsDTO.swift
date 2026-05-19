// PR 7 — GET /admin/api/logs.

import Foundation

struct LogsDTO: Codable, Equatable, Sendable {
    let logs: String
    let totalLines: Int
    let logFile: String
    let availableFiles: [String]
}

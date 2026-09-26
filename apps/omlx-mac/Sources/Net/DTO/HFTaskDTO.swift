// PR 8 — HuggingFace download task + recommended-models surface.

import Foundation

struct HFTaskListResponse: Codable, Sendable {
    let tasks: [HFTaskDTO]
}

struct HFTaskDTO: Codable, Equatable, Sendable, Identifiable {
    let taskId: String
    let repoId: String
    let status: String
    let progress: Double
    let totalSize: Int64
    let downloadedSize: Int64
    /// Transfer rate in bytes/second. 0 whenever no bytes are landing —
    /// cancelled/failed/stalled/finished tasks reset it server-side, and
    /// the meter reads 0 in between samples with no growth; absent only
    /// when talking to a server that predates the field.
    let speedBps: Double?
    let error: String
    let createdAt: Double
    let startedAt: Double
    let completedAt: Double
    let retryCount: Int

    var id: String { taskId }

    /// Human-readable transfer rate, e.g. "43.2 MB/s"; nil only for a
    /// server that predates the speed field. A stopped task reads an
    /// explicit "0 B/s" so an interrupted row shows a visible zero
    /// instead of dropping the readout.
    var speedText: String? {
        guard let bps = speedBps else { return nil }
        var value = bps
        let units = ["B/s", "KB/s", "MB/s", "GB/s", "TB/s"]
        var unit = 0
        while value >= 1024 && unit < units.count - 1 {
            value /= 1024
            unit += 1
        }
        if unit == 0 {
            return String(format: "%.0f %@", value, units[unit])
        }
        return String(format: value >= 100 ? "%.0f %@" : "%.1f %@", value, units[unit])
    }

    enum Status: String {
        case pending, downloading, completed, failed, cancelled, paused
    }

    var statusEnum: Status? { Status(rawValue: status) }
    var isActive: Bool {
        statusEnum == .pending || statusEnum == .downloading
    }
}

struct StartHFDownloadRequest: Encodable, Sendable {
    let repoId: String
    let hfToken: String
}

struct StartHFDownloadResponse: Decodable, Sendable {
    let success: Bool
    let task: HFTaskDTO?
}

// GET /admin/api/hf/recommended returns two parallel lists (mirrors
// hf_downloader.py:237-240). The dashboard JS paginates them separately;
// the Swift screen merges them into a single deduped list ordered
// trending-first.
struct HFRecommendedResponse: Codable, Sendable {
    let trending: [HFModelInfo]
    let popular: [HFModelInfo]
}

/// Response shape for GET /admin/api/hf/search?q=<query>. Backed by the same
/// HFModelInfo rows that /recommended returns, plus an optional `total`
/// count that the server populates when paginating.
struct HFSearchResponse: Codable, Sendable {
    let models: [HFModelInfo]
    let total: Int?
}

struct HFModelInfo: Codable, Equatable, Sendable, Identifiable {
    let repoId: String
    let name: String?
    let downloads: Int?
    let likes: Int?
    let trendingScore: Double?
    let size: Int64?
    let sizeFormatted: String?
    let params: Int64?
    let paramsFormatted: String?

    var id: String { repoId }
}

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
    /// Transfer rate in bytes/second; nil for servers without the field.
    let speedBps: Double?
    /// Seconds left at the server's one-minute average rate; nil when unknown.
    let etaSeconds: Int?
    let error: String
    let createdAt: Double
    let startedAt: Double
    let completedAt: Double
    let retryCount: Int
    /// Repo paths the task fetches; nil for the whole repo.
    let files: [String]?
    /// The GGUF model files among `files`, as the server groups them.
    let ggufFiles: [String]?

    var id: String { taskId }

    /// Formatted transfer rate, e.g. "43.2 MB/s".
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

    /// Formatted remaining time, e.g. "2h 30m".
    var etaText: String? {
        guard let seconds = etaSeconds else { return nil }
        let h = seconds / 3600, m = seconds % 3600 / 60, s = seconds % 60
        if h > 0 { return "\(h)h \(m)m" }
        if m > 0 { return "\(m)m \(s)s" }
        return "\(s)s"
    }

    /// The GGUF model files a selection fetches, shown beside the repo.
    var ggufFilesText: String? {
        guard let names = ggufFiles, !names.isEmpty else { return nil }
        return names.joined(separator: ", ")
    }

    enum Status: String {
        case pending, downloading, completed, failed, cancelled, paused
    }

    var statusEnum: Status? { Status(rawValue: status) }
    var isActive: Bool {
        statusEnum == .pending || statusEnum == .downloading
    }
}

/// The Hugging Face downloader's repo filter for search and suggestions.
enum HFModelFormat: String, CaseIterable, Hashable, Sendable {
    case mlx, gguf, all
}

struct StartHFDownloadRequest: Encodable, Sendable {
    let repoId: String
    let hfToken: String
    let files: [String]?
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
    /// GGUF architecture from the Hub metadata, e.g. "qwen35".
    let architecture: String?

    var id: String { repoId }
}

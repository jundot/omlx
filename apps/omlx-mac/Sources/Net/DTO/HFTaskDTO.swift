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
    let error: String
    let createdAt: Double
    let startedAt: Double
    let completedAt: Double
    let retryCount: Int

    var id: String { taskId }

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

struct HFRecommendedResponse: Codable, Sendable {
    let models: [HFModelInfo]
}

struct HFModelInfo: Codable, Equatable, Sendable, Identifiable {
    let repoId: String
    let downloads: Int?
    let likes: Int?
    let lastModified: String?
    let totalParams: Int64?
    let totalParamsFormatted: String?
    let estimatedSize: Int64?
    let estimatedSizeFormatted: String?
    let pipelineTag: String?

    var id: String { repoId }
}

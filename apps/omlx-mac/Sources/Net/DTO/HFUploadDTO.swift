// PR 13 — HF Uploader DTOs.
//
// Mirrors omlx/admin/hf_uploader.py task to_dict shape and the request/
// response models in omlx/admin/routes.py (HFUploadRequest @288,
// HFValidateTokenRequest @312, /api/upload/* routes @4994-5097).
//
// Used by the Upload sheet + Upload Tasks section inside
// QuantizationScreen — the HTML admin panel exposes the same surface as
// a standalone "Uploader" tab; the native app folds it into the
// post-quantization flow.

import Foundation

// MARK: - Token validation

struct HFValidateTokenRequest: Encodable, Sendable {
    let hfToken: String
}

struct HFOrgInfo: Codable, Equatable, Sendable, Identifiable {
    let name: String
    var id: String { name }
}

struct HFValidateTokenResponse: Codable, Sendable {
    let username: String
    let orgs: [HFOrgInfo]
}

// MARK: - Listing local models for upload

/// One row in the upload-candidate list. `hasReadme` lets the modal hide
/// the auto-README toggle when the source bundle already ships one.
struct HFUploadModelInfo: Codable, Equatable, Sendable, Identifiable {
    let path: String
    let name: String
    let sizeFormatted: String
    let hasReadme: Bool

    var id: String { path }
}

struct HFUploadModelsResponse: Codable, Sendable {
    let oqModels: [HFUploadModelInfo]
    let allModels: [HFUploadModelInfo]
}

// MARK: - Start upload

/// Body for `POST /admin/api/upload/start`. Field names map to
/// `HFUploadRequest` (routes.py:300) via the client's
/// `convertToSnakeCase` encoder.
///
/// Note: `private` is a Swift keyword — we backtick-escape it so the
/// JSON key still encodes to `private`.
struct HFUploadStartRequest: Encodable, Sendable {
    let modelPath: String
    let repoId: String
    let hfToken: String
    let readmeSourcePath: String
    let autoReadme: Bool
    let redownloadNotice: Bool
    let `private`: Bool
}

struct HFUploadStartResponse: Decodable, Sendable {
    let success: Bool
    let task: HFUploadTaskDTO?
}

// MARK: - Task

/// Mirrors the task `to_dict` shape on the server side. `repoUrl` only
/// populates on completion; while uploading the server emits an empty
/// string, so the Swift screen treats `!repoUrl.isEmpty` as "ready to
/// open in browser".
struct HFUploadTaskDTO: Codable, Equatable, Sendable, Identifiable {
    let taskId: String
    let modelName: String
    let modelPath: String
    let repoId: String
    let status: String
    let progress: Double
    let error: String
    let createdAt: Double
    let startedAt: Double
    let completedAt: Double
    let totalSize: Int64
    let totalSizeFormatted: String
    let repoUrl: String

    var id: String { taskId }

    enum Status: String {
        case pending, uploading, completed, failed, cancelled
    }

    var statusEnum: Status? { Status(rawValue: status) }

    /// True while the task holds the uploader's semaphore. Used to drive
    /// the 2 s / 6 s polling cadence on QuantizationScreenVM.
    var isActive: Bool {
        statusEnum == .pending || statusEnum == .uploading
    }
}

struct HFUploadTasksResponse: Codable, Sendable {
    let tasks: [HFUploadTaskDTO]
}

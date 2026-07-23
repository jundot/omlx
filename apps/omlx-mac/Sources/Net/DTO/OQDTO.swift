// PR 12 — oQ Quantization DTOs.
//
// Mirrors omlx/admin/oq_manager.py:QuantTask.to_dict (line 79-98) and the
// /admin/api/oq/* responses defined in omlx/admin/routes.py:4881-4992.
//
// The client uses `convertFromSnakeCase` so snake_case server keys map to
// camelCase Swift fields automatically (e.g. `output_size_formatted` →
// `outputSizeFormatted`).

import Foundation

// MARK: - Model listing

/// One row in the source-model list. Quantizable rows come from
/// `source_models` (added `numLayers`, `numExperts`, `memoryStreaming`); the
/// sensitivity picker pulls from `allModels` (no extra fields needed).
struct OQModelInfo: Codable, Equatable, Sendable, Identifiable {
    let path: String
    let name: String
    let size: Int64
    let sizeFormatted: String
    let modelType: String
    let isQuantized: Bool
    let isVlm: Bool
    let hasMtpHeads: Bool

    // Only populated on `source_models` rows.
    let numLayers: Int?
    let numExperts: Int?
    let memoryStreaming: OQMemoryEstimate?

    var id: String { path }
}

/// Streaming-memory estimate attached to each source model (oq.py:1427).
struct OQMemoryEstimate: Codable, Equatable, Sendable {
    let peakBytes: Int64
    let peakFormatted: String
}

struct OQModelsResponse: Codable, Sendable {
    let models: [OQModelInfo]
    let allModels: [OQModelInfo]
}

// MARK: - Estimate

/// Response from `GET /admin/api/oq/estimate`. The Python side may also
/// include `memory_streaming_formatted` so we surface it as optional.
struct OQEstimateResponse: Codable, Equatable, Sendable {
    let effectiveBpw: Double
    let outputSizeBytes: Int64
    let outputSizeFormatted: String
    let memoryStreamingFormatted: String?
}

// MARK: - Start quantization

/// Body for `POST /admin/api/oq/start`. Field names match
/// `OQStartRequest` (routes.py:288) and rely on the client's
/// `convertToSnakeCase` encoder.
struct OQStartRequest: Encodable, Sendable {
    let modelPath: String
    let oqLevel: Double
    let groupSize: Int
    let sensitivityModelPath: String
    let textOnly: Bool
    let dtype: String
    let preserveMtp: Bool
    let enhanced: Bool
    let imatrixCachePath: String
    let imatrixReuseCache: Bool
    let imatrixStrict: Bool
}

struct OQStartResponse: Decodable, Sendable {
    let success: Bool
    let task: OQTaskDTO?
}

// MARK: - Tasks

/// Mirrors `QuantTask.to_dict` (oq_manager.py:79). The `status` field is a
/// string in the JSON; we expose `statusEnum` for callers that switch on it.
struct OQTaskDTO: Codable, Equatable, Sendable, Identifiable {
    let taskId: String
    let modelName: String
    let modelPath: String
    let oqLevel: Double
    let outputName: String
    let outputPath: String
    let status: String
    let progress: Double
    let phase: String
    let error: String
    let createdAt: Double
    let startedAt: Double
    let completedAt: Double
    let sourceSize: Int64
    let outputSize: Int64
    let dtype: String

    var id: String { taskId }

    enum Status: String {
        case pending, loading, quantizing, saving, completed, failed, cancelled
    }

    var statusEnum: Status? { Status(rawValue: status) }

    /// True while the task occupies the quantization slot (and the screen
    /// should keep polling).
    var isActive: Bool {
        switch statusEnum {
        case .pending, .loading, .quantizing, .saving: return true
        default: return false
        }
    }
}

struct OQTasksResponse: Codable, Sendable {
    let tasks: [OQTaskDTO]
}

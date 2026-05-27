// PR 13 — Benchmark DTOs (throughput + accuracy).
//
// Throughput mirrors omlx/admin/routes.py:/api/bench/* (start @4619,
// stream @4679, cancel @4724, results @4748). Accuracy mirrors
// /api/bench/accuracy/* (queue/add @4460, queue/status @4507, results
// @4531, reset @4547, cancel @4558, stream @4569).
//
// Both feature areas use *polling* in the Swift screens (the HTML uses
// SSE). The shape of `GET /api/bench/{id}/results` and
// `GET /api/bench/accuracy/results` matches what each respective SSE
// stream emits once aggregated, so polling produces functionally
// equivalent state — just without per-token granularity.

import Foundation

// =============================================================================
// MARK: - Device info (shared by both screens)
// =============================================================================

/// Response from `GET /admin/api/device-info`. Surfaced as a small chip
/// at the top of the Throughput screen so the user can sanity-check
/// which hardware they're benchmarking.
struct DeviceInfoDTO: Codable, Sendable {
    let chipName: String?
    let chipVariant: String?
    let memoryGb: Int?
    let gpuCores: Int?
    let ownerHash: String?
}

// =============================================================================
// MARK: - Throughput bench
// =============================================================================

/// Body for `POST /admin/api/bench/start`. `prompt_lengths` and
/// `batch_sizes` are server-validated against a known whitelist
/// (1024…200000 / 2…8). `generation_length` is free-form int.
///
/// The server always publishes results to the public omlx.ai
/// leaderboard after the bench completes (matching the browser admin
/// panel). Submission is anonymous — the payload carries an
/// owner_hash derived from hardware fingerprint, not user identity.
struct BenchStartRequest: Encodable, Sendable {
    let modelId: String
    let promptLengths: [Int]
    let generationLength: Int
    let batchSizes: [Int]
}

struct BenchStartResponse: Codable, Sendable {
    let benchId: String
    let status: String
    let totalTests: Int
}

/// Single + batch results share one envelope; `testType` is the
/// discriminator. We use optional fields so a single struct can decode
/// both variants without resorting to an enum-with-associated-values
/// (which would have to be hand-coded for Codable).
struct BenchResultDTO: Codable, Equatable, Sendable {
    let testType: String

    // Single-only
    let pp: Int?
    let tg: Int?
    let ttftMs: Double?
    let tpotMs: Double?
    let processingTps: Double?
    let genTps: Double?
    let e2eLatencyS: Double?
    let totalThroughput: Double?
    let peakMemoryBytes: Int64?

    // Batch-only
    let batchSize: Int?
    let tgTps: Double?
    let ppTps: Double?
    let avgTtftMs: Double?
}

struct BenchResultsResponse: Codable, Sendable {
    let benchId: String
    let status: String
    let results: [BenchResultDTO]
    let error: String?
    /// Mirror of the SSE `upload` / `upload_done` / `upload_skipped` events
    /// (omlx/admin/benchmark.py:_upload_to_omlx_ai). Populated server-side
    /// as the upload progresses so polling clients render the same state
    /// the HTML admin panel sees over its event stream.
    let uploadState: BenchUploadStateDTO?
}

/// Per-run upload state. Lives on `BenchmarkRun.upload_state` server-side.
struct BenchUploadStateDTO: Codable, Equatable, Sendable {
    /// "idle" | "uploading" | "done" | "skipped"
    let phase: String
    let results: [BenchUploadResultDTO]
    let total: Int
    let successCount: Int
    let failedCount: Int
    /// Display owner hash (verify char stripped). Populated on phase=done.
    let ownerHash: String?
    /// Set when phase=skipped. Today the only reason is
    /// "experimental_features".
    let skippedReason: String?
    let skippedFeatures: [String]
}

/// One context-length's upload outcome. Exactly one of `url` / `error`
/// should be populated; `duplicate=true` flags rows the server already had.
///
/// The JSON `id` field from the server (a submission UUID, nullable on
/// errors) is renamed to `submissionId` so the SwiftUI `Identifiable`
/// conformance can use `contextLength` as a stable, non-optional key.
struct BenchUploadResultDTO: Codable, Equatable, Sendable, Identifiable {
    let contextLength: Int
    let submissionId: String?
    let url: String?
    let duplicate: Bool?
    let error: String?

    var id: Int { contextLength }

    // The client's JSONDecoder uses convertFromSnakeCase, which transforms
    // JSON keys *before* matching against CodingKey raw values. So we only
    // declare a custom raw value for `submissionId`, whose JSON name (`id`)
    // can't be derived from snake_case conversion. The rest use synthesized
    // raw values (`contextLength`, etc.) that the strategy produces from
    // the server's snake_case keys.
    enum CodingKeys: String, CodingKey {
        case contextLength
        case submissionId = "id"
        case url
        case duplicate
        case error
    }
}

struct BenchCancelResponse: Codable, Sendable {
    let status: String
    let benchId: String?
}

// =============================================================================
// MARK: - Accuracy bench
// =============================================================================

/// Body for `POST /admin/api/bench/accuracy/queue/add`. `benchmarks` is
/// a dict of benchmark-key → sample-size; the catalog of valid keys
/// lives in AccuracyBenchScreen.swift to keep the DTO server-agnostic.
struct AccuracyQueueAddRequest: Encodable, Sendable {
    let modelId: String
    let benchmarks: [String: Int]
    let batchSize: Int
    let enableThinking: Bool
}

struct AccuracyQueueItem: Codable, Equatable, Sendable {
    let modelId: String
    let benchmarks: [String]
}

/// Mirrors the queue-status snapshot returned by every accuracy
/// queue/add, queue/status, queue/remove endpoint. `lastProgress` is
/// the most recent progress event the SSE stream emitted; polling
/// callers use it to drive the in-flight progress message.
///
/// `phase` is finer-grained than `running` — it distinguishes
/// "evaluating" (still scoring) from "unloading" (post-result cleanup),
/// so the UI can hide the running row once the result card has
/// appeared. Values: "pending" | "loading" | "evaluating" |
/// "unloading" | "completed" | "cancelled" | "error". Nullable when no
/// run is in flight.
struct AccuracyQueueStatus: Codable, Sendable {
    let running: Bool
    let currentModel: String
    let currentBenchId: String
    let lastProgress: AccuracyProgressDTO?
    let phase: String?
    let queue: [AccuracyQueueItem]

    /// True only while the run is actively producing results. Hides the
    /// running row during post-result unload and after the bench task
    /// finishes but before `_queue_running` flips back.
    var isActivelyEvaluating: Bool {
        guard running else { return false }
        switch phase {
        case "evaluating", "loading", "pending", nil: return true
        default: return false
        }
    }
}

struct AccuracyProgressDTO: Codable, Equatable, Sendable {
    let modelId: String?
    let message: String?
    let current: Int?
    let total: Int?
    let benchCurrent: Int?
    let benchTotal: Int?
    let benchmark: String?
}

struct AccuracyResultDTO: Codable, Equatable, Sendable, Identifiable {
    let benchmark: String
    let modelId: String
    let accuracy: Double
    let correct: Int
    let total: Int
    let timeS: Double
    let thinkingUsed: Bool
    let categoryScores: [String: Double]?

    /// Synthetic ID — the server doesn't emit one and `(benchmark,
    /// model)` is unique within an accAllResults array.
    var id: String { "\(benchmark)::\(modelId)" }
}

struct AccuracyResultsResponse: Codable, Sendable {
    let results: [AccuracyResultDTO]
    let running: Bool
    let currentModel: String
    let currentBenchId: String
}

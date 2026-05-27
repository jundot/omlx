// PR 7 — typed wrappers + URL constants for the /admin/api/* surface used by
// PR 7 screens. Kept separate from `OMLXClient.swift` so PR 8/9 can grow this
// file without churning the client itself.
//
// The actual `getX()` methods live as extensions on `OMLXClient` (see
// `OMLXClient.swift`). This file is the canonical list of paths so future
// fixture-capture scripts (`Scripts/capture_fixtures.sh`) can read them.

import Foundation

enum AdminAPI {
    static let prefix = "/admin/api"

    static let login           = "\(prefix)/login"
    static let globalSettings  = "\(prefix)/global-settings"
    static let serverInfo      = "\(prefix)/server-info"
    static let stats           = "\(prefix)/stats"
    static let statsClear      = "\(prefix)/stats/clear"
    static let statsClearAlltime = "\(prefix)/stats/clear-alltime"
    static let ssdCacheClear   = "\(prefix)/ssd-cache/clear"
    static let hotCacheClear   = "\(prefix)/hot-cache/clear"
    static let logs            = "\(prefix)/logs"

    static let models          = "\(prefix)/models"
    static func loadModel(_ id: String) -> String   { "\(models)/\(id)/load" }
    static func unloadModel(_ id: String) -> String { "\(models)/\(id)/unload" }
    static func modelSettings(_ id: String) -> String { "\(models)/\(id)/settings" }
    static let reloadModels    = "\(prefix)/reload"

    static func modelProfiles(_ id: String) -> String { "\(models)/\(id)/profiles" }
    static func modelProfile(_ id: String, _ name: String) -> String {
        "\(models)/\(id)/profiles/\(name)"
    }
    static func applyModelProfile(_ id: String, _ name: String) -> String {
        "\(models)/\(id)/profiles/\(name)/apply"
    }
    static let profileTemplates = "\(prefix)/profile-templates"
    static func profileTemplate(_ name: String) -> String {
        "\(profileTemplates)/\(name)"
    }
    /// Proxy that fetches the remote preset bundle from omlx.ai and returns
    /// the parsed JSON. Mirrors HTML's `refreshPresets()` flow — clients
    /// cache the result locally and use it as the read-only preset source.
    static let presetsRefresh  = "\(prefix)/presets/refresh"

    static let hfTasks         = "\(prefix)/hf/tasks"
    static let hfDownload      = "\(prefix)/hf/download"
    static func hfCancel(_ taskId: String) -> String { "\(prefix)/hf/cancel/\(taskId)" }
    static func hfRetry(_ taskId: String) -> String  { "\(prefix)/hf/retry/\(taskId)" }
    static func hfTask(_ taskId: String) -> String   { "\(prefix)/hf/task/\(taskId)" }
    static let hfRecommended   = "\(prefix)/hf/recommended"
    static let hfSearch        = "\(prefix)/hf/search"
    static func hfModel(_ name: String) -> String { "\(prefix)/hf/models/\(name)" }

    // Phase 2 — ModelScope downloader (mirrors the /hf/* surface 1:1, just
    // pointed at the parallel pipeline backed by ms_downloader.py).
    static let msStatus        = "\(prefix)/ms/status"
    static let msTasks         = "\(prefix)/ms/tasks"
    static let msDownload      = "\(prefix)/ms/download"
    static func msCancel(_ taskId: String) -> String { "\(prefix)/ms/cancel/\(taskId)" }
    static func msRetry(_ taskId: String) -> String  { "\(prefix)/ms/retry/\(taskId)" }
    static func msTask(_ taskId: String) -> String   { "\(prefix)/ms/task/\(taskId)" }
    static let msRecommended   = "\(prefix)/ms/recommended"
    static let msSearch        = "\(prefix)/ms/search"

    // PR 9
    static let setupApiKey     = "\(prefix)/setup-api-key"
    static let subKeys         = "\(prefix)/sub-keys"

    // PR 12 — oQ Quantization
    static let oqModels        = "\(prefix)/oq/models"
    static let oqEstimate      = "\(prefix)/oq/estimate"
    static let oqStart         = "\(prefix)/oq/start"
    static let oqTasks         = "\(prefix)/oq/tasks"
    static func oqCancel(_ taskId: String) -> String { "\(prefix)/oq/cancel/\(taskId)" }
    static func oqTask(_ taskId: String) -> String   { "\(prefix)/oq/task/\(taskId)" }

    // PR 13 — HF Upload
    static let uploadValidateToken = "\(prefix)/upload/validate-token"
    static let uploadModels        = "\(prefix)/upload/oq-models"
    static let uploadStart         = "\(prefix)/upload/start"
    static let uploadTasks         = "\(prefix)/upload/tasks"
    static func uploadCancel(_ taskId: String) -> String { "\(prefix)/upload/cancel/\(taskId)" }
    static func uploadTask(_ taskId: String) -> String   { "\(prefix)/upload/task/\(taskId)" }

    // PR 13 — Throughput bench
    static let deviceInfo      = "\(prefix)/device-info"
    static let benchStart      = "\(prefix)/bench/start"
    static func benchResults(_ benchId: String) -> String { "\(prefix)/bench/\(benchId)/results" }
    static func benchCancel(_ benchId: String) -> String  { "\(prefix)/bench/\(benchId)/cancel" }

    // PR 13 — Accuracy bench
    static let accuracyQueueAdd    = "\(prefix)/bench/accuracy/queue/add"
    static let accuracyQueueStatus = "\(prefix)/bench/accuracy/queue/status"
    static func accuracyQueueRemove(_ idx: Int) -> String { "\(prefix)/bench/accuracy/queue/\(idx)" }
    static let accuracyResults     = "\(prefix)/bench/accuracy/results"
    static let accuracyReset       = "\(prefix)/bench/accuracy/results/reset"
    static let accuracyCancel      = "\(prefix)/bench/accuracy/cancel"
}

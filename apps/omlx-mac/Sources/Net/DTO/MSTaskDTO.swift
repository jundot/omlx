// Phase 2 — ModelScope download task surface.
//
// The server's MS downloader (`omlx/admin/ms_downloader.py:19`) imports
// `DownloadTask` directly from `hf_downloader.py`, so the wire shapes for
// /ms/tasks, /ms/recommended and /ms/search are byte-for-byte identical to
// their /hf/ counterparts. Modeling them as typealiases keeps the active-
// downloads / completed-tasks / suggested-models view code generic across
// sources — only the form + mirror editor differ.
//
// Only the start-download request shape genuinely differs: the server's
// `MSDownloadRequest` (omlx/admin/routes.py:287-291) uses `model_id` /
// `ms_token` field names instead of HF's `repo_id` / `hf_token`.

import Foundation

typealias MSTaskDTO              = HFTaskDTO
typealias MSTaskListResponse     = HFTaskListResponse
typealias MSModelInfo            = HFModelInfo
typealias MSSearchResponse       = HFSearchResponse
typealias MSRecommendedResponse  = HFRecommendedResponse

struct StartMSDownloadRequest: Encodable, Sendable {
    let modelId: String
    let msToken: String
}

struct StartMSDownloadResponse: Decodable, Sendable {
    let success: Bool
    let task: MSTaskDTO?
}

/// GET /admin/api/ms/status — `{"available": bool}`. False when the
/// modelscope Python SDK isn't installed (the server's `MS_SDK_AVAILABLE`
/// flag in `omlx/server.py:1278`). The Downloads tab gates the MS branch
/// on this so we don't show a flow that will only ever 503.
struct MSStatusResponse: Decodable, Sendable {
    let available: Bool
}

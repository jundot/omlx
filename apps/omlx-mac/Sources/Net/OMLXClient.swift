// PR 7 — async HTTP client for /admin/api/*. Cookie-jar session, JSON in/out,
// auto-login on 401 if an API key is configured.
//
// The client is host/port-mutable (when the user changes Listen Address or
// Port in ServerScreen we re-point it without rebuilding the URLSession). It
// owns no schedule of its own — callers are screen view-models that load on
// appear and refresh on a timer.

import Foundation

enum OMLXClientError: Error, CustomStringConvertible {
    case invalidURL
    case invalidResponse
    case unauthenticated
    case http(status: Int, body: String?)

    var description: String {
        switch self {
        case .invalidURL:           return "Invalid URL"
        case .invalidResponse:      return "Invalid response from server"
        case .unauthenticated:      return "Not authenticated (no API key configured)"
        case .http(let s, let b):   return "HTTP \(s)" + (b.map { ": \($0)" } ?? "")
        }
    }
}

extension Error {
    /// String suitable for the `lastError` / banner surface on screen VMs.
    /// OMLXClientError formats via its `description` (HTTP status + body,
    /// authentication state, etc.); other errors fall back to
    /// `localizedDescription`. Replaces the per-VM `describe(_:)` helper.
    var omlxDescription: String {
        if let omlx = self as? OMLXClientError { return String(describing: omlx) }
        return localizedDescription
    }
}

@MainActor
final class OMLXClient: ObservableObject {
    private(set) var host: String
    private(set) var port: Int
    private(set) var apiKey: String?

    private let session: URLSession
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder

    init(host: String = "127.0.0.1", port: Int = 8000, apiKey: String? = nil) {
        self.host = host
        self.port = port
        self.apiKey = apiKey

        let cfg = URLSessionConfiguration.default
        cfg.httpCookieStorage = HTTPCookieStorage.shared
        cfg.httpShouldSetCookies = true
        cfg.httpCookieAcceptPolicy = .always
        cfg.timeoutIntervalForRequest = 15
        cfg.requestCachePolicy = .reloadIgnoringLocalCacheData
        self.session = URLSession(configuration: cfg)

        let enc = JSONEncoder()
        enc.keyEncodingStrategy = .convertToSnakeCase
        self.encoder = enc

        let dec = JSONDecoder()
        dec.keyDecodingStrategy = .convertFromSnakeCase
        self.decoder = dec
    }

    func configure(host: String, port: Int, apiKey: String?) {
        self.host = host
        self.port = port
        self.apiKey = apiKey
    }

    // MARK: - Endpoints

    func getGlobalSettings() async throws -> GlobalSettingsDTO {
        try await get("/admin/api/global-settings")
    }

    func updateGlobalSettings(_ patch: GlobalSettingsPatch) async throws -> UpdateGlobalSettingsResponse {
        try await post("/admin/api/global-settings", body: patch)
    }

    func getServerInfo() async throws -> ServerInfoDTO {
        try await get("/admin/api/server-info")
    }

    func getStats(scope: String = "session", model: String = "") async throws -> StatsDTO {
        try await get("/admin/api/stats", query: [
            URLQueryItem(name: "scope", value: scope),
            URLQueryItem(name: "model", value: model),
        ])
    }

    @discardableResult
    func clearStats() async throws -> SimpleStatusResponse {
        try await postEmpty(AdminAPI.statsClear)
    }

    @discardableResult
    func clearAlltimeStats() async throws -> SimpleStatusResponse {
        try await postEmpty(AdminAPI.statsClearAlltime)
    }

    @discardableResult
    func clearSsdCache() async throws -> ClearSsdCacheResponse {
        try await postEmpty(AdminAPI.ssdCacheClear)
    }

    @discardableResult
    func clearHotCache() async throws -> ClearHotCacheResponse {
        try await postEmpty(AdminAPI.hotCacheClear)
    }

    func getLogs(lines: Int = 200, file: String? = nil) async throws -> LogsDTO {
        var q = [URLQueryItem(name: "lines", value: String(lines))]
        if let file, !file.isEmpty {
            q.append(URLQueryItem(name: "file", value: file))
        }
        return try await get(AdminAPI.logs, query: q)
    }

    // PR 8 — Models / Profiles / HF

    func listModels() async throws -> ListModelsResponse {
        try await get(AdminAPI.models)
    }

    @discardableResult
    func loadModel(id: String) async throws -> SimpleStatusResponse {
        try await postEmpty(AdminAPI.loadModel(id))
    }

    @discardableResult
    func unloadModel(id: String) async throws -> SimpleStatusResponse {
        try await postEmpty(AdminAPI.unloadModel(id))
    }

    @discardableResult
    func reloadModels() async throws -> SimpleStatusResponse {
        try await postEmpty(AdminAPI.reloadModels)
    }

    @discardableResult
    func updateModelSettings(id: String, patch: ModelSettingsPatch) async throws -> SimpleStatusResponse {
        try await put(AdminAPI.modelSettings(id), body: patch)
    }

    func listModelProfiles(id: String) async throws -> ProfileListResponse {
        try await get(AdminAPI.modelProfiles(id))
    }

    func createModelProfile(id: String, body: CreateProfileRequest) async throws -> CreateProfileResponse {
        try await post(AdminAPI.modelProfiles(id), body: body)
    }

    func updateModelProfile(id: String, name: String, body: UpdateProfileRequest) async throws -> UpdateProfileResponse {
        try await put(AdminAPI.modelProfile(id, name), body: body)
    }

    @discardableResult
    func deleteModelProfile(id: String, name: String) async throws -> DeleteResponse {
        try await delete(AdminAPI.modelProfile(id, name))
    }

    @discardableResult
    func applyModelProfile(id: String, name: String) async throws -> ApplyProfileResponse {
        try await postEmpty(AdminAPI.applyModelProfile(id, name))
    }

    func listProfileTemplates() async throws -> TemplateListResponse {
        try await get(AdminAPI.profileTemplates)
    }

    func createProfileTemplate(body: CreateTemplateRequest) async throws -> CreateTemplateResponse {
        try await post(AdminAPI.profileTemplates, body: body)
    }

    func updateProfileTemplate(name: String, body: UpdateTemplateRequest) async throws -> UpdateTemplateResponse {
        try await put(AdminAPI.profileTemplate(name), body: body)
    }

    /// Force-refresh the preset bundle from omlx.ai (via the server proxy
    /// at `/api/presets/refresh`). Callers cache the result on disk; the
    /// `PresetBundleStore` is the single in-app consumer.
    func refreshPresetBundle() async throws -> PresetBundleDTO {
        try await postEmpty(AdminAPI.presetsRefresh)
    }

    @discardableResult
    func deleteProfileTemplate(name: String) async throws -> DeleteResponse {
        try await delete(AdminAPI.profileTemplate(name))
    }

    func listHFTasks() async throws -> HFTaskListResponse {
        try await get(AdminAPI.hfTasks)
    }

    func startHFDownload(repoId: String, hfToken: String = "") async throws -> StartHFDownloadResponse {
        try await post(AdminAPI.hfDownload, body: StartHFDownloadRequest(
            repoId: repoId, hfToken: hfToken
        ))
    }

    @discardableResult
    func cancelHFDownload(taskId: String) async throws -> SimpleStatusResponse {
        try await postEmpty(AdminAPI.hfCancel(taskId))
    }

    @discardableResult
    func retryHFDownload(taskId: String) async throws -> StartHFDownloadResponse {
        try await postEmpty(AdminAPI.hfRetry(taskId))
    }

    @discardableResult
    func removeHFTask(taskId: String) async throws -> SimpleStatusResponse {
        try await delete(AdminAPI.hfTask(taskId))
    }

    func getHFRecommended(mlxOnly: Bool = true) async throws -> HFRecommendedResponse {
        try await get(AdminAPI.hfRecommended, query: [
            URLQueryItem(name: "mlx_only", value: mlxOnly ? "true" : "false"),
        ])
    }

    /// Search HF Hub for repos matching a free-text query. Same endpoint the
    /// browser admin panel uses for its model picker (dashboard.js:3879).
    /// Defaults match the JS caller: trending sort, MLX-only filter, cap 20
    /// rows (we render at most 8 in the dropdown anyway).
    func searchHFModels(
        query: String,
        sort: String = "trending",
        limit: Int = 20,
        mlxOnly: Bool = true
    ) async throws -> HFSearchResponse {
        try await get(AdminAPI.hfSearch, query: [
            URLQueryItem(name: "q", value: query),
            URLQueryItem(name: "sort", value: sort),
            URLQueryItem(name: "limit", value: String(limit)),
            URLQueryItem(name: "mlx_only", value: mlxOnly ? "true" : "false"),
        ])
    }

    /// Fetch the README for a Hugging Face repo, the same payload the
    /// browser admin renders in its model-card slide-over. Cache-aware
    /// on the server (uses `hf_hub_download` under the hood), so post-
    /// download lookups skip the network. Returns an empty
    /// `modelCard` string when the upstream repo has no README — that
    /// is a "no card" state, not an error.
    func getHFModelCard(repoId: String) async throws -> ModelCardDTO {
        try await get(AdminAPI.hfModelInfo, query: [
            URLQueryItem(name: "repo_id", value: repoId),
        ])
    }

    // MARK: - ModelScope (Phase 2)
    //
    // 1:1 mirror of the /hf/* surface above, pointed at the parallel
    // pipeline implemented by `omlx/admin/ms_downloader.py`. Task / model
    // shapes are identical to HF (`MSTaskDTO = HFTaskDTO` typealias), so
    // most call sites can read the response with the existing types.

    /// GET /ms/status — returns `{available: bool}`. Use to gate the MS
    /// branch of the Downloads UI so we don't render a flow that will only
    /// ever 503 when the modelscope Python SDK isn't installed in the
    /// bundle's venv layer.
    func getMSStatus() async throws -> MSStatusResponse {
        try await get(AdminAPI.msStatus)
    }

    func listMSTasks() async throws -> MSTaskListResponse {
        try await get(AdminAPI.msTasks)
    }

    func startMSDownload(modelId: String, msToken: String = "") async throws -> StartMSDownloadResponse {
        try await post(AdminAPI.msDownload, body: StartMSDownloadRequest(
            modelId: modelId, msToken: msToken
        ))
    }

    @discardableResult
    func cancelMSDownload(taskId: String) async throws -> SimpleStatusResponse {
        try await postEmpty(AdminAPI.msCancel(taskId))
    }

    @discardableResult
    func retryMSDownload(taskId: String) async throws -> StartMSDownloadResponse {
        try await postEmpty(AdminAPI.msRetry(taskId))
    }

    @discardableResult
    func removeMSTask(taskId: String) async throws -> SimpleStatusResponse {
        try await delete(AdminAPI.msTask(taskId))
    }

    func getMSRecommended(mlxOnly: Bool = true) async throws -> MSRecommendedResponse {
        try await get(AdminAPI.msRecommended, query: [
            URLQueryItem(name: "mlx_only", value: mlxOnly ? "true" : "false"),
        ])
    }

    func searchMSModels(
        query: String,
        sort: String = "trending",
        limit: Int = 20,
        mlxOnly: Bool = true
    ) async throws -> MSSearchResponse {
        try await get(AdminAPI.msSearch, query: [
            URLQueryItem(name: "q", value: query),
            URLQueryItem(name: "sort", value: sort),
            URLQueryItem(name: "limit", value: String(limit)),
            URLQueryItem(name: "mlx_only", value: mlxOnly ? "true" : "false"),
        ])
    }

    /// ModelScope mirror of `getHFModelCard(repoId:)`. Returns the same
    /// shape (`{model_card: "<markdown>"}`); empty string when the
    /// upstream repo has no README.
    func getMSModelCard(modelId: String) async throws -> ModelCardDTO {
        try await get(AdminAPI.msModelInfo, query: [
            URLQueryItem(name: "model_id", value: modelId),
        ])
    }

    /// Delete a downloaded model directory from disk. The server unloads
    /// the engine first if it's currently loaded, then rmtree's the model
    /// directory and refreshes the pool. 404 if the name doesn't resolve.
    @discardableResult
    func deleteHFModel(modelName: String) async throws -> SimpleStatusResponse {
        try await delete(AdminAPI.hfModel(modelName))
    }

    // PR 9 — Security

    @discardableResult
    func setupApiKey(_ key: String, confirm: String) async throws -> SimpleStatusResponse {
        try await post(AdminAPI.setupApiKey, body: SetupApiKeyRequest(
            apiKey: key, apiKeyConfirm: confirm
        ))
    }

    @discardableResult
    func createSubKey(key: String, name: String) async throws -> CreateSubKeyResponse {
        try await post(AdminAPI.subKeys, body: CreateSubKeyRequest(key: key, name: name))
    }

    @discardableResult
    func deleteSubKey(key: String) async throws -> SimpleStatusResponse {
        try await deleteWithBody(AdminAPI.subKeys, body: DeleteSubKeyRequest(key: key))
    }

    // PR 12 — oQ Quantization

    /// List quantizable source models + every on-disk model (sensitivity picker).
    func listOQModels() async throws -> OQModelsResponse {
        try await get(AdminAPI.oqModels)
    }

    /// Server-side precise estimate of effective bpw + output size for a
    /// given source model at the chosen oQ level. Result is cheap (no
    /// quantization happens); the screen debounces calls at ~300 ms.
    func estimateOQ(
        modelPath: String,
        oqLevel: Double,
        preserveMtp: Bool = false
    ) async throws -> OQEstimateResponse {
        // `oq_level` accepts ints and fractional levels. Send it without a
        // trailing `.0` so the server parses an int when the user picked one.
        let levelStr: String = (oqLevel.rounded() == oqLevel)
            ? String(Int(oqLevel))
            : String(oqLevel)
        return try await get(AdminAPI.oqEstimate, query: [
            URLQueryItem(name: "model_path", value: modelPath),
            URLQueryItem(name: "oq_level", value: levelStr),
            URLQueryItem(name: "preserve_mtp", value: preserveMtp ? "true" : "false"),
        ])
    }

    @discardableResult
    func startOQQuantization(_ body: OQStartRequest) async throws -> OQStartResponse {
        try await post(AdminAPI.oqStart, body: body)
    }

    func listOQTasks() async throws -> OQTasksResponse {
        try await get(AdminAPI.oqTasks)
    }

    @discardableResult
    func cancelOQTask(taskId: String) async throws -> SimpleStatusResponse {
        try await postEmpty(AdminAPI.oqCancel(taskId))
    }

    @discardableResult
    func removeOQTask(taskId: String) async throws -> SimpleStatusResponse {
        try await delete(AdminAPI.oqTask(taskId))
    }

    // PR 13 — HF Upload (merged into QuantizationScreen)

    func validateHFUploadToken(hfToken: String) async throws -> HFValidateTokenResponse {
        try await post(AdminAPI.uploadValidateToken, body: HFValidateTokenRequest(hfToken: hfToken))
    }

    func listHFUploadModels() async throws -> HFUploadModelsResponse {
        try await get(AdminAPI.uploadModels)
    }

    @discardableResult
    func startHFUpload(_ body: HFUploadStartRequest) async throws -> HFUploadStartResponse {
        try await post(AdminAPI.uploadStart, body: body)
    }

    func listHFUploadTasks() async throws -> HFUploadTasksResponse {
        try await get(AdminAPI.uploadTasks)
    }

    @discardableResult
    func cancelHFUploadTask(taskId: String) async throws -> SimpleStatusResponse {
        try await postEmpty(AdminAPI.uploadCancel(taskId))
    }

    @discardableResult
    func removeHFUploadTask(taskId: String) async throws -> SimpleStatusResponse {
        try await delete(AdminAPI.uploadTask(taskId))
    }

    // PR 13 — Throughput bench

    func getDeviceInfo() async throws -> DeviceInfoDTO {
        try await get(AdminAPI.deviceInfo)
    }

    @discardableResult
    func startThroughputBench(_ body: BenchStartRequest) async throws -> BenchStartResponse {
        try await post(AdminAPI.benchStart, body: body)
    }

    func getBenchResults(benchId: String) async throws -> BenchResultsResponse {
        try await get(AdminAPI.benchResults(benchId))
    }

    @discardableResult
    func cancelBench(benchId: String) async throws -> BenchCancelResponse {
        try await postEmpty(AdminAPI.benchCancel(benchId))
    }

    // PR 13 — Accuracy bench

    @discardableResult
    func addAccuracyQueue(_ body: AccuracyQueueAddRequest) async throws -> AccuracyQueueStatus {
        try await post(AdminAPI.accuracyQueueAdd, body: body)
    }

    func getAccuracyQueueStatus() async throws -> AccuracyQueueStatus {
        try await get(AdminAPI.accuracyQueueStatus)
    }

    @discardableResult
    func removeAccuracyQueue(index: Int) async throws -> AccuracyQueueStatus {
        try await delete(AdminAPI.accuracyQueueRemove(index))
    }

    func listAccuracyResults() async throws -> AccuracyResultsResponse {
        try await get(AdminAPI.accuracyResults)
    }

    @discardableResult
    func resetAccuracyResults() async throws -> SimpleStatusResponse {
        try await postEmpty(AdminAPI.accuracyReset)
    }

    @discardableResult
    func cancelAccuracyBench() async throws -> SimpleStatusResponse {
        try await postEmpty(AdminAPI.accuracyCancel)
    }

    // MARK: - Core request

    private func get<T: Decodable>(_ path: String, query: [URLQueryItem] = []) async throws -> T {
        try await request("GET", path: path, query: query, body: nil)
    }

    private func post<U: Encodable, T: Decodable>(_ path: String, body: U) async throws -> T {
        let data = try encoder.encode(body)
        return try await request("POST", path: path, body: data)
    }

    private func put<U: Encodable, T: Decodable>(_ path: String, body: U) async throws -> T {
        let data = try encoder.encode(body)
        return try await request("PUT", path: path, body: data)
    }

    private func postEmpty<T: Decodable>(_ path: String) async throws -> T {
        try await request("POST", path: path, body: nil)
    }

    private func delete<T: Decodable>(_ path: String) async throws -> T {
        try await request("DELETE", path: path, body: nil)
    }

    /// DELETE with a JSON body. The /admin/api/sub-keys endpoint is the
    /// only caller — the server reads `key` from the request body rather
    /// than the URL.
    private func deleteWithBody<U: Encodable, T: Decodable>(_ path: String, body: U) async throws -> T {
        let data = try encoder.encode(body)
        return try await request("DELETE", path: path, body: data)
    }

    private func request<T: Decodable>(
        _ method: String,
        path: String,
        query: [URLQueryItem] = [],
        body: Data?,
        isRetry: Bool = false
    ) async throws -> T {
        var components = URLComponents()
        components.scheme = "http"
        components.host = host
        components.port = port
        components.path = path
        if !query.isEmpty { components.queryItems = query }
        guard let url = components.url else { throw OMLXClientError.invalidURL }

        var req = URLRequest(url: url)
        req.httpMethod = method
        req.setValue("application/json", forHTTPHeaderField: "Accept")
        if body != nil {
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = body
        }

        let (data, resp) = try await session.data(for: req)
        guard let http = resp as? HTTPURLResponse else { throw OMLXClientError.invalidResponse }

        if http.statusCode == 401, !isRetry {
            guard let key = apiKey, !key.isEmpty else {
                throw OMLXClientError.unauthenticated
            }
            try await login(apiKey: key)
            return try await request(method, path: path, query: query, body: body, isRetry: true)
        }

        guard 200..<300 ~= http.statusCode else {
            let bodyStr = String(data: data, encoding: .utf8)
            throw OMLXClientError.http(status: http.statusCode, body: bodyStr)
        }

        if T.self == EmptyResponse.self {
            return EmptyResponse() as! T
        }
        return try decoder.decode(T.self, from: data)
    }

    private func login(apiKey: String) async throws {
        struct LoginReq: Encodable { let apiKey: String; let remember: Bool }
        var components = URLComponents()
        components.scheme = "http"
        components.host = host
        components.port = port
        components.path = "/admin/api/login"
        guard let url = components.url else { throw OMLXClientError.invalidURL }

        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try encoder.encode(LoginReq(apiKey: apiKey, remember: true))

        let (data, resp) = try await session.data(for: req)
        guard let http = resp as? HTTPURLResponse else { throw OMLXClientError.invalidResponse }
        guard 200..<300 ~= http.statusCode else {
            let bodyStr = String(data: data, encoding: .utf8)
            throw OMLXClientError.http(status: http.statusCode, body: bodyStr)
        }
    }
}

struct EmptyResponse: Decodable {}

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

@MainActor
final class OMLXClient: ObservableObject {
    private(set) var host: String
    private(set) var port: Int
    private(set) var apiKey: String?

    private let session: URLSession
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder

    init(host: String = "127.0.0.1", port: Int = 8080, apiKey: String? = nil) {
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

    func getLogs(lines: Int = 200, file: String? = nil) async throws -> LogsDTO {
        var q = [URLQueryItem(name: "lines", value: String(lines))]
        if let file, !file.isEmpty {
            q.append(URLQueryItem(name: "file", value: file))
        }
        return try await get("/admin/api/logs", query: q)
    }

    // MARK: - Core request

    private func get<T: Decodable>(_ path: String, query: [URLQueryItem] = []) async throws -> T {
        try await request("GET", path: path, query: query, body: nil)
    }

    private func post<U: Encodable, T: Decodable>(_ path: String, body: U) async throws -> T {
        let data = try encoder.encode(body)
        return try await request("POST", path: path, body: data)
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

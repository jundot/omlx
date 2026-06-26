// Menubar poller: use lightweight /api/status for live display and liveness,
// with occasional /admin/api/stats?scope=alltime fetches for the All-Time
// submenu. Emits NotificationCenter posts so the menubar refreshes without
// polling state itself.
//
// PR 7's OMLXClient will absorb this auth machinery; for now the poller owns
// its own URLSession + cookie jar to keep the menubar self-contained.

import Foundation

@MainActor
final class MenubarStatsPoller {
    static let didUpdateNotification = Notification.Name("OMLXMenubarStatsDidUpdate")

    /// Subset shared by /api/status and /admin/api/stats responses — extend
    /// as the menubar surfaces more fields. Keys mirror server JSON.
    struct Stats: Codable, Sendable, Equatable {
        var totalPromptTokens: Int?
        var totalCachedTokens: Int?
        var cacheEfficiency: Double?
        var avgPrefillTps: Double?
        var avgGenerationTps: Double?
        var totalRequests: Int?

        enum CodingKeys: String, CodingKey {
            case totalPromptTokens = "total_prompt_tokens"
            case totalCachedTokens = "total_cached_tokens"
            case cacheEfficiency  = "cache_efficiency"
            case avgPrefillTps    = "avg_prefill_tps"
            case avgGenerationTps = "avg_generation_tps"
            case totalRequests    = "total_requests"
        }
    }

    private let baseURL: URL
    private let apiKey: String?
    private let interval: TimeInterval
    private let session: URLSession
    private var task: Task<Void, Never>?
    /// Number of ticks between all-time fetches. Live status is cheap and
    /// needs steady refresh; all-time stats only show in the "All-Time"
    /// submenu, so polling them at the same rate burns server CPU for no UX
    /// benefit. At interval=2s this fetches every 30s.
    private let alltimeEveryNTicks = 15
    private var tickCount = 0

    private(set) var sessionStats: Stats?
    private(set) var alltimeStats: Stats?
    private(set) var lastStatusSuccessAt: Date?

    init(baseURL: URL, apiKey: String?, interval: TimeInterval = 2.0) {
        self.baseURL = baseURL
        self.apiKey = apiKey
        self.interval = interval

        let cfg = URLSessionConfiguration.default
        // `HTTPCookieStorage()` returns a detached instance that never
        // actually persists cookies, so the post-login session cookie was
        // dropped and every subsequent /api/stats request 401-ed. Since
        // FastAPI's 401 body still JSON-decodes into our all-Optional Stats
        // struct (all keys missing → all fields nil), the menubar rendered
        // "—" everywhere with no error trail. Use the process-wide shared
        // jar — matches OMLXClient and inherits its login session.
        cfg.httpCookieStorage = HTTPCookieStorage.shared
        cfg.httpShouldSetCookies = true
        cfg.httpCookieAcceptPolicy = .always
        cfg.requestCachePolicy = .reloadIgnoringLocalCacheData
        cfg.timeoutIntervalForRequest = 5.0
        self.session = URLSession(configuration: cfg)
    }

    func start() {
        stop()
        task = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                await self.tick()
                try? await Task.sleep(for: .seconds(self.interval))
            }
        }
    }

    func stop() {
        task?.cancel()
        task = nil
    }

    deinit {
        // Detached cancel — actor-isolated stop() can't run from deinit.
        task?.cancel()
    }

    // MARK: - Polling

    private func tick() async {
        let fetchAlltime = (tickCount % alltimeEveryNTicks == 0)
        tickCount &+= 1
        do {
            let s = try await fetchPublicStatus()
            self.sessionStats = s
            self.lastStatusSuccessAt = Date()
            NotificationCenter.default.post(
                name: Self.didUpdateNotification, object: self
            )
            if fetchAlltime, hasAPIKey,
               let alltime = try? await fetchAdminStats(scope: "alltime") {
                self.alltimeStats = alltime
            }
        } catch {
            // Suppress: server may be transitioning, paused, or 401-pending.
            // Next tick retries; we log only the once-per-tick failure mode.
        }
    }

    private func fetchPublicStatus() async throws -> Stats {
        let url = try makeURL(path: "/api/status")
        var req = URLRequest(url: url)
        req.setValue("application/json", forHTTPHeaderField: "Accept")
        if let key = apiKey, !key.isEmpty {
            req.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        }
        let (data, response) = try await session.data(for: req)
        try validateOK(response)
        return try JSONDecoder().decode(Stats.self, from: data)
    }

    private func fetchAdminStats(scope: String) async throws -> Stats {
        let url = try makeURL(
            path: "/admin/api/stats",
            queryItems: [URLQueryItem(name: "scope", value: scope)]
        )
        var req = URLRequest(url: url)
        req.setValue("application/json", forHTTPHeaderField: "Accept")
        let (data, response) = try await session.data(for: req)

        if let http = response as? HTTPURLResponse, http.statusCode == 401 {
            try await login()
            let (data2, response2) = try await session.data(for: req)
            try validateOK(response2)
            return try JSONDecoder().decode(Stats.self, from: data2)
        }
        try validateOK(response)
        return try JSONDecoder().decode(Stats.self, from: data)
    }

    private func login() async throws {
        guard let apiKey, !apiKey.isEmpty else {
            throw URLError(.userAuthenticationRequired)
        }
        var req = URLRequest(url: try makeURL(path: "/admin/api/login"))
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONEncoder().encode(["api_key": apiKey])
        let (_, response) = try await session.data(for: req)
        try validateOK(response)
    }

    private var hasAPIKey: Bool {
        guard let apiKey else { return false }
        return !apiKey.isEmpty
    }

    private func makeURL(path: String, queryItems: [URLQueryItem] = []) throws -> URL {
        var comps = URLComponents(url: baseURL, resolvingAgainstBaseURL: false)
        comps?.path = path.hasPrefix("/") ? path : "/" + path
        if !queryItems.isEmpty {
            comps?.queryItems = queryItems
        }
        guard let url = comps?.url else {
            throw URLError(.badURL)
        }
        return url
    }

    private func validateOK(_ response: URLResponse) throws {
        guard let http = response as? HTTPURLResponse else {
            throw URLError(.badServerResponse)
        }
        guard (200..<300).contains(http.statusCode) else {
            throw URLError(.userAuthenticationRequired)
        }
    }
}

// PR 4 — 1Hz poller against /admin/api/stats. Mirrors the requests.Session
// flow in app.py:1745-1840: persistent cookies, auto re-login on 401, scope
// query for session vs all-time stats. Emits NotificationCenter posts so the
// menubar refreshes without polling state itself.
//
// PR 7's OMLXClient will absorb this auth machinery; for now the poller owns
// its own URLSession + cookie jar to keep the menubar self-contained.

import Foundation

@MainActor
final class MenubarStatsPoller {
    static let didUpdateNotification = Notification.Name("OMLXMenubarStatsDidUpdate")

    /// Subset of /admin/api/stats response — extend as the menubar surfaces
    /// more fields. Keys mirror routes.py JSON.
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
    private let apiKey: String
    private let interval: TimeInterval
    private let session: URLSession
    private var task: Task<Void, Never>?

    private(set) var sessionStats: Stats?
    private(set) var alltimeStats: Stats?

    init(baseURL: URL, apiKey: String, interval: TimeInterval = 1.0) {
        self.baseURL = baseURL
        self.apiKey = apiKey
        self.interval = interval

        let cfg = URLSessionConfiguration.default
        cfg.httpCookieStorage = HTTPCookieStorage()
        cfg.httpCookieAcceptPolicy = .always
        cfg.requestCachePolicy = .reloadIgnoringLocalCacheData
        cfg.timeoutIntervalForRequest = 2.0
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
        do {
            async let session = fetchStats(scope: nil)
            async let alltime = fetchStats(scope: "alltime")
            let s = try await session
            let a = try await alltime
            self.sessionStats = s
            self.alltimeStats = a
            NotificationCenter.default.post(
                name: Self.didUpdateNotification, object: self
            )
        } catch {
            // Suppress: server may be transitioning, paused, or 401-pending.
            // Next tick retries; we log only the once-per-tick failure mode.
        }
    }

    private func fetchStats(scope: String?) async throws -> Stats {
        let url = try statsURL(scope: scope)
        let req = URLRequest(url: url)
        let (data, response) = try await session.data(for: req)

        if let http = response as? HTTPURLResponse, http.statusCode == 401 {
            try await login()
            let (data2, _) = try await session.data(for: req)
            return try JSONDecoder().decode(Stats.self, from: data2)
        }
        return try JSONDecoder().decode(Stats.self, from: data)
    }

    private func login() async throws {
        var req = URLRequest(url: baseURL.appendingPathComponent("/admin/api/login"))
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONEncoder().encode(["api_key": apiKey])
        _ = try await session.data(for: req)
    }

    private func statsURL(scope: String?) throws -> URL {
        var comps = URLComponents(
            url: baseURL.appendingPathComponent("/admin/api/stats"),
            resolvingAgainstBaseURL: false
        )
        if let scope {
            comps?.queryItems = [URLQueryItem(name: "scope", value: scope)]
        }
        guard let url = comps?.url else {
            throw URLError(.badURL)
        }
        return url
    }
}

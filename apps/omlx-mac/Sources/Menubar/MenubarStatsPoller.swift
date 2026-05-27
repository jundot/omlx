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
    /// Number of ticks between all-time fetches. Session stats need
    /// sub-second freshness for the live tok/s display in the header;
    /// all-time stats only show in the "All-Time" submenu and don't
    /// jitter, so polling them at the same rate burns server CPU for
    /// no UX benefit. At interval=1s this fetches every 10s.
    private let alltimeEveryNTicks = 10
    private var tickCount = 0

    private(set) var sessionStats: Stats?
    private(set) var alltimeStats: Stats?

    init(baseURL: URL, apiKey: String, interval: TimeInterval = 1.0) {
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
        let fetchAlltime = (tickCount % alltimeEveryNTicks == 0)
        tickCount &+= 1
        do {
            let s = try await fetchStats(scope: nil)
            self.sessionStats = s
            if fetchAlltime {
                self.alltimeStats = try await fetchStats(scope: "alltime")
            }
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
            let (data2, response2) = try await session.data(for: req)
            if let http2 = response2 as? HTTPURLResponse,
               !(200..<300).contains(http2.statusCode) {
                // Stats has all-Optional fields, so a FastAPI error body
                // (`{"detail": "..."}`) decodes into an all-nil struct and
                // silently overwrites real stats with dashes. Fail loudly
                // instead so the outer tick() catches and we keep the last
                // good values.
                throw URLError(.userAuthenticationRequired)
            }
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

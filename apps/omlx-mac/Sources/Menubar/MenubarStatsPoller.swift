// Menubar poller: use lightweight /api/status for liveness, authenticated
// /admin/api/activity reads for current request activity, and occasional
// all-time stats reads for the Serving Stats submenu. Emits
// NotificationCenter posts so the menubar refreshes without polling state
// itself.
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
        var activeModels: ActiveModels?

        var liveActivity: LiveActivity? {
            LiveActivity(activeModels: activeModels)
        }

        enum CodingKeys: String, CodingKey {
            case totalPromptTokens = "total_prompt_tokens"
            case totalCachedTokens = "total_cached_tokens"
            case cacheEfficiency  = "cache_efficiency"
            case avgPrefillTps    = "avg_prefill_tps"
            case avgGenerationTps = "avg_generation_tps"
            case totalRequests    = "total_requests"
            case activeModels     = "active_models"
        }

        struct ActiveModels: Codable, Sendable, Equatable {
            let models: [ActiveModel]
            let totalWaitingRequests: Int?

            enum CodingKeys: String, CodingKey {
                case models
                case totalWaitingRequests = "total_waiting_requests"
            }
        }

        struct ActiveModel: Codable, Sendable, Equatable {
            let id: String
            let prefilling: [PrefillProgress]?
            let generating: [GenerationProgress]?
            let activities: [NonStreamingActivity]?
        }

        struct PrefillProgress: Codable, Sendable, Equatable {
            let processed: Int?
            let total: Int?
            let speed: Double?
            let eta: Double?
        }

        struct GenerationProgress: Codable, Sendable, Equatable {
            let generatedTokens: Int?
            let tokensPerSecond: Double?
            let elapsedSeconds: Double?

            enum CodingKeys: String, CodingKey {
                case generatedTokens = "generated_tokens"
                case tokensPerSecond = "tokens_per_second"
                case elapsedSeconds = "elapsed_seconds"
            }
        }

        struct NonStreamingActivity: Codable, Sendable, Equatable {
            let kind: String?
            let detail: String?
            let elapsedSeconds: Double?

            enum CodingKeys: String, CodingKey {
                case kind
                case detail
                case elapsedSeconds = "elapsed_seconds"
            }
        }

        struct LiveActivity: Equatable, Sendable {
            let menuBarTitle: String
            let detail: String

            private init(menuBarTitle: String, detail: String) {
                self.menuBarTitle = menuBarTitle
                self.detail = detail
            }

            init?(activeModels: ActiveModels?) {
                guard let activeModels else {
                    return nil
                }

                for activeModel in activeModels.models {
                    if let prefill = activeModel.prefilling?.first {
                        self = Self.prefill(modelID: activeModel.id, progress: prefill)
                        return
                    }
                }

                for activeModel in activeModels.models {
                    if let generation = activeModel.generating?.first {
                        self = Self.generation(modelID: activeModel.id, progress: generation)
                        return
                    }
                }

                for activeModel in activeModels.models {
                    if let activity = activeModel.activities?.first {
                        self = Self.nonStreaming(modelID: activeModel.id, activity: activity)
                        return
                    }
                }

                if let waitingRequestCount = activeModels.totalWaitingRequests,
                   waitingRequestCount > 0 {
                    self = Self.waiting(requestCount: waitingRequestCount)
                    return
                }

                return nil
            }

            private static func prefill(
                modelID: String,
                progress: PrefillProgress
            ) -> LiveActivity {
                let processedTokens = max(0, progress.processed ?? 0)
                let totalTokens = max(0, progress.total ?? 0)
                let percentage = totalTokens > 0
                    ? Int((Double(processedTokens) / Double(totalTokens) * 100).rounded())
                    : 0

                var detailParts = [modelID]
                if let tokensPerSecond = progress.speed, tokensPerSecond > 0 {
                    detailParts.append("\(Int(tokensPerSecond.rounded())) tok/s")
                }
                if let etaSeconds = progress.eta, etaSeconds >= 0 {
                    detailParts.append("\(formatDuration(etaSeconds)) left")
                }

                return LiveActivity(
                    menuBarTitle: "PP \(percentage)% · \(formatTokenCount(processedTokens))/\(formatTokenCount(totalTokens))",
                    detail: detailParts.joined(separator: " · ")
                )
            }

            private static func generation(
                modelID: String,
                progress: GenerationProgress
            ) -> LiveActivity {
                let tokensPerSecond = max(0, progress.tokensPerSecond ?? 0)
                let generatedTokens = max(0, progress.generatedTokens ?? 0)

                var detailParts = [modelID, "\(generatedTokens) tok"]
                if let elapsedSeconds = progress.elapsedSeconds {
                    detailParts.append(formatDuration(elapsedSeconds))
                }

                return LiveActivity(
                    menuBarTitle: "GEN \(String(format: "%.1f", tokensPerSecond)) tok/s",
                    detail: detailParts.joined(separator: " · ")
                )
            }

            private static func waiting(requestCount: Int) -> LiveActivity {
                LiveActivity(
                    menuBarTitle: "WAIT \(requestCount)",
                    detail: "\(requestCount) queued request\(requestCount == 1 ? "" : "s")"
                )
            }

            private static func nonStreaming(
                modelID: String,
                activity: NonStreamingActivity
            ) -> LiveActivity {
                let elapsed = activity.elapsedSeconds.map(formatDuration)
                let activityDetail = activity.detail ?? activity.kind ?? "Active request"
                var detailParts = [modelID, activityDetail]
                if let elapsed {
                    detailParts.append(elapsed)
                }

                return LiveActivity(
                    menuBarTitle: elapsed.map { "RUN \($0)" } ?? "RUN",
                    detail: detailParts.joined(separator: " · ")
                )
            }

            private static func formatTokenCount(_ tokenCount: Int) -> String {
                if tokenCount >= 1_000_000 {
                    let millions = Double(tokenCount) / 1_000_000
                    return String(format: millions >= 10 ? "%.0fM" : "%.1fM", millions)
                }
                if tokenCount >= 1_000 {
                    return "\(Int((Double(tokenCount) / 1_000).rounded()))k"
                }
                return "\(tokenCount)"
            }

            private static func formatDuration(_ seconds: Double) -> String {
                let roundedSeconds = max(0, Int(seconds.rounded()))
                if roundedSeconds >= 60 {
                    let minutes = roundedSeconds / 60
                    let remainingSeconds = roundedSeconds % 60
                    return remainingSeconds == 0
                        ? "\(minutes)m"
                        : "\(minutes)m \(remainingSeconds)s"
                }
                return "\(roundedSeconds)s"
            }
        }
    }

    private let baseURL: URL
    private let apiKey: String?
    private let idleInterval: TimeInterval
    private let liveActivityInterval: TimeInterval
    private let session: URLSession
    private var task: Task<Void, Never>?
    /// Number of ticks between all-time fetches. Current request activity
    /// needs steady refresh; all-time stats only show in the "All-Time"
    /// submenu, so polling them at the same rate burns server CPU for no UX
    /// benefit. At the one-second live interval this fetches every 30 seconds.
    private let alltimeRefreshInterval: TimeInterval = 30
    private var tickCount = 0
    private var isLiveActivityPollingEnabled = false

    private(set) var sessionStats: Stats?
    private(set) var liveStats: Stats?
    private(set) var alltimeStats: Stats?
    private(set) var lastStatusSuccessAt: Date?

    init(
        baseURL: URL,
        apiKey: String?,
        interval: TimeInterval = 2.0,
        liveActivityInterval: TimeInterval = 1.0,
        sessionConfiguration: URLSessionConfiguration? = nil
    ) {
        self.baseURL = baseURL
        self.apiKey = apiKey
        self.idleInterval = interval
        self.liveActivityInterval = liveActivityInterval

        let cfg = sessionConfiguration ?? URLSessionConfiguration.default
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
                await self.refreshOnce()
                try? await Task.sleep(for: .seconds(self.currentPollingInterval))
            }
        }
    }

    func stop() {
        task?.cancel()
        task = nil
    }

    func setLiveActivityPollingEnabled(_ isEnabled: Bool) {
        guard isLiveActivityPollingEnabled != isEnabled else {
            return
        }

        isLiveActivityPollingEnabled = isEnabled
        if !isEnabled {
            clearLiveStats()
        }
    }

    var currentPollingInterval: TimeInterval {
        isLiveActivityPollingEnabled ? liveActivityInterval : idleInterval
    }

    deinit {
        // Detached cancel — actor-isolated stop() can't run from deinit.
        task?.cancel()
    }

    // MARK: - Polling

    func refreshOnce() async {
        let alltimeEveryNTicks = max(
            1,
            Int((alltimeRefreshInterval / currentPollingInterval).rounded())
        )
        let fetchAlltime = (tickCount % alltimeEveryNTicks == 0)
        tickCount &+= 1
        do {
            let s = try await fetchPublicStatus()
            self.sessionStats = s
            self.lastStatusSuccessAt = Date()
            if isLiveActivityPollingEnabled {
                do {
                    let live = try await fetchAdminActivity()
                    if isLiveActivityPollingEnabled {
                        self.liveStats = live
                    }
                } catch {
                    if isLiveActivityPollingEnabled {
                        clearLiveStats(shouldPostUpdate: false)
                    }
                }
            }
            if fetchAlltime, hasAPIKey,
               let alltime = try? await fetchAdminStats(scope: "alltime") {
                self.alltimeStats = alltime
            }
            NotificationCenter.default.post(
                name: Self.didUpdateNotification, object: self
            )
        } catch {
            // Suppress: server may be transitioning, paused, or 401-pending.
            // Next tick retries; we log only the once-per-tick failure mode.
            clearLiveStats()
        }
    }

    private func clearLiveStats(shouldPostUpdate: Bool = true) {
        guard liveStats != nil else {
            return
        }

        liveStats = nil
        if shouldPostUpdate {
            NotificationCenter.default.post(name: Self.didUpdateNotification, object: self)
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

    private func fetchAdminActivity() async throws -> Stats {
        let url = try makeURL(path: "/admin/api/activity")
        var req = URLRequest(url: url)
        req.setValue("application/json", forHTTPHeaderField: "Accept")
        let (data, response) = try await session.data(for: req)

        if let http = response as? HTTPURLResponse, http.statusCode == 401 {
            try await login()
            let (authenticatedData, authenticatedResponse) = try await session.data(for: req)
            try validateOK(authenticatedResponse)
            return try JSONDecoder().decode(Stats.self, from: authenticatedData)
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

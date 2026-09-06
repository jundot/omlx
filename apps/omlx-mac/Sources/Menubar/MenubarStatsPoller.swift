// Menubar poller: use lightweight /api/status for liveness, authenticated
// /admin/api/activity reads for current request activity, and occasional
// all-time stats reads only while a consumer needs them. Emits
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
            /// A hard cap keeps the LIV popover useful without allowing a busy
            /// server to grow an unbounded menubar panel.
            static let maximumVisibleRequests = 6

            struct Request: Equatable, Sendable, Identifiable {
                enum Kind: Equatable, Sendable {
                    case prefill
                    case generating
                    case nonStreaming
                }

                let id: String
                let title: String
                let detail: String
                let kind: Kind
            }

            struct ModelGroup: Equatable, Sendable, Identifiable {
                let modelID: String
                let requests: [Request]

                var id: String { modelID }
            }

            let groups: [ModelGroup]
            let queuedRequestCount: Int
            let hiddenRequestCount: Int

            var isIdle: Bool {
                groups.isEmpty && queuedRequestCount == 0
            }

            var detail: String {
                if let group = groups.first, let request = group.requests.first {
                    return [group.modelID, request.detail]
                        .filter { !$0.isEmpty }
                        .joined(separator: " · ")
                }
                return queuedRequestCount > 0
                    ? "\(queuedRequestCount) queued request\(queuedRequestCount == 1 ? "" : "s")"
                    : ""
            }

            init?(activeModels: ActiveModels?) {
                guard let activeModels else {
                    return nil
                }

                var visibleCount = 0
                var hiddenRequestCount = 0
                var groups: [ModelGroup] = []
                for activeModel in activeModels.models {
                    let requests = Self.requests(for: activeModel)
                    let available = max(0, Self.maximumVisibleRequests - visibleCount)
                    let shown = Array(requests.prefix(available))
                    hiddenRequestCount += requests.count - shown.count
                    visibleCount += shown.count
                    if !shown.isEmpty {
                        groups.append(ModelGroup(modelID: activeModel.id, requests: shown))
                    }
                }

                self.groups = groups
                self.queuedRequestCount = max(0, activeModels.totalWaitingRequests ?? 0)
                self.hiddenRequestCount = hiddenRequestCount
            }

            private static func requests(for model: ActiveModel) -> [Request] {
                let prefill = (model.prefilling ?? []).enumerated().map { index, progress in
                    Self.prefill(index: index, progress: progress)
                }
                let generation = (model.generating ?? []).enumerated().map { index, progress in
                    Self.generation(index: index, progress: progress)
                }
                let activity = (model.activities ?? []).enumerated().map { index, item in
                    Self.nonStreaming(index: index, activity: item)
                }
                return prefill + generation + activity
            }

            private static func prefill(index: Int, progress: PrefillProgress) -> Request {
                let processedTokens = max(0, progress.processed ?? 0)
                let totalTokens = max(0, progress.total ?? 0)
                let percentage = totalTokens > 0
                    ? Int((Double(processedTokens) / Double(totalTokens) * 100).rounded())
                    : 0

                return Request(
                    id: "prefill-\(index)",
                    title: "\(percentage)% · \(formatPrefillProgressTokenCount(processedTokens)) / \(formatPrefillProgressTokenCount(totalTokens)) tk",
                    detail: "\(ActivityFormat.rate(max(0, progress.speed ?? 0))) tk/s",
                    kind: .prefill
                )
            }

            private static func formatPrefillProgressTokenCount(_ count: Int) -> String {
                guard count >= 1_000 else { return "\(count)" }
                guard count < 10_000 else { return ActivityFormat.tokens(count) }
                return String(format: "%.1fk", Double(count) / 1_000)
            }

            private static func generation(index: Int, progress: GenerationProgress) -> Request {
                let tokensPerSecond = max(0, progress.tokensPerSecond ?? 0)
                let generatedTokens = max(0, progress.generatedTokens ?? 0)

                return Request(
                    id: "generation-\(index)",
                    title: "\(ActivityFormat.tokens(generatedTokens)) tk",
                    detail: "\(ActivityFormat.rate(tokensPerSecond)) tk/s",
                    kind: .generating
                )
            }

            private static func nonStreaming(index: Int, activity: NonStreamingActivity) -> Request {
                let elapsed = activity.elapsedSeconds.map(formatDuration)
                let activityDetail = activity.detail ?? activity.kind ?? "Active request"
                var detailParts = [activityDetail]
                if let elapsed {
                    detailParts.append(elapsed)
                }

                return Request(
                    id: "working-\(index)",
                    title: elapsed.map { "RUN \($0)" } ?? "RUN",
                    detail: detailParts.joined(separator: " · "),
                    kind: .nonStreaming
                )
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
    private let session: URLSession
    private var task: Task<Void, Never>?
    private var refreshInProgress = false
    /// Seconds between all-time fetches. All-time averages only change when a
    /// request completes and the endpoint is heavyweight (it also builds
    /// active_models/engines/runtime_cache), so it is never polled at the
    /// user-facing refresh interval: 5 s keeps an enabled ALL menubar item
    /// feeling live, 30 s is plenty for the Serving Stats submenu.
    private var alltimeRefreshInterval: TimeInterval {
        enabledMetrics.alltime ? 5 : 30
    }
    private var tickCount = 0
    private(set) var servingStatsSubmenuOpen = false
    private(set) var enabledMetrics = EnabledMetrics(
        live: false, average: false, alltime: false
    )
    private(set) var lastTickWasSuccess = false

    private(set) var sessionStats: Stats?
    private(set) var liveStats: Stats?
    private(set) var alltimeStats: Stats?
    private(set) var lastStatusSuccessAt: Date?

    init(
        baseURL: URL,
        apiKey: String?,
        interval: TimeInterval = 2.0,
        sessionConfiguration: URLSessionConfiguration? = nil
    ) {
        self.baseURL = baseURL
        self.apiKey = apiKey
        self.idleInterval = interval

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

    func setEnabledMetrics(_ metrics: EnabledMetrics) {
        guard enabledMetrics != metrics else {
            return
        }

        let alltimeTurnedOn = !needsAlltimeStats && (metrics.alltime || servingStatsSubmenuOpen)
        enabledMetrics = metrics
        if alltimeTurnedOn {
            // Force an all-time fetch on the next tick so a freshly enabled
            // ALL item doesn't sit on "–" for up to a full cadence period.
            tickCount = 0
        }
    }

    /// The hosted Serving Stats menu is a consumer independent of the optional
    /// LIV / AVG / ALL status items. Opening it starts the admin-backed scopes;
    /// closing it leaves only explicitly enabled metric items polling them.
    func setServingStatsSubmenuOpen(_ isOpen: Bool) {
        guard servingStatsSubmenuOpen != isOpen else {
            return
        }

        servingStatsSubmenuOpen = isOpen
        if isOpen {
            // The next pass fetches all-time data immediately instead of waiting
            // for the submenu's lower-frequency cadence.
            tickCount = 0
        }
    }

    /// Any enabled menubar metric item polls at the user-configured refresh
    /// interval (read live from UserDefaults so setting changes apply on the
    /// next loop pass); otherwise the idle 2 s cadence keeps public session
    /// data fresh without background admin traffic.
    var currentPollingInterval: TimeInterval {
        enabledMetrics.any ? MenubarMetricPrefs.refreshInterval : idleInterval
    }

    var needsLiveActivity: Bool {
        enabledMetrics.live || servingStatsSubmenuOpen
    }

    var needsAlltimeStats: Bool {
        enabledMetrics.alltime || servingStatsSubmenuOpen
    }

    deinit {
        // Detached cancel — actor-isolated stop() can't run from deinit.
        task?.cancel()
    }

    // MARK: - Polling

    func refreshOnce() async {
        // Opening the hosted submenu requests a prompt refresh. If it lands
        // during the timer's fetch, that fetch completes with the current
        // demand and the next timer pass picks up any newly enabled scope.
        guard !refreshInProgress else {
            return
        }
        refreshInProgress = true
        defer { refreshInProgress = false }

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
            do {
                if needsLiveActivity {
                    self.liveStats = try await fetchAdminActivity()
                }
            } catch {
                clearLiveStats(shouldPostUpdate: false)
            }
            if fetchAlltime, needsAlltimeStats, hasAPIKey,
               let alltime = try? await fetchAdminStats(scope: "alltime") {
                self.alltimeStats = alltime
            }
            lastTickWasSuccess = true
            NotificationCenter.default.post(
                name: Self.didUpdateNotification, object: self
            )
        } catch {
            // Suppress: server may be transitioning, paused, or 401-pending.
            // Next tick retries; we log only the once-per-tick failure mode.
            let wasSuccess = lastTickWasSuccess
            lastTickWasSuccess = false
            clearLiveStats(shouldPostUpdate: false)
            if wasSuccess {
                // Exactly one "server went away" repaint for the menubar
                // metric items, instead of silence or per-tick spam.
                NotificationCenter.default.post(
                    name: Self.didUpdateNotification, object: self
                )
            }
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

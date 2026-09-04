// Regression coverage for the menubar-shows-stale-port bug.
//
// Before the fix, MenubarController captured `config: AppConfig` by value
// at init and rendered `config.port` in the running-status header / port
// alert / Chat URL. The user-facing flow:
//   1. ServerScreen's Apply commits a new port via
//      `AppServices.applyServerEndpoint(port:)`.
//   2. AppServices calls `server.reconfigure(port:)` and restarts the
//      ServerProcess on the new port.
//   3. The server transitions to `.running(newPid)`; the menubar's
//      stateDidChange observer fires `refreshMenuState()`.
//   4. `refreshMenuState()` rebuilds the header — and read the OLD port
//      from the stale `config` snapshot. The user saw `:8080` after
//      changing to `:8964`.
//
// Fix: `MenubarController.displayPort(server:fallback:)` sources from
// the live server (which `reconfigure(port:)` updates), falling back to
// the captured config snapshot only when there is no server (bootstrap
// failed). These tests exercise the helper directly — instantiating the
// full controller in a unit test would require a live `NSStatusBar`.

import AppKit
import Foundation
import XCTest
@testable import oMLX

private final class MenubarStatsRequestRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var activityRequestCount = 0
    private var activityResponseStatusCode = 200
    private var publicStatusResponseStatusCode = 200
    private var updateNotificationCount = 0

    func reset() {
        lock.lock()
        defer { lock.unlock() }
        activityRequestCount = 0
        activityResponseStatusCode = 200
        publicStatusResponseStatusCode = 200
        updateNotificationCount = 0
    }

    func recordActivityRequest() {
        lock.lock()
        defer { lock.unlock() }
        activityRequestCount += 1
    }

    func recordedActivityRequestCount() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return activityRequestCount
    }

    func setActivityResponseStatusCode(_ statusCode: Int) {
        lock.lock()
        defer { lock.unlock() }
        activityResponseStatusCode = statusCode
    }

    func currentActivityResponseStatusCode() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return activityResponseStatusCode
    }

    func setPublicStatusResponseStatusCode(_ statusCode: Int) {
        lock.lock()
        defer { lock.unlock() }
        publicStatusResponseStatusCode = statusCode
    }

    func currentPublicStatusResponseStatusCode() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return publicStatusResponseStatusCode
    }

    func recordUpdateNotification() {
        lock.lock()
        defer { lock.unlock() }
        updateNotificationCount += 1
    }

    func recordedUpdateNotificationCount() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return updateNotificationCount
    }
}

private final class MenubarStatsURLProtocol: URLProtocol, @unchecked Sendable {
    private static let requestRecorder = MenubarStatsRequestRecorder()

    static func resetRequestRecorder() {
        requestRecorder.reset()
    }

    static func recordedActivityRequestCount() -> Int {
        requestRecorder.recordedActivityRequestCount()
    }

    static func setActivityResponseStatusCode(_ statusCode: Int) {
        requestRecorder.setActivityResponseStatusCode(statusCode)
    }

    static func setPublicStatusResponseStatusCode(_ statusCode: Int) {
        requestRecorder.setPublicStatusResponseStatusCode(statusCode)
    }

    static func recordUpdateNotification() {
        requestRecorder.recordUpdateNotification()
    }

    static func recordedUpdateNotificationCount() -> Int {
        requestRecorder.recordedUpdateNotificationCount()
    }

    override class func canInit(with request: URLRequest) -> Bool {
        true
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        guard let url = request.url else {
            client?.urlProtocolDidFinishLoading(self)
            return
        }

        let components = URLComponents(url: url, resolvingAgainstBaseURL: false)
        let scope = components?.queryItems?.first(where: { $0.name == "scope" })?.value
        let statusCode: Int
        if url.path == "/api/status" {
            statusCode = Self.requestRecorder.currentPublicStatusResponseStatusCode()
        } else if url.path == "/admin/api/activity" {
            Self.requestRecorder.recordActivityRequest()
            statusCode = Self.requestRecorder.currentActivityResponseStatusCode()
        } else {
            statusCode = 200
        }
        let payload: String
        switch (url.path, scope) {
        case ("/api/status", _):
            payload = #"{"total_prompt_tokens":99}"#
        case ("/admin/api/activity", _):
            payload = #"""
            {
              "active_models": {
                "models": [
                  {
                    "id": "Laguna XS.2",
                    "generating": [
                      {
                        "request_id": "generation-1",
                        "generated_tokens": 128,
                        "tokens_per_second": 42.1,
                        "elapsed_seconds": 3.0
                      }
                    ]
                  }
                ]
              }
            }
            """#
        case ("/admin/api/stats", "alltime"):
            payload = #"{"total_requests":3}"#
        default:
            payload = "{}"
        }

        let response = HTTPURLResponse(
            url: url,
            statusCode: statusCode,
            httpVersion: nil,
            headerFields: ["Content-Type": "application/json"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(payload.utf8))
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

@MainActor
final class MenubarControllerPortTests: XCTestCase {

    override func setUp() {
        super.setUp()
        MenubarStatsURLProtocol.resetRequestRecorder()
    }

    /// Test-only PythonRuntime. ServerProcess holds it but doesn't
    /// dereference until `start()` — these tests never start, they just
    /// read `.port` / `.host` after `reconfigure`.
    private func makeRuntime() -> PythonRuntime {
        PythonRuntime(
            executable: URL(fileURLWithPath: "/usr/bin/true"),
            homebrewPaths: [],
            pythonPath: [],
            pythonHome: nil,
            isBundled: false
        )
    }

    func testSpawnEnvironmentAdvertisesMenubarSupervisor() {
        let env = makeRuntime().makeEnvironment()
        XCTAssertEqual(env["OMLX_SUPERVISED"], "menubar")
    }

    func testLiveActivityPrioritizesPrefillOverGeneration() throws {
        let data = try XCTUnwrap(
            """
            {
              "active_models": {
                "models": [
                  {
                    "id": "Laguna XS.2",
                    "prefilling": [
                      {
                        "request_id": "prefill-1",
                        "processed": 12000,
                        "total": 32000,
                        "speed": 321.4,
                        "eta": 46.9
                      }
                    ],
                    "generating": [
                      {
                        "request_id": "generation-1",
                        "generated_tokens": 128,
                        "tokens_per_second": 42.1,
                        "elapsed_seconds": 3.0
                      }
                    ]
                  }
                ],
                "total_waiting_requests": 2
              }
            }
            """.data(using: .utf8)
        )

        let stats = try JSONDecoder().decode(MenubarStatsPoller.Stats.self, from: data)
        let activity = try XCTUnwrap(stats.liveActivity)

        XCTAssertEqual(activity.menuBarTitle, "PP 38% · 12k/32k")
        XCTAssertEqual(activity.detail, "Laguna XS.2 · 321 tok/s · 47s left")
    }

    func testLiveActivityShowsGenerationWhenNoPrefillIsActive() throws {
        let data = try XCTUnwrap(
            """
            {
              "active_models": {
                "models": [
                  {
                    "id": "Laguna XS.2",
                    "generating": [
                      {
                        "request_id": "generation-1",
                        "generated_tokens": 128,
                        "tokens_per_second": 42.1,
                        "elapsed_seconds": 3.0
                      }
                    ]
                  }
                ]
              }
            }
            """.data(using: .utf8)
        )

        let stats = try JSONDecoder().decode(MenubarStatsPoller.Stats.self, from: data)
        let activity = try XCTUnwrap(stats.liveActivity)

        XCTAssertEqual(activity.menuBarTitle, "GEN 42.1 tok/s")
        XCTAssertEqual(activity.detail, "Laguna XS.2 · 128 tok · 3s")
    }

    func testLiveActivityShowsQueuedRequestsWhenNoRequestIsRunning() throws {
        let data = try XCTUnwrap(
            """
            {
              "active_models": {
                "models": [],
                "total_waiting_requests": 2
              }
            }
            """.data(using: .utf8)
        )

        let stats = try JSONDecoder().decode(MenubarStatsPoller.Stats.self, from: data)
        let activity = try XCTUnwrap(stats.liveActivity)

        XCTAssertEqual(activity.menuBarTitle, "WAIT 2")
        XCTAssertEqual(activity.detail, "2 queued requests")
    }

    func testLiveActivityShowsNonStreamingEngineWork() throws {
        let data = try XCTUnwrap(
            """
            {
              "active_models": {
                "models": [
                  {
                    "id": "embed-model",
                    "activities": [
                      {
                        "kind": "embedding",
                        "detail": "Embedding",
                        "elapsed_seconds": 12.0
                      }
                    ]
                  }
                ]
              }
            }
            """.data(using: .utf8)
        )

        let stats = try JSONDecoder().decode(MenubarStatsPoller.Stats.self, from: data)
        let activity = try XCTUnwrap(stats.liveActivity)

        XCTAssertEqual(activity.menuBarTitle, "RUN 12s")
        XCTAssertEqual(activity.detail, "embed-model · Embedding · 12s")
    }

    func testRefreshOnceLoadsLiveActivityFromActivityEndpoint() async {
        let sessionConfiguration = URLSessionConfiguration.ephemeral
        sessionConfiguration.protocolClasses = [MenubarStatsURLProtocol.self]
        let poller = MenubarStatsPoller(
            baseURL: URL(string: "http://omlx.test")!,
            apiKey: "test-key",
            sessionConfiguration: sessionConfiguration
        )
        poller.setEnabledMetrics(EnabledMetrics(live: true, average: false, alltime: false))

        await poller.refreshOnce()

        XCTAssertEqual(poller.sessionStats?.totalPromptTokens, 99)
        XCTAssertEqual(poller.liveStats?.liveActivity?.menuBarTitle, "GEN 42.1 tok/s")
        XCTAssertEqual(poller.alltimeStats?.totalRequests, 3)
        XCTAssertEqual(MenubarStatsURLProtocol.recordedActivityRequestCount(), 1)
    }

    func testPollingFollowsRefreshIntervalOnlyWhileAMetricItemIsEnabled() {
        let defaults = UserDefaults.standard
        defaults.removeObject(forKey: MenubarMetricPrefs.refreshIntervalKey)
        defer { defaults.removeObject(forKey: MenubarMetricPrefs.refreshIntervalKey) }

        let poller = MenubarStatsPoller(
            baseURL: URL(string: "http://omlx.test")!,
            apiKey: "test-key"
        )

        XCTAssertEqual(poller.currentPollingInterval, 2.0)

        poller.setEnabledMetrics(EnabledMetrics(live: true, average: false, alltime: false))
        XCTAssertEqual(poller.currentPollingInterval, 1.0, "absent pref defaults to 1 s")

        defaults.set(0.5, forKey: MenubarMetricPrefs.refreshIntervalKey)
        XCTAssertEqual(poller.currentPollingInterval, 0.5)

        defaults.set(42.0, forKey: MenubarMetricPrefs.refreshIntervalKey)
        XCTAssertEqual(poller.currentPollingInterval, 1.0, "out-of-set values clamp to 1 s")

        poller.setEnabledMetrics(EnabledMetrics(live: false, average: true, alltime: false))
        defaults.set(3.0, forKey: MenubarMetricPrefs.refreshIntervalKey)
        XCTAssertEqual(
            poller.currentPollingInterval, 3.0,
            "any enabled metric item drives the configured cadence, not just live"
        )

        poller.setEnabledMetrics(EnabledMetrics(live: false, average: false, alltime: false))
        XCTAssertEqual(poller.currentPollingInterval, 2.0)
    }

    func testRefreshOnceLoadsActivityWithoutAPIKeyWhenServerAllowsIt() async {
        let sessionConfiguration = URLSessionConfiguration.ephemeral
        sessionConfiguration.protocolClasses = [MenubarStatsURLProtocol.self]
        let poller = MenubarStatsPoller(
            baseURL: URL(string: "http://omlx.test")!,
            apiKey: nil,
            sessionConfiguration: sessionConfiguration
        )
        poller.setEnabledMetrics(EnabledMetrics(live: true, average: false, alltime: false))

        await poller.refreshOnce()

        XCTAssertEqual(poller.liveStats?.liveActivity?.menuBarTitle, "GEN 42.1 tok/s")
        XCTAssertEqual(MenubarStatsURLProtocol.recordedActivityRequestCount(), 1)
    }

    func testRefreshOncePostsOneConsolidatedUpdateNotification() async {
        let sessionConfiguration = URLSessionConfiguration.ephemeral
        sessionConfiguration.protocolClasses = [MenubarStatsURLProtocol.self]
        let poller = MenubarStatsPoller(
            baseURL: URL(string: "http://omlx.test")!,
            apiKey: "test-key",
            sessionConfiguration: sessionConfiguration
        )
        poller.setEnabledMetrics(EnabledMetrics(live: true, average: false, alltime: false))
        let observer = NotificationCenter.default.addObserver(
            forName: MenubarStatsPoller.didUpdateNotification,
            object: poller,
            queue: nil
        ) { _ in
            MenubarStatsURLProtocol.recordUpdateNotification()
        }
        defer { NotificationCenter.default.removeObserver(observer) }

        await poller.refreshOnce()

        XCTAssertEqual(MenubarStatsURLProtocol.recordedUpdateNotificationCount(), 1)
    }

    func testRefreshOnceSkipsLiveAdminStatsWhenActivityDisplayIsDisabled() async {
        let sessionConfiguration = URLSessionConfiguration.ephemeral
        sessionConfiguration.protocolClasses = [MenubarStatsURLProtocol.self]
        let poller = MenubarStatsPoller(
            baseURL: URL(string: "http://omlx.test")!,
            apiKey: "test-key",
            sessionConfiguration: sessionConfiguration
        )

        poller.setEnabledMetrics(EnabledMetrics(live: false, average: false, alltime: false))
        await poller.refreshOnce()

        XCTAssertEqual(poller.sessionStats?.totalPromptTokens, 99)
        XCTAssertNil(poller.liveStats)
        XCTAssertEqual(MenubarStatsURLProtocol.recordedActivityRequestCount(), 0)
    }

    func testRefreshOnceClearsLiveActivityAfterActivityFailure() async {
        let sessionConfiguration = URLSessionConfiguration.ephemeral
        sessionConfiguration.protocolClasses = [MenubarStatsURLProtocol.self]
        let poller = MenubarStatsPoller(
            baseURL: URL(string: "http://omlx.test")!,
            apiKey: "test-key",
            sessionConfiguration: sessionConfiguration
        )
        poller.setEnabledMetrics(EnabledMetrics(live: true, average: false, alltime: false))

        await poller.refreshOnce()
        XCTAssertNotNil(poller.liveStats?.liveActivity)

        MenubarStatsURLProtocol.setActivityResponseStatusCode(500)
        await poller.refreshOnce()

        XCTAssertNil(poller.liveStats)
    }

    func testRefreshOnceClearsLiveActivityAfterPublicStatusFailure() async {
        let sessionConfiguration = URLSessionConfiguration.ephemeral
        sessionConfiguration.protocolClasses = [MenubarStatsURLProtocol.self]
        let poller = MenubarStatsPoller(
            baseURL: URL(string: "http://omlx.test")!,
            apiKey: "test-key",
            sessionConfiguration: sessionConfiguration
        )
        poller.setEnabledMetrics(EnabledMetrics(live: true, average: false, alltime: false))

        await poller.refreshOnce()
        XCTAssertNotNil(poller.liveStats?.liveActivity)

        MenubarStatsURLProtocol.setPublicStatusResponseStatusCode(500)
        await poller.refreshOnce()

        XCTAssertNil(poller.liveStats)
    }

    func testRefreshOncePostsExactlyOneNotificationWhenServerGoesAway() async {
        let sessionConfiguration = URLSessionConfiguration.ephemeral
        sessionConfiguration.protocolClasses = [MenubarStatsURLProtocol.self]
        let poller = MenubarStatsPoller(
            baseURL: URL(string: "http://omlx.test")!,
            apiKey: "test-key",
            sessionConfiguration: sessionConfiguration
        )
        let observer = NotificationCenter.default.addObserver(
            forName: MenubarStatsPoller.didUpdateNotification,
            object: poller,
            queue: nil
        ) { _ in
            MenubarStatsURLProtocol.recordUpdateNotification()
        }
        defer { NotificationCenter.default.removeObserver(observer) }

        await poller.refreshOnce()
        XCTAssertEqual(MenubarStatsURLProtocol.recordedUpdateNotificationCount(), 1)

        // First failing tick after a success: one "server went away" repaint…
        MenubarStatsURLProtocol.setPublicStatusResponseStatusCode(500)
        await poller.refreshOnce()
        XCTAssertEqual(MenubarStatsURLProtocol.recordedUpdateNotificationCount(), 2)

        // …then silence while the server stays down (no per-tick spam).
        await poller.refreshOnce()
        XCTAssertEqual(MenubarStatsURLProtocol.recordedUpdateNotificationCount(), 2)
    }

    // MARK: - displayPort

    func testDisplayPortFallsBackToConfigWhenNoServer() {
        XCTAssertEqual(
            MenubarController.displayPort(server: nil, fallback: 8080),
            8080,
            "With no server (bootstrap failed), the displayed port must come from the AppConfig snapshot."
        )
    }

    func testDisplayPortPrefersLiveServer() {
        let server = ServerProcess(runtime: makeRuntime(), port: 8888)
        XCTAssertEqual(
            MenubarController.displayPort(server: server, fallback: 8080),
            8888,
            "When a server is present, its `port` is authoritative — `fallback` is only for the no-server case."
        )
    }

    func testDisplayPortFollowsReconfigure() throws {
        // The original bug: menubar's `config.port` snapshot never sees
        // this change, so the running-header text keeps showing 8080.
        let server = ServerProcess(runtime: makeRuntime(), port: 8080)
        try server.reconfigure(port: 8964)
        XCTAssertEqual(
            MenubarController.displayPort(server: server, fallback: 8080),
            8964,
            "After Server screen's Apply commits a new port (which calls server.reconfigure(port:)), the menubar must source from the live server."
        )
    }

    // MARK: - displayHost

    func testDisplayHostFallsBackToConfigWhenNoServer() {
        XCTAssertEqual(
            MenubarController.displayHost(server: nil, fallback: "127.0.0.1"),
            "127.0.0.1"
        )
    }

    func testDisplayHostPrefersLiveServer() {
        let server = ServerProcess(runtime: makeRuntime(), bindAddress: "127.0.0.1", port: 8080)
        XCTAssertEqual(
            MenubarController.displayHost(server: server, fallback: "127.0.0.1"),
            "127.0.0.1"
        )
    }

    func testDisplayHostUsesServerConnectableHost() {
        let server = ServerProcess(runtime: makeRuntime(), bindAddress: "0.0.0.0", port: 8080)
        XCTAssertEqual(
            MenubarController.displayHost(server: server, fallback: "127.0.0.1"),
            "127.0.0.1",
            "ServerProcess.host returns the connectable host (0.0.0.0 → 127.0.0.1)."
        )
    }

    func testDisplayHostFollowsReconfigure() throws {
        let server = ServerProcess(runtime: makeRuntime(), bindAddress: "127.0.0.1", port: 8080)
        try server.reconfigure(bindAddress: "localhost")
        XCTAssertEqual(
            MenubarController.displayHost(server: server, fallback: "127.0.0.1"),
            "127.0.0.1",
            "Listen Address changes propagate through ServerProcess.host, which returns the connectable loopback host."
        )
    }

    func testDisplayHostHandlesCommaSeparatedBindAddress() throws {
        let server = ServerProcess(
            runtime: makeRuntime(),
            bindAddress: "0.0.0.0,127.0.0.1",
            port: 8080
        )
        XCTAssertEqual(
            MenubarController.displayHost(server: server, fallback: "127.0.0.1"),
            "127.0.0.1",
            "The menubar should use the first configured bind host and normalize wildcards before building URLs."
        )
    }

    // MARK: - webAdminURL
    //
    // The "Open Web Dashboard" menubar item routes through the server's
    // /admin/auto-login endpoint so the dashboard opens without the manual
    // login form. The action method itself needs a live NSStatusBar, so we
    // test the pure URL builder it delegates to.

    func testWebAdminURLUsesAutoLoginWithRedirect() throws {
        let url = try XCTUnwrap(
            MenubarController.webAdminURL(host: "127.0.0.1", port: 8000, apiKey: "secret")
        )
        let comps = try XCTUnwrap(URLComponents(url: url, resolvingAgainstBaseURL: false))
        XCTAssertEqual(comps.scheme, "http")
        XCTAssertEqual(comps.host, "127.0.0.1")
        XCTAssertEqual(comps.port, 8000)
        XCTAssertEqual(comps.path, "/admin/auto-login")
        let items = comps.queryItems ?? []
        XCTAssertEqual(items.first { $0.name == "redirect" }?.value, "/admin/dashboard")
        XCTAssertEqual(items.first { $0.name == "key" }?.value, "secret")
    }

    func testWebAdminURLBuildsIPv6Host() throws {
        let url = try XCTUnwrap(
            MenubarController.webAdminURL(host: "[::1]", port: 8000, apiKey: nil)
        )
        XCTAssertTrue(url.absoluteString.hasPrefix("http://[::1]:8000/admin/auto-login"))
    }

    func testWebAdminURLPercentEncodesKey() throws {
        // A key with URL-reserved characters must survive intact — raw
        // string interpolation would corrupt it; URLComponents encodes it.
        let url = try XCTUnwrap(
            MenubarController.webAdminURL(host: "127.0.0.1", port: 8000, apiKey: "a+b/c&d")
        )
        // The decoded query item value round-trips to the original key.
        let comps = try XCTUnwrap(URLComponents(url: url, resolvingAgainstBaseURL: false))
        XCTAssertEqual(comps.queryItems?.first { $0.name == "key" }?.value, "a+b/c&d")
        // And the raw URL string carries the encoded form, not the literal.
        XCTAssertTrue(url.absoluteString.contains("key=a%2Bb/c%26d"),
                      "key should be percent-encoded in the URL string, got \(url.absoluteString)")
    }

    func testWebAdminURLOmitsKeyWhenMissing() throws {
        for key in [nil, ""] as [String?] {
            let url = try XCTUnwrap(
                MenubarController.webAdminURL(host: "127.0.0.1", port: 8000, apiKey: key)
            )
            let comps = try XCTUnwrap(URLComponents(url: url, resolvingAgainstBaseURL: false))
            XCTAssertNil(comps.queryItems?.first { $0.name == "key" },
                         "empty/nil key must not emit a key= param (server redirects to login instead)")
            XCTAssertEqual(comps.queryItems?.first { $0.name == "redirect" }?.value,
                           "/admin/dashboard")
        }
    }

    // MARK: - menuAvailability

    func testMenuAvailabilityKeepsSettingsEnabledWhenServerIsOffline() {
        for state in [ServerProcess.State.stopped, .failed(message: "Port 8000 in use")] {
            let availability = MenubarController.menuAvailability(for: state)
            XCTAssertTrue(availability.settings)
            XCTAssertFalse(availability.webDashboard)
            XCTAssertFalse(availability.chat)
        }
    }

    func testMenuAvailabilityEnablesBrowserItemsOnlyWhenRunning() {
        let availability = MenubarController.menuAvailability(for: .running(pid: 123))
        XCTAssertTrue(availability.settings)
        XCTAssertTrue(availability.webDashboard)
        XCTAssertTrue(availability.chat)
    }

    func testMenuAvailabilityKeepsBrowserItemsDisabledDuringTransitions() {
        let states: [ServerProcess.State] = [
            .starting,
            .stopping,
            .unresponsive(pid: 123),
        ]

        for state in states {
            let availability = MenubarController.menuAvailability(for: state)
            XCTAssertTrue(availability.settings)
            XCTAssertFalse(availability.webDashboard)
            XCTAssertFalse(availability.chat)
        }
    }

    // MARK: - failure alerts

    func testGenericFailureAlertSkipsPortConflictMessages() {
        XCTAssertFalse(
            MenubarController.shouldShowGenericFailureAlert(message: "Port 8000 in use")
        )
        XCTAssertTrue(
            MenubarController.shouldShowGenericFailureAlert(
                message: "Server exited with code 1 during startup"
            )
        )
    }

    func testAccessFailureHintDetectsPermissionErrors() {
        XCTAssertNotNil(
            MenubarController.accessFailureHint(
                message: "Server exited with code 1 during startup",
                logTail: "PermissionError: [Errno 1] Operation not permitted"
            )
        )
        XCTAssertNil(
            MenubarController.accessFailureHint(
                message: "Server exited with code 1 during startup",
                logTail: "ValueError: no models found"
            )
        )
    }
}

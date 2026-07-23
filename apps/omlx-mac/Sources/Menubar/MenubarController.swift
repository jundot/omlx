// PR 4 — full menubar parity port. Mirrors the Python menu construction
// (app.py:1450-1700) and refresh strategy (menuWillOpen + per-second poll).
//
// Items, top-down:
//   • Status header                     (colored, non-clickable)
//   • Force Restart   (UNRESPONSIVE/ERROR only)
//   • Stop Server     (RUNNING / STARTING / STOPPING / UNRESPONSIVE)
//   • Start Server    (STOPPED / IDLE / FAILED)
//   • Serving Stats     (Session + All-Time submenu)
//   • Open Web Dashboard (enabled when running — opens the web admin
//                        dashboard in the browser via /admin/auto-login)
//   • Chat with oMLX    (enabled when running — opens /admin/chat in browser)
//   • Settings…         (Cmd-, — opens the SwiftUI AppView window via the
//                        openAppView callback; available even when stopped)
//   • About oMLX
//   • Quit oMLX       (Cmd-Q)
//
// Icon templates: MenubarOutline (stopped) / MenubarFilled (running). Stats
// poll uses lightweight /api/status with occasional all-time admin stats;
// visibility watcher probes once at +3 s post-launch with a single
// recreate-and-retry before alerting.

import AppKit
import SwiftUI

@MainActor
final class MenubarController: NSObject {

    // MARK: - Inputs / state

    private let server: ServerProcess?
    private let config: AppConfig
    private let updates: UpdateController?
    private let bootstrapError: Error?
    private let client: OMLXClient?
    private let openModelSettings: (String) -> Void
    private let openAppView: () -> Void
    private let openAppearanceSettings: () -> Void
    private let requestQuit: () -> Void

    private var statusItem: NSStatusItem
    private let menu = NSMenu()

    /// Shared throughput model + the optional LIV/AVG/ALL status items it
    /// feeds. The store outlives poller re-points (port changes) so the
    /// popover graphs keep their history.
    private let metricsStore = MenubarMetricsStore()
    private var metricItemsController: MenubarMetricItemsController!

    private var statsPoller: MenubarStatsPoller?
    /// Endpoint the live `statsPoller` was started against, so a runtime
    /// host/port change can detect divergence and re-point the poller.
    private var statsPollerBaseURL: URL?
    private var visibilityWatcher: MenubarVisibilityWatcher?
    private var lastPresentedFailureMessage: String?
    private var lastPresentedPortConflictKey: String?

    // Strong refs to dynamic menu items so refreshMenuState() can edit
    // without rebuilding the live NSMenu (matches Python's
    // _refresh_menu_in_place — safe while menu is open).
    private var statusHeader: NSMenuItem!
    private var startItem: NSMenuItem!
    private var stopItem: NSMenuItem!
    private var restartItem: NSMenuItem!
    private var modelsParentItem: NSMenuItem!
    private var modelsSubmenu: NSMenu!
    private var models: [ModelDTO] = []
    private var unloadingIDs: Set<String> = []
    private var loadingIDs: Set<String> = []
    /// False until the first successful model-list fetch after a server start,
    /// so the empty submenu can say "Loading…" instead of "No models available"
    /// while the initial fetch is still in flight.
    private var modelsFetched = false
    private var modelsFetchTask: Task<Void, Never>?
    private var statsParentItem: NSMenuItem!
    private var statsSubmenu: NSMenu!
    private var systemStatsParentItem: NSMenuItem!
    private var systemStatsSubmenu: NSMenu!
    private let systemStatsSampler = SystemStatsSampler()
    private var systemItemsController: SystemMenubarItemsController!
    private var systemStatsTimer: Timer?
    private var systemStatsTimerInterval: TimeInterval = 0
    private var systemStatsSubmenuOpen = false
    private var lastSystemTickAt: Date?
    private var cpuPanelHost: NSHostingView<AnyView>?
    private var gpuPanelHost: NSHostingView<AnyView>?
    private var memoryPanelHost: NSHostingView<AnyView>?
    private var adminPanelItem: NSMenuItem!
    private var webAdminItem: NSMenuItem!
    private var chatItem: NSMenuItem!
    private var updateItem: NSMenuItem!

    private let iconOutline: NSImage?
    private let iconFilled: NSImage?

    // MARK: - Init

    init(
        server: ServerProcess?,
        config: AppConfig,
        updates: UpdateController? = nil,
        lastError: Error? = nil,
        client: OMLXClient? = nil,
        openModelSettings: @escaping (String) -> Void = { _ in },
        openAppView: @escaping () -> Void = {},
        openAppearanceSettings: @escaping () -> Void = {},
        requestQuit: @escaping () -> Void = { NSApp.terminate(nil) }
    ) {
        self.server = server
        self.config = config
        self.updates = updates
        self.bootstrapError = lastError
        self.client = client
        self.openModelSettings = openModelSettings
        self.openAppView = openAppView
        self.openAppearanceSettings = openAppearanceSettings
        self.requestQuit = requestQuit

        self.statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)

        // Cap icons at 18×18 pt (the standard macOS menubar icon size).
        // Our SVGs are 497×497 natural; without this, the status item
        // auto-sizes to that natural width and dominates the menubar.
        // Mirrors Python's _load_menubar_icon (app.py:973).
        // SF Symbol fallback covers an asset-catalog miss in Debug builds.
        let menubarIconSize = NSSize(width: 18, height: 18)
        let fallback = NSImage(
            systemSymbolName: "cube.transparent",
            accessibilityDescription: "oMLX"
        )

        let outline = NSImage(named: "MenubarOutline") ?? fallback
        outline?.size = menubarIconSize
        outline?.isTemplate = true
        self.iconOutline = outline

        let filled = NSImage(named: "MenubarFilled") ?? fallback
        filled?.size = menubarIconSize
        filled?.isTemplate = true
        self.iconFilled = filled

        super.init()

        statusItem.behavior = []
        statusItem.menu = menu
        statusItem.button?.image = outline
        statusItem.button?.setAccessibilityLabel("oMLX")
        statusItem.button?.toolTip = "oMLX"
        // This menu is state-driven by refreshMenuState(). If AppKit's
        // automatic target/action enabling stays on, stopped-server items
        // such as Web Dashboard and Chat can be re-enabled while opening.
        menu.autoenablesItems = false
        menu.delegate = self

        metricItemsController = MenubarMetricItemsController(
            store: metricsStore,
            openAppearanceSettings: { [weak self] in
                self?.openAppearanceSettings()
            },
            openDashboard: { [weak self] in
                self?.openWebAdmin()
            }
        )
        systemItemsController = SystemMenubarItemsController()
        metricItemsController.willShowPopover = { [weak self] in
            self?.systemItemsController.closeAllPopovers()
        }
        systemItemsController.willShowPopover = { [weak self] in
            self?.metricItemsController.closeAllPopovers()
        }

        buildMenu()
        refreshMenuState()

        // Appearance toggles are written by the settings screen (and only
        // read here); the defaults notification is the one propagation path
        // that also works while the server is down and the poller is silent.
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(defaultsDidChange(_:)),
            name: UserDefaults.didChangeNotification,
            object: UserDefaults.standard
        )

        if let server {
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(serverStateChanged(_:)),
                name: ServerProcess.stateDidChangeNotification,
                object: server
            )
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(serverPortConflict(_:)),
                name: ServerProcess.portConflictNotification,
                object: server
            )
        }
        if let updates {
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(updateStateChanged(_:)),
                name: UpdateController.stateDidChangeNotification,
                object: updates
            )
        }

        startStatsPoller()
        startVisibilityWatcher()
        applyMetricPreferences()

        if let bootstrapError {
            DispatchQueue.main.async { [weak self] in
                self?.presentServerFailureAlert(
                    message: String(describing: bootstrapError),
                    logURL: ServerProcess.defaultLogURL()
                )
            }
        }
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
    }

    // MARK: - Menu construction

    private func buildMenu() {
        menu.removeAllItems()

        statusHeader = NSMenuItem(
            title: String(localized: "menubar.header.loading",
                          defaultValue: "Server: …",
                          comment: "Initial menubar header text before the server state is known"),
            action: nil,
            keyEquivalent: ""
        )
        statusHeader.isEnabled = false
        menu.addItem(statusHeader)

        menu.addItem(.separator())

        restartItem = item(String(localized: "menubar.item.force_restart",
                                  defaultValue: "Force Restart",
                                  comment: "Menubar item that force-restarts a stuck or failed server"),
                           action: #selector(forceRestartServer),
                           symbol: "arrow.clockwise.circle")
        menu.addItem(restartItem)

        stopItem = item(String(localized: "menubar.item.stop_server",
                               defaultValue: "Stop Server",
                               comment: "Menubar item that stops the running server"),
                        action: #selector(stopServer),
                        symbol: "stop.circle")
        menu.addItem(stopItem)

        startItem = item(String(localized: "menubar.item.start_server",
                                defaultValue: "Start Server",
                                comment: "Menubar item that starts the server"),
                         action: #selector(startServer),
                         symbol: "play.circle")
        menu.addItem(startItem)

        modelsParentItem = item(String(localized: "menubar.item.models",
                                       defaultValue: "Models",
                                       comment: "Menubar parent item opening the per-model control submenu"),
                                action: nil,
                                symbol: "shippingbox")
        modelsSubmenu = NSMenu()
        modelsSubmenu.autoenablesItems = false
        modelsParentItem.submenu = modelsSubmenu
        menu.addItem(modelsParentItem)

        menu.addItem(.separator())

        systemStatsParentItem = item(String(localized: "menubar.item.system_stats",
                                            defaultValue: "System Stats",
                                            comment: "Menubar parent item opening the CPU/GPU/Memory submenu"),
                                     action: nil,
                                     symbol: "cpu")
        systemStatsSubmenu = NSMenu()
        systemStatsSubmenu.autoenablesItems = false
        systemStatsSubmenu.delegate = self
        systemStatsParentItem.submenu = systemStatsSubmenu
        buildSystemStatsSubmenu()
        menu.addItem(systemStatsParentItem)

        statsParentItem = item(String(localized: "menubar.item.serving_stats",
                                      defaultValue: "Serving Stats",
                                      comment: "Menubar parent item opening the Serving Stats submenu"),
                               action: nil,
                               symbol: "chart.bar")
        statsSubmenu = NSMenu()
        statsParentItem.submenu = statsSubmenu
        menu.addItem(statsParentItem)
        rebuildStatsSubmenu()
        rebuildModelsSubmenu()

        menu.addItem(.separator())

        webAdminItem = item(String(localized: "menubar.item.web_dashboard",
                                   defaultValue: "Open Web Dashboard",
                                   comment: "Menubar item that opens the browser-based web admin dashboard with auto-login"),
                            action: #selector(openWebAdmin),
                            symbol: "globe")
        menu.addItem(webAdminItem)

        chatItem = item(String(localized: "menubar.item.chat",
                               defaultValue: "Chat with oMLX",
                               comment: "Menubar item that opens the browser-based chat dashboard"),
                        action: #selector(openChat),
                        symbol: "message")
        menu.addItem(chatItem)

        menu.addItem(.separator())

        updateItem = item(String(localized: "menubar.item.update_available",
                                 defaultValue: "Install Update…",
                                 comment: "Menubar item shown when an app update is available"),
                          action: #selector(installUpdate),
                          symbol: "arrow.down.circle")
        menu.addItem(updateItem)

        adminPanelItem = item(String(localized: "menubar.item.settings",
                                     defaultValue: "Settings…",
                                     comment: "Menubar item that opens the native settings/preferences window"),
                              action: #selector(openAdminPanel),
                              symbol: "gearshape",
                              keyEquivalent: ",")
        menu.addItem(adminPanelItem)

        let about = item(String(localized: "menubar.item.about",
                                defaultValue: "About oMLX",
                                comment: "Menubar item that opens the standard About window"),
                         action: #selector(showAbout),
                         symbol: "info.circle")
        menu.addItem(about)

        menu.addItem(.separator())

        let quit = item(String(localized: "menubar.item.quit",
                               defaultValue: "Quit oMLX",
                               comment: "Menubar item that terminates the app (Cmd-Q)"),
                        action: #selector(quitApp),
                        symbol: "power",
                        keyEquivalent: "q")
        menu.addItem(quit)
    }

    private func item(
        _ title: String,
        action: Selector?,
        symbol: String?,
        keyEquivalent: String = ""
    ) -> NSMenuItem {
        let item = NSMenuItem(title: title, action: action, keyEquivalent: keyEquivalent)
        item.target = (action != nil) ? self : nil
        if let symbol,
           let img = NSImage(systemSymbolName: symbol, accessibilityDescription: nil)
        {
            img.isTemplate = true
            item.image = img
        }
        return item
    }

    // MARK: - Refresh

    private func refreshMenuState() {
        let state = server?.state ?? .stopped
        let isRunning = serverIsRunning
        let isStarting: Bool
        if case .starting = state { isStarting = true } else { isStarting = false }
        let isStopping: Bool
        if case .stopping = state { isStopping = true } else { isStopping = false }
        let isUnresponsive: Bool
        if case .unresponsive = state { isUnresponsive = true } else { isUnresponsive = false }
        let isFailed: Bool
        if case .failed = state { isFailed = true } else { isFailed = false }

        // Status header
        let (text, color) = headerDisplay(state)
        statusHeader.attributedTitle = NSAttributedString(
            string: text,
            attributes: [.foregroundColor: color]
        )

        // Server-control item visibility — mirrors server_manager.py:
        //   STOPPED/FAILED → Start
        //   RUNNING/STARTING/STOPPING/UNRESPONSIVE → Stop
        //   UNRESPONSIVE/FAILED → Force Restart
        let liveLike = isRunning || isStarting || isStopping || isUnresponsive
        startItem.isHidden = liveLike
        stopItem.isHidden = !liveLike
        restartItem.isHidden = !(isFailed || isUnresponsive)

        // Disabled when no server bootstrap (ServerProcess is nil) or in
        // a transitional state we shouldn't double-trigger.
        startItem.isEnabled = (server != nil) && !liveLike
        stopItem.isEnabled = liveLike && !isStopping

        modelsParentItem.isEnabled = isRunning

        // Native Settings is the recovery surface for stopped/failed servers.
        // Web Dashboard / Chat open browser URLs against the live port, so
        // they stay gated on a healthy running server.
        let availability = MenubarController.menuAvailability(for: state)
        adminPanelItem.isEnabled = availability.settings
        webAdminItem.isEnabled = availability.webDashboard
        chatItem.isEnabled = availability.chat

        refreshUpdateMenuItem()

        // Icon swap — outline when not actively serving, filled otherwise.
        // Live throughput lives in the dedicated LIV/AVG/ALL metric items,
        // so the main item stays a fixed square icon.
        let serving = state.isRunningLike
        statusItem.button?.image = serving ? iconFilled : iconOutline
    }

    private func refreshUpdateMenuItem() {
        guard let updates else {
            updateItem.isHidden = true
            return
        }
        switch updates.state {
        case .available(let info), .ready(let info):
            updateItem.isHidden = false
            updateItem.isEnabled = info.dmgURL != nil
            updateItem.title = String(localized: "menubar.item.install_update_version",
                                      defaultValue: "Install oMLX \(info.version)…",
                                      comment: "Menubar update item when an app update is available; placeholder is the version")
        case .downloading(let pct):
            updateItem.isHidden = false
            updateItem.isEnabled = false
            updateItem.title = String(localized: "menubar.item.downloading_update",
                                      defaultValue: "Downloading update… \(pct)%",
                                      comment: "Menubar update item while an app update is downloading; placeholder is the percent")
        default:
            updateItem.isHidden = true
            updateItem.isEnabled = false
        }
    }

    private func headerDisplay(_ state: ServerProcess.State) -> (String, NSColor) {
        switch state {
        case .stopped:
            if let err = bootstrapError {
                return (
                    String(localized: "menubar.header.bootstrap_failed",
                           defaultValue: "Server: bootstrap failed (\(String(describing: err)))",
                           comment: "Menubar status header when the server bootstrap threw an error; placeholder is the error description"),
                    .systemRed
                )
            }
            return (
                String(localized: "menubar.header.stopped",
                       defaultValue: "oMLX stopped",
                       comment: "Menubar status header when the server is stopped"),
                .secondaryLabelColor
            )
        case .starting:
            return (
                String(localized: "menubar.header.starting",
                       defaultValue: "Server: starting…",
                       comment: "Menubar status header while the server is starting"),
                .systemBlue
            )
        case .running:
            let port = MenubarController.displayPort(server: server, fallback: config.port)
            return (
                String(localized: "menubar.header.running",
                       defaultValue: "Server: running (port \(String(port)))",
                       comment: "Menubar status header when the server is running; placeholder is the port (rendered as a plain integer, no grouping)"),
                .systemGreen
            )
        case .stopping:
            return (
                String(localized: "menubar.header.stopping",
                       defaultValue: "Server: stopping…",
                       comment: "Menubar status header while the server is stopping"),
                .systemOrange
            )
        case .unresponsive:
            return (
                String(localized: "menubar.header.unresponsive",
                       defaultValue: "Server: unresponsive (auto-recover or Force Restart)",
                       comment: "Menubar status header when the server is unresponsive"),
                .systemOrange
            )
        case .failed(let msg):
            return (
                String(localized: "menubar.header.failed",
                       defaultValue: "Server: failed — \(msg)",
                       comment: "Menubar status header when the server failed; placeholder is the failure message"),
                .systemRed
            )
        }
    }

    // MARK: - System Stats submenu

    /// CPU / GPU / Memory panels as custom-view items. The hosting views
    /// are created once; the sampling timer swaps their root views with
    /// fresh snapshots only while the submenu is open.
    private func buildSystemStatsSubmenu() {
        systemStatsSubmenu.removeAllItems()

        let snapshot = SystemStatsSnapshot()
        let interval = MenubarMetricPrefs.refreshInterval
        let cpu = NSHostingView(rootView: AnyView(
            CPUStatsPanel(snapshot: snapshot, refreshInterval: interval).omlxThemed()
        ))
        let gpu = NSHostingView(rootView: AnyView(
            GPUStatsPanel(snapshot: snapshot, refreshInterval: interval).omlxThemed()
        ))
        let memory = NSHostingView(rootView: AnyView(
            MemoryStatsPanel(snapshot: snapshot).omlxThemed()
        ))
        cpuPanelHost = cpu
        gpuPanelHost = gpu
        memoryPanelHost = memory

        systemStatsSubmenu.addItem(panelItem(hosting: cpu))
        systemStatsSubmenu.addItem(.separator())
        systemStatsSubmenu.addItem(panelItem(hosting: gpu))
        systemStatsSubmenu.addItem(.separator())
        systemStatsSubmenu.addItem(panelItem(hosting: memory))
    }

    private func panelItem(hosting view: NSHostingView<AnyView>) -> NSMenuItem {
        view.frame.size = view.fittingSize
        let item = NSMenuItem()
        item.view = view
        item.isEnabled = false
        return item
    }

    /// One sampler timer serves both consumers: the System Stats submenu
    /// panels (while open) and the CPU/GPU/MEM status items (while any is
    /// enabled). It runs only when at least one consumer needs it, and joins
    /// the common run-loop modes because menu tracking parks the main run
    /// loop in the event-tracking mode, where default-mode timers (and
    /// main-queue hops) never fire.
    private var needsSystemSampling: Bool {
        systemStatsSubmenuOpen || MenubarMetricPrefs.enabledSystemItems.any
    }

    private func reconcileSystemSampling() {
        let interval = MenubarMetricPrefs.refreshInterval
        if needsSystemSampling {
            if systemStatsTimer == nil || systemStatsTimerInterval != interval {
                startSystemSamplingTimer(interval: interval)
            }
        } else if systemStatsTimer != nil {
            systemStatsTimer?.invalidate()
            systemStatsTimer = nil
        }
    }

    private func startSystemSamplingTimer(interval: TimeInterval) {
        systemStatsTimer?.invalidate()
        systemStatsTimerInterval = interval
        systemSamplingTick()
        let timer = Timer(timeInterval: interval, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.systemSamplingTick()
            }
        }
        RunLoop.main.add(timer, forMode: .common)
        systemStatsTimer = timer
    }

    private func systemSamplingTick() {
        // Timer start and menu-open both request an immediate paint; a
        // back-to-back double sample would produce a zero-length CPU tick
        // delta (no usable reading), so collapse them.
        if let last = lastSystemTickAt, Date().timeIntervalSince(last) < 0.1 {
            return
        }
        lastSystemTickAt = Date()
        let snapshot = systemStatsSampler.sample()
        if systemStatsSubmenuOpen {
            updateSystemStatsPanels(with: snapshot)
        }
        systemItemsController.apply(snapshot)
    }

    private func updateSystemStatsPanels(with snapshot: SystemStatsSnapshot) {
        let interval = MenubarMetricPrefs.refreshInterval
        cpuPanelHost?.rootView = AnyView(
            CPUStatsPanel(snapshot: snapshot, refreshInterval: interval).omlxThemed()
        )
        gpuPanelHost?.rootView = AnyView(
            GPUStatsPanel(snapshot: snapshot, refreshInterval: interval).omlxThemed()
        )
        memoryPanelHost?.rootView = AnyView(
            MemoryStatsPanel(snapshot: snapshot).omlxThemed()
        )
        for host in [cpuPanelHost, gpuPanelHost, memoryPanelHost] {
            guard let host else { continue }
            let fitting = host.fittingSize
            if host.frame.size != fitting {
                host.frame.size = fitting
            }
        }
    }

    private func rebuildStatsSubmenu() {
        statsSubmenu.removeAllItems()

        let isRunning: Bool
        if case .running = server?.state { isRunning = true } else { isRunning = false }

        if !isRunning {
            statsSubmenu.addItem(disabled(String(localized: "menubar.stats.server_off",
                                                 defaultValue: "Server is off",
                                                 comment: "Disabled placeholder in the Serving Stats submenu when the server isn't running")))
            return
        }
        let session = statsPoller?.sessionStats
        let liveActivity = statsPoller?.liveStats?.liveActivity
        let alltime = statsPoller?.alltimeStats
        if session == nil && alltime == nil {
            statsSubmenu.addItem(disabled(statsPoller == nil
                                          ? String(localized: "menubar.stats.no_api_key",
                                                   defaultValue: "Set OMLX_API_KEY to enable stats",
                                                   comment: "Disabled placeholder in the Serving Stats submenu when no API key is configured")
                                          : String(localized: "menubar.stats.loading",
                                                   defaultValue: "Loading stats…",
                                                   comment: "Disabled placeholder shown while stats are loading")))
            return
        }

        if let liveActivity {
            statsSubmenu.addItem(disabled(String(
                localized: "menubar.stats.live_activity",
                defaultValue: "Live Activity",
                comment: "Section header inside the Serving Stats submenu for the current active request"
            )))
            statsSubmenu.addItem(disabled(liveActivity.menuBarTitle))
            statsSubmenu.addItem(disabled(liveActivity.detail))
            statsSubmenu.addItem(.separator())
        }

        statsSubmenu.addItem(disabled(String(localized: "menubar.stats.session_section",
                                             defaultValue: "Session",
                                             comment: "Section header inside the Serving Stats submenu for current-session metrics")))
        appendStat(String(localized: "menubar.stats.total_tokens",
                          defaultValue: "Total Tokens Processed",
                          comment: "Stats row label for total tokens processed"),
                   compact(session?.totalPromptTokens))
        appendStat(String(localized: "menubar.stats.cached_tokens",
                          defaultValue: "Cached Tokens",
                          comment: "Stats row label for cached tokens count"),
                   compact(session?.totalCachedTokens))
        appendStat(String(localized: "menubar.stats.cache_efficiency",
                          defaultValue: "Cache Efficiency",
                          comment: "Stats row label for the cache efficiency percentage"),
                   percent(session?.cacheEfficiency))
        appendStat(String(localized: "menubar.stats.avg_pp_speed",
                          defaultValue: "Avg PP Speed",
                          comment: "Stats row label for the average prompt-processing (prefill) speed"),
                   tps(session?.avgPrefillTps))
        appendStat(String(localized: "menubar.stats.avg_tg_speed",
                          defaultValue: "Avg TG Speed",
                          comment: "Stats row label for the average token-generation speed"),
                   tps(session?.avgGenerationTps))

        statsSubmenu.addItem(.separator())

        statsSubmenu.addItem(disabled(String(localized: "menubar.stats.alltime_section",
                                             defaultValue: "All-Time",
                                             comment: "Section header inside the Serving Stats submenu for all-time metrics")))
        appendStat(String(localized: "menubar.stats.total_tokens",
                          defaultValue: "Total Tokens Processed",
                          comment: "Stats row label for total tokens processed"),
                   compact(alltime?.totalPromptTokens))
        appendStat(String(localized: "menubar.stats.cached_tokens",
                          defaultValue: "Cached Tokens",
                          comment: "Stats row label for cached tokens count"),
                   compact(alltime?.totalCachedTokens))
        appendStat(String(localized: "menubar.stats.cache_efficiency",
                          defaultValue: "Cache Efficiency",
                          comment: "Stats row label for the cache efficiency percentage"),
                   percent(alltime?.cacheEfficiency))
        appendStat(String(localized: "menubar.stats.total_requests",
                          defaultValue: "Total Requests",
                          comment: "Stats row label for total request count"),
                   compact(alltime?.totalRequests))
    }

    // MARK: - Pollers

    /// Bind endpoint the stats poller should hit. Sourced from the live
    /// `ServerProcess` (which `reconfigure(bindAddress:port:)` keeps current) so a
    /// runtime port/host change re-points the poller, falling back to the
    /// config snapshot only when there is no server. Mirrors the
    /// `displayPort`/`displayHost` resolution used for the visible items.
    private func liveBaseURL() -> URL? {
        let host = MenubarController.displayHost(server: server, fallback: config.host)
        let port = MenubarController.displayPort(server: server, fallback: config.port)
        return AppConfig.httpURL(host: host, port: port)
    }

    private func startStatsPoller() {
        guard let baseURL = liveBaseURL() else { return }
        // Tear down any existing poller (and its observer) first so a
        // re-point doesn't leave a second instance polling the old endpoint.
        if let existing = statsPoller {
            existing.stop()
            NotificationCenter.default.removeObserver(
                self,
                name: MenubarStatsPoller.didUpdateNotification,
                object: existing
            )
        }
        let p = MenubarStatsPoller(baseURL: baseURL, apiKey: config.apiKey)
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(statsDidUpdate(_:)),
            name: MenubarStatsPoller.didUpdateNotification,
            object: p
        )
        p.setEnabledMetrics(MenubarMetricPrefs.enabledMetrics)
        p.start()
        self.statsPoller = p
        self.statsPollerBaseURL = baseURL
    }

    /// A port/host change via Server screen's Apply restarts the server on a
    /// new bind, but the stats poller was created once at init with the old
    /// baseURL and would keep polling the dead endpoint (stats freeze after
    /// a port change). Re-point it when the live endpoint diverges from what
    /// the poller currently targets.
    private func refreshStatsPollerEndpoint() {
        guard let want = liveBaseURL() else { return }
        if statsPollerBaseURL == want { return }
        startStatsPoller()
    }

    private func startVisibilityWatcher() {
        let watcher = MenubarVisibilityWatcher(initial: statusItem) { [weak self] in
            self?.recreateStatusItem() ?? NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        }
        watcher.scheduleInitialCheck(after: 3.0)
        self.visibilityWatcher = watcher
    }

    private func recreateStatusItem() -> NSStatusItem {
        NSStatusBar.system.removeStatusItem(statusItem)
        let newStatusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        newStatusItem.behavior = []
        newStatusItem.menu = menu
        newStatusItem.button?.setAccessibilityLabel("oMLX")
        newStatusItem.button?.toolTip = "oMLX"
        statusItem = newStatusItem
        refreshMenuState()
        return newStatusItem
    }

    // MARK: - Notification handlers

    @objc private func serverStateChanged(_ note: Notification) {
        refreshMenuState()
        rebuildStatsSubmenu()
        rebuildModelsSubmenu()
        refreshStatsPollerEndpoint()

        guard let server else { return }

        // Drop cached model state when the server is no longer serving so a
        // stale list can't linger (or flash on the next start) and any in-flight
        // fetch is cancelled rather than landing against a dead endpoint.
        switch server.state {
        case .stopped, .failed:
            modelsFetchTask?.cancel()
            modelsFetchTask = nil
            models = []
            unloadingIDs.removeAll()
            loadingIDs.removeAll()
            modelsFetched = false
            metricsStore.markServerStopped()
        default:
            break
        }
        metricItemsController.sync()

        if case .failed(let message) = server.state,
           MenubarController.shouldShowGenericFailureAlert(message: message) {
            presentServerFailureAlert(message: message, logURL: server.serverLogURL)
        }
    }

    @objc private func serverPortConflict(_ note: Notification) {
        guard let conflict = note.userInfo?["conflict"] as? PortConflict else { return }
        let key = "\(conflict.pid.map { String($0) } ?? "unknown"):\(conflict.isOMLX)"
        guard lastPresentedPortConflictKey != key else { return }
        lastPresentedPortConflictKey = key
        presentPortConflictAlert(conflict)
    }

    @objc private func statsDidUpdate(_ note: Notification) {
        if let poller = note.object as? MenubarStatsPoller,
           let lastStatusSuccessAt = poller.lastStatusSuccessAt {
            server?.recordAuxiliaryHealthSuccess(at: lastStatusSuccessAt)
        }
        // A failure tick (server going away) posts once with stale snapshots
        // still cached — surface it as unknown ("–") rather than freezing
        // the last good numbers on the glyphs.
        let tickOK = statsPoller?.lastTickWasSuccess ?? false
        metricsStore.applyTick(
            live: tickOK
                ? MenubarMetricsStore.liveRates(from: statsPoller?.liveStats)
                : nil,
            average: tickOK
                ? MenubarMetricsStore.averageRates(from: statsPoller?.sessionStats)
                : nil,
            alltime: tickOK
                ? MenubarMetricsStore.averageRates(from: statsPoller?.alltimeStats)
                : nil,
            serverRunning: serverIsRunning && tickOK
        )
        metricItemsController.sync()
        // Stats only need to redraw if the submenu is open or about to open;
        // menuWillOpen (NSMenuDelegate) handles the latter, so for now we
        // rebuild eagerly — the next render will pick up fresh values.
        refreshMenuState()
        rebuildStatsSubmenu()
    }

    @objc private func updateStateChanged(_ note: Notification) {
        refreshUpdateMenuItem()
    }

    /// UserDefaults writes can come from any thread; hop to the main actor
    /// before touching the poller or the status items.
    @objc nonisolated private func defaultsDidChange(_ note: Notification) {
        Task { @MainActor in
            self.applyMetricPreferences()
        }
    }

    /// Pushes the current Appearance toggles into the poller and reconciles
    /// every optional status item plus the system sampling timer. Cheap and
    /// idempotent: the poller guards on equality, sync() only re-rasterizes
    /// on signature changes, and the reconcile is a nil-check when nothing
    /// flipped — so the chattiness of the defaults notification doesn't
    /// matter.
    private func applyMetricPreferences() {
        statsPoller?.setEnabledMetrics(MenubarMetricPrefs.enabledMetrics)
        metricItemsController.sync()
        systemItemsController.sync()
        reconcileSystemSampling()
    }

    // MARK: - Actions

    @objc private func startServer() {
        guard let server else { return }
        do {
            switch try server.start() {
            case .started, .alreadyRunning:
                break
            case .portConflict:
                break
            }
        } catch {
            NSLog("oMLX: start failed — \(error)")
        }
    }

    @objc private func stopServer() {
        guard let server else { return }
        Task { @MainActor in
            await server.stop()
        }
    }

    @objc private func forceRestartServer() {
        guard let server else { return }
        Task { @MainActor in
            do {
                _ = try await server.forceRestart()
            } catch {
                NSLog("oMLX: force-restart failed — \(error)")
            }
        }
    }

    private func presentPortConflictAlert(_ conflict: PortConflict) {
        NSApp.activate(ignoringOtherApps: true)
        let alert = NSAlert()
        let port = MenubarController.displayPort(server: server, fallback: config.port)
        alert.messageText = String(localized: "menubar.alert.port_in_use.title",
                                   defaultValue: "Port \(String(port)) is in use.",
                                   comment: "Title of the port-conflict alert; placeholder is the port number (plain integer, no grouping)")
        let pidStr = conflict.pid.map {
            String(localized: "menubar.alert.pid_known",
                   defaultValue: "PID \(String($0))",
                   comment: "Substring describing a known PID; placeholder is the PID number (plain integer, no grouping)")
        } ?? String(localized: "menubar.alert.pid_unknown",
                    defaultValue: "unknown PID",
                    comment: "Substring used when the conflicting process PID couldn't be determined")
        alert.informativeText = conflict.isOMLX
            ? String(localized: "menubar.alert.port_in_use.omlx",
                     defaultValue: "Another oMLX server is already running on this port (\(pidStr)). Stop it before starting a new instance, or change the port in Settings.",
                     comment: "Port-conflict alert body when the conflicting process is another oMLX instance")
            : String(localized: "menubar.alert.port_in_use.other",
                     defaultValue: "Another process (\(pidStr)) is listening on port \(String(port)). Choose a different port in Settings or terminate that process.",
                     comment: "Port-conflict alert body when an unrelated process owns the port")
        alert.addButton(withTitle: String(localized: "menubar.alert.ok",
                                          defaultValue: "OK",
                                          comment: "Default dismiss button on the port-conflict alert"))
        alert.window.level = .floating
        alert.runModal()
    }

    private func presentServerFailureAlert(message: String, logURL: URL) {
        guard !MenubarController.isRunningUnitTests else { return }
        guard lastPresentedFailureMessage != message else { return }
        lastPresentedFailureMessage = message

        NSApp.activate(ignoringOtherApps: true)
        let logTail = MenubarController.recentLogTail(from: logURL)
        let accessHint = MenubarController.accessFailureHint(
            message: message,
            logTail: logTail
        )

        var parts = [
            String(localized: "menubar.alert.server_failed.body",
                   defaultValue: "The server process exited before it became healthy.",
                   comment: "Introductory body text for the server startup failure alert"),
            message,
        ]
        if let accessHint {
            parts.append(accessHint)
        }
        parts.append(
            String(localized: "menubar.alert.server_failed.log_path",
                   defaultValue: "Log: \(logURL.path)",
                   comment: "Server startup failure alert line that shows the server log path")
        )

        let alert = NSAlert()
        alert.messageText = String(localized: "menubar.alert.server_failed.title",
                                   defaultValue: "oMLX Server Failed to Start",
                                   comment: "Title for the server startup failure alert")
        alert.informativeText = parts.joined(separator: "\n\n")
        alert.addButton(withTitle: String(localized: "menubar.alert.open_log",
                                          defaultValue: "Open Log",
                                          comment: "Button that opens the server log file"))
        alert.addButton(withTitle: String(localized: "menubar.alert.open_settings",
                                          defaultValue: "Open Settings",
                                          comment: "Button that opens the oMLX settings window"))
        alert.addButton(withTitle: String(localized: "menubar.alert.dismiss",
                                          defaultValue: "Dismiss",
                                          comment: "Button that dismisses an alert"))
        alert.window.level = .floating

        switch alert.runModal() {
        case .alertFirstButtonReturn:
            MenubarController.openLogFile(logURL)
        case .alertSecondButtonReturn:
            openAppView()
        default:
            break
        }
    }

    @objc private func openAdminPanel() {
        // AppDelegate owns the SwiftUI Window scene; we just ask it to
        // present. This avoids the Settings-scene + .accessory bug where
        // `showSettingsWindow:` silently no-ops when no window is up.
        openAppView()
    }

    @objc private func openWebAdmin() {
        guard serverIsRunning else { return }
        let host = MenubarController.displayHost(server: server, fallback: config.host)
        let port = MenubarController.displayPort(server: server, fallback: config.port)
        guard let url = MenubarController.webAdminURL(host: host, port: port, apiKey: config.apiKey) else { return }
        NSWorkspace.shared.open(url)
    }

    @objc private func openChat() {
        guard serverIsRunning else { return }
        let host = MenubarController.displayHost(server: server, fallback: config.host)
        let port = MenubarController.displayPort(server: server, fallback: config.port)
        guard let url = AppConfig.httpURL(host: host, port: port, path: "/admin/chat") else { return }
        NSWorkspace.shared.open(url)
    }

    @objc private func installUpdate() {
        updates?.requestUpdateConfirmation()
    }

    @objc private func showAbout() {
        NSApp.activate(ignoringOtherApps: true)
        NSApp.orderFrontStandardAboutPanel(nil)
    }

    @objc private func quitApp() {
        // Real quit (menubar item) — calls AppDelegate.requestQuit which
        // sets the explicit-quit flag and then terminates. Cmd-Q / Dock →
        // Quit go through `applicationShouldTerminate` and are intercepted
        // to close the window only.
        requestQuit()
    }

    // MARK: - Models submenu

    private func rebuildModelsSubmenu() {
        modelsSubmenu.removeAllItems()

        guard serverIsRunning else { return }

        if models.isEmpty {
            let placeholder = modelsFetched
                ? String(localized: "menubar.models.empty",
                         defaultValue: "No models available",
                         comment: "Disabled placeholder in the Models submenu when no models are discovered")
                : String(localized: "menubar.models.fetching",
                         defaultValue: "Loading…",
                         comment: "Disabled placeholder in the Models submenu while the model list is being fetched")
            modelsSubmenu.addItem(disabled(placeholder))
            return
        }

        // Loaded models first, then unloaded favorites, then the rest of the
        // library — deduplicated top-down, empty sections omitted. The
        // Appearance scope pref can hide the library tail; loaded models
        // always show since they are active either way.
        let (loaded, favorites, library) = MenubarController.partitionForMenu(models)
        let showLibrary = MenubarMetricPrefs.modelLibraryScope == .all
        var sections: [(header: String, group: [ModelDTO])] = [
            (String(localized: "menubar.models.section.loaded",
                    defaultValue: "Loaded",
                    comment: "Models submenu section header for loaded or loading models"),
             loaded),
            (String(localized: "menubar.models.section.favorites",
                    defaultValue: "Favorites",
                    comment: "Models submenu section header for favorite models that are not loaded"),
             favorites),
        ]
        if showLibrary {
            sections.append(
                (String(localized: "menubar.models.section.library",
                        defaultValue: "Library",
                        comment: "Models submenu section header for the remaining model library"),
                 library)
            )
        }

        guard sections.contains(where: { !$0.group.isEmpty }) else {
            modelsSubmenu.addItem(disabled(String(
                localized: "menubar.models.no_favorites",
                defaultValue: "No favorite models yet",
                comment: "Disabled placeholder in the Models submenu when the favorites-only scope hides every model"
            )))
            return
        }

        var needsSeparator = false
        for section in sections where !section.group.isEmpty {
            if needsSeparator { modelsSubmenu.addItem(.separator()) }
            needsSeparator = true
            modelsSubmenu.addItem(.sectionHeader(title: section.header))
            for m in section.group { modelsSubmenu.addItem(modelItem(for: m)) }
        }
    }

    private func modelItem(for m: ModelDTO) -> NSMenuItem {
        let title = MenubarController.modelMenuTitle(for: m)
        let parentItem: NSMenuItem
        if m.loaded, !m.sizeLabel.isEmpty {
            let attrTitle = NSMutableAttributedString()
            let titleFont = NSFont.menuFont(ofSize: 0)
            let sizeFont = NSFont.menuFont(ofSize: NSFont.smallSystemFontSize)
            attrTitle.append(NSAttributedString(
                string: title,
                attributes: [.font: titleFont]
            ))
            let paraStyle = NSMutableParagraphStyle()
            paraStyle.lineSpacing = 1
            attrTitle.append(NSAttributedString(
                string: "\n\(m.sizeLabel)",
                attributes: [
                    .font: sizeFont,
                    .foregroundColor: NSColor.secondaryLabelColor,
                    .paragraphStyle: paraStyle,
                ]
            ))
            parentItem = NSMenuItem()
            parentItem.attributedTitle = attrTitle
        } else {
            parentItem = NSMenuItem(title: title, action: nil, keyEquivalent: "")
        }
        // Native checkmark marks loaded models; the raw id lives in the
        // tooltip since the title shows the (possibly aliased) display name.
        parentItem.state = m.loaded ? .on : .off
        parentItem.toolTip = m.id
        parentItem.isEnabled = true

        let sub = NSMenu()
        sub.autoenablesItems = false

        let toggleItem: NSMenuItem
        switch MenubarController.toggleState(for: m, unloading: unloadingIDs, loading: loadingIDs) {
        case .unloading:
            toggleItem = item(String(localized: "menubar.models.unloading",
                                     defaultValue: "Unloading model…",
                                     comment: "Disabled menu item while a model unload is in progress"),
                              action: nil,
                              symbol: "hourglass")
            toggleItem.isEnabled = false
        case .loading:
            toggleItem = item(String(localized: "menubar.models.loading",
                                     defaultValue: "Loading model…",
                                     comment: "Disabled menu item while a model load is in progress"),
                              action: nil,
                              symbol: "hourglass")
            toggleItem.isEnabled = false
        case .unload:
            toggleItem = item(String(localized: "menubar.models.unload",
                                     defaultValue: "Unload model",
                                     comment: "Menu item to unload a currently loaded model"),
                              action: #selector(unloadModelAction(_:)),
                              symbol: "eject")
            toggleItem.isEnabled = true
        case .load:
            toggleItem = item(String(localized: "menubar.models.load",
                                     defaultValue: "Load model",
                                     comment: "Menu item to load a model"),
                              action: #selector(loadModelAction(_:)),
                              symbol: "play.circle")
            toggleItem.isEnabled = true
        }
        toggleItem.representedObject = m.id
        sub.addItem(toggleItem)

        let settingsItem = item(String(localized: "menubar.models.settings",
                                       defaultValue: "Model Settings…",
                                       comment: "Menu item that opens the in-app per-model settings pane"),
                                action: #selector(openModelSettingsAction(_:)),
                                symbol: "gearshape")
        settingsItem.representedObject = m.id
        settingsItem.isEnabled = true
        sub.addItem(settingsItem)

        let copyItem = item(String(localized: "menubar.models.copy_name",
                                   defaultValue: "Copy name",
                                   comment: "Menu item that copies the model id to the clipboard"),
                            action: #selector(copyModelNameAction(_:)),
                            symbol: "doc.on.doc")
        copyItem.representedObject = m.id
        copyItem.isEnabled = true
        sub.addItem(copyItem)

        parentItem.submenu = sub
        return parentItem
    }

    /// Supersedes any in-flight model-list fetch with a fresh one. Keeping a
    /// single cancellable task means a stale fetch (e.g. one started just before
    /// the server stopped) can't clobber current state — `refreshModels` bails on
    /// `Task.isCancelled`.
    private func scheduleModelsRefresh() {
        modelsFetchTask?.cancel()
        modelsFetchTask = Task { [weak self] in await self?.refreshModels() }
    }

    private func refreshModels() async {
        guard serverIsRunning, let client else { return }
        guard let resp = try? await client.listModels() else { return }
        guard !Task.isCancelled else { return }
        models = MenubarController.visibleMenuModels(resp.models)
        modelsFetched = true
        unloadingIDs = MenubarController.reconcileUnloading(unloadingIDs, against: models)
        loadingIDs = MenubarController.reconcileLoading(loadingIDs, against: models)
        rebuildModelsSubmenu()
    }

    // MARK: - Per-model actions

    @objc private func loadModelAction(_ sender: NSMenuItem) {
        guard let id = sender.representedObject as? String else { return }
        loadingIDs.insert(id)
        rebuildModelsSubmenu()
        // The load POST runs in its own task so it isn't cancelled when the menu
        // closes; only the trailing list refresh routes through the scheduler.
        Task {
            // A failed load must drop the pending id so the entry falls back
            // to "Load model" instead of sticking on the in-progress state.
            do { _ = try await client?.loadModel(id: id) }
            catch { loadingIDs.remove(id) }
            scheduleModelsRefresh()
        }
    }

    @objc private func unloadModelAction(_ sender: NSMenuItem) {
        guard let id = sender.representedObject as? String else { return }
        unloadingIDs.insert(id)
        rebuildModelsSubmenu()
        Task {
            // A failed unload (500, timeout) must drop the pending id, or
            // reconcileUnloading keeps the entry wedged on "Unloading model…"
            // for as long as the server still reports the model loaded.
            do { _ = try await client?.unloadModel(id: id) }
            catch { unloadingIDs.remove(id) }
            scheduleModelsRefresh()
        }
    }

    @objc private func openModelSettingsAction(_ sender: NSMenuItem) {
        guard let id = sender.representedObject as? String else { return }
        openModelSettings(id)
    }

    @objc private func copyModelNameAction(_ sender: NSMenuItem) {
        guard let id = sender.representedObject as? String else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(id, forType: .string)
    }

    // MARK: - Models — pure helpers

    /// The load/unload control state for a model, given the set of ids whose
    /// unload is currently in flight. Extracted so it can be unit-tested without
    /// a live `NSStatusBar` (which instantiating the controller requires).
    enum ModelToggleState: Equatable { case unloading, loading, unload, load }

    nonisolated static func toggleState(
        for model: ModelDTO,
        unloading: Set<String>,
        loading: Set<String>
    ) -> ModelToggleState {
        if unloading.contains(model.id) { return .unloading }
        if model.isLoading || loading.contains(model.id) { return .loading }
        if model.loaded { return .unload }
        return .load
    }

    /// Keeps only the ids whose model still reports `loaded == true` — i.e. drops
    /// any pending-unload id once the server confirms it's gone (or unknown).
    nonisolated static func reconcileUnloading(_ unloading: Set<String>, against models: [ModelDTO]) -> Set<String> {
        unloading.filter { id in models.first(where: { $0.id == id })?.loaded == true }
    }

    /// Keeps only the pending-load ids the server hasn't picked up yet — i.e.
    /// drops an id once its model reports `isLoading` or `loaded` (or vanishes).
    nonisolated static func reconcileLoading(_ loading: Set<String>, against models: [ModelDTO]) -> Set<String> {
        loading.filter { id in
            guard let m = models.first(where: { $0.id == id }) else { return false }
            return !m.loaded && !m.isLoading
        }
    }

    /// Menu-visible models: virtual builtin entries (e.g. the MarkItDown
    /// document converter) have no real load/unload lifecycle and are dropped.
    nonisolated static func visibleMenuModels(_ models: [ModelDTO]) -> [ModelDTO] {
        models.filter { $0.virtual != true }
    }

    /// Splits models into the three menu sections — loaded/loading models,
    /// unloaded favorites, and the rest of the library — deduplicated top-down
    /// and each alphabetical by display title.
    nonisolated static func partitionForMenu(
        _ models: [ModelDTO]
    ) -> (loaded: [ModelDTO], favorites: [ModelDTO], library: [ModelDTO]) {
        func alphabetical(_ group: [ModelDTO]) -> [ModelDTO] {
            group.sorted {
                $0.displayTitle.localizedCaseInsensitiveCompare($1.displayTitle) == .orderedAscending
            }
        }
        let active = models.filter { $0.loaded || $0.isLoading }
        let rest = models.filter { !$0.loaded && !$0.isLoading }
        return (
            loaded: alphabetical(active),
            favorites: alphabetical(rest.filter { $0.isFavorite ?? false }),
            library: alphabetical(rest.filter { !($0.isFavorite ?? false) })
        )
    }

    /// Menu label for a model: the same display title the Models screen shows.
    nonisolated static func modelMenuTitle(for model: ModelDTO) -> String {
        model.displayTitle
    }

    // MARK: - Helpers

    private func disabled(_ title: String) -> NSMenuItem {
        let it = NSMenuItem(title: title, action: nil, keyEquivalent: "")
        it.isEnabled = false
        return it
    }

    private var serverIsRunning: Bool {
        if case .running = server?.state { return true }
        return false
    }

    private func appendStat(_ label: String, _ value: String) {
        let it = NSMenuItem(title: "\(label):  \(value)", action: nil, keyEquivalent: "")
        it.isEnabled = false
        statsSubmenu.addItem(it)
    }

    private func compact(_ value: Int?) -> String {
        guard let n = value else { return "—" }
        if n >= 1_000_000_000 { return String(format: "%.1fB", Double(n) / 1e9) }
        if n >= 1_000_000     { return String(format: "%.1fM", Double(n) / 1e6) }
        if n >= 1_000         { return String(format: "%.1fK", Double(n) / 1e3) }
        return "\(n)"
    }

    private func percent(_ value: Double?) -> String {
        guard let v = value else { return "—" }
        return String(format: "%.1f%%", v)
    }

    private func tps(_ value: Double?) -> String {
        guard let v = value else { return "—" }
        return String(format: "%.1f tok/s", v)
    }
}

// MARK: - Live endpoint resolution

extension MenubarController {
    struct MenuAvailability: Equatable {
        let settings: Bool
        let webDashboard: Bool
        let chat: Bool
    }

    static func menuAvailability(for state: ServerProcess.State) -> MenuAvailability {
        let browserItemsEnabled: Bool
        if case .running = state {
            browserItemsEnabled = true
        } else {
            browserItemsEnabled = false
        }
        return MenuAvailability(
            settings: true,
            webDashboard: browserItemsEnabled,
            chat: browserItemsEnabled
        )
    }

    /// Source-of-truth port for any menubar item that renders the
    /// current bind port (status header, port-conflict alert, Chat URL).
    /// The running server is authoritative — `config` here is just the
    /// snapshot we were constructed with and goes stale after the user
    /// changes the port via Server screen's Apply, which calls
    /// `server.reconfigure(port:)` and updates `AppServices.config` but
    /// not the menubar's local `config` copy.
    ///
    /// Internal access (not private) so `MenubarControllerPortTests`
    /// can exercise it without instantiating the full controller (which
    /// requires a live `NSStatusBar`).
    static func displayPort(server: ServerProcess?, fallback: Int) -> Int {
        server?.port ?? fallback
    }

    /// Companion to `displayPort(server:fallback:)` — same rationale.
    static func displayHost(server: ServerProcess?, fallback: String) -> String {
        server?.host ?? fallback
    }

    /// Builds the browser URL for the web admin dashboard. Uses the
    /// `/admin/auto-login` endpoint so the dashboard opens without the
    /// manual login form: the server validates the main API key, sets the
    /// session cookie, then redirects to `redirect`. A missing/stale key
    /// makes the endpoint redirect to the login page instead — a graceful
    /// fallback, so we still emit the URL.
    ///
    /// `URLComponents.queryItems` percent-encodes the key, so a key
    /// containing `&`, `=`, `/`, spaces etc. is transmitted intact. The one
    /// exception is `+`: URLComponents leaves it unescaped and servers
    /// decode `+` as a space (form-urlencoded semantics), which would
    /// corrupt a key containing `+`. We escape it explicitly below.
    ///
    /// Internal (not private) so `MenubarControllerPortTests` can exercise
    /// it without a live `NSStatusBar`.
    static func webAdminURL(host: String, port: Int, apiKey: String?) -> URL? {
        guard let baseURL = AppConfig.httpURL(
            host: AppConfig.connectableHost(for: host),
            port: port
        ),
              var comps = URLComponents(url: baseURL, resolvingAgainstBaseURL: false)
        else {
            return nil
        }
        comps.path = "/admin/auto-login"
        var items = [URLQueryItem(name: "redirect", value: "/admin/dashboard")]
        if let key = apiKey, !key.isEmpty {
            items.append(URLQueryItem(name: "key", value: key))
        }
        comps.queryItems = items
        comps.percentEncodedQuery = comps.percentEncodedQuery?
            .replacingOccurrences(of: "+", with: "%2B")
        return comps.url
    }

    static func shouldShowGenericFailureAlert(message: String) -> Bool {
        let lower = message.lowercased()
        return !(lower.hasPrefix("port ") && lower.contains(" in use"))
    }

    static var isRunningUnitTests: Bool {
        ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil
    }

    static func accessFailureHint(message: String, logTail: String?) -> String? {
        let haystack = ([message, logTail].compactMap { $0 }).joined(separator: "\n")
            .lowercased()
        guard haystack.contains("permissionerror")
            || haystack.contains("operation not permitted")
            || haystack.contains("permission denied")
        else {
            return nil
        }
        return String(localized: "menubar.alert.server_failed.access_hint",
                      defaultValue: "Check that the configured model directory is mounted and readable. If it is on an external or protected location, grant oMLX access in macOS Privacy & Security settings.",
                      comment: "Hint shown when the server failure log suggests a model directory permission problem")
    }

    static func recentLogTail(from url: URL, maxBytes: UInt64 = 8 * 1024) -> String? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }

        let end = (try? handle.seekToEnd()) ?? 0
        let offset = end > maxBytes ? end - maxBytes : 0
        do {
            try handle.seek(toOffset: offset)
            let data = try handle.readToEnd() ?? Data()
            return String(data: data, encoding: .utf8)
        } catch {
            return nil
        }
    }

    static func openLogFile(_ logURL: URL) {
        if FileManager.default.fileExists(atPath: logURL.path) {
            NSWorkspace.shared.activateFileViewerSelecting([logURL])
            return
        }
        NSWorkspace.shared.open(logURL.deletingLastPathComponent())
    }
}

// MARK: - NSMenuDelegate

extension MenubarController: NSMenuDelegate {
    func menuWillOpen(_ menu: NSMenu) {
        if menu === systemStatsSubmenu {
            systemStatsSubmenuOpen = true
            reconcileSystemSampling()
            // The reconcile may have found the timer already running (bar
            // items enabled) — paint the panels immediately in that case.
            systemSamplingTick()
            return
        }
        // One menubar dropdown at a time — the metric popovers don't dismiss
        // on a click that lands on our own main status item.
        metricItemsController.closeAllPopovers()
        systemItemsController.closeAllPopovers()
        refreshMenuState()
        rebuildStatsSubmenu()
        rebuildModelsSubmenu()
        scheduleModelsRefresh()
    }

    func menuDidClose(_ menu: NSMenu) {
        // The submenu posts its own close; the main-menu close is the
        // belt-and-suspenders for teardown paths that skip it.
        if menu === systemStatsSubmenu || menu === self.menu {
            systemStatsSubmenuOpen = false
            reconcileSystemSampling()
        }
    }
}

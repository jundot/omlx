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
// poll runs at 1Hz against /admin/api/stats; visibility watcher probes once
// at +3 s post-launch with a single recreate-and-retry before alerting.

import AppKit

@MainActor
final class MenubarController: NSObject {

    // MARK: - Inputs / state

    private let server: ServerProcess?
    private let config: AppConfig
    private let updates: UpdateController?
    private let bootstrapError: Error?
    private let openAppView: () -> Void
    private let requestQuit: () -> Void

    private var statusItem: NSStatusItem
    private let menu = NSMenu()

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
    private var statsParentItem: NSMenuItem!
    private var statsSubmenu: NSMenu!
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
        openAppView: @escaping () -> Void = {},
        requestQuit: @escaping () -> Void = { NSApp.terminate(nil) }
    ) {
        self.server = server
        self.config = config
        self.updates = updates
        self.bootstrapError = lastError
        self.openAppView = openAppView
        self.requestQuit = requestQuit

        self.statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)

        // Cap icons at 18×18 pt (the standard macOS menubar icon size).
        // Our SVGs are 497×497 natural; without this, the status item
        // auto-sizes to that natural width and dominates the menubar.
        // Mirrors Python's _load_menubar_icon (app.py:973).
        let menubarIconSize = NSSize(width: 18, height: 18)

        let outline = NSImage(named: "MenubarOutline")
        outline?.size = menubarIconSize
        outline?.isTemplate = true
        self.iconOutline = outline

        let filled = NSImage(named: "MenubarFilled")
        filled?.size = menubarIconSize
        filled?.isTemplate = true
        self.iconFilled = filled

        super.init()

        statusItem.button?.image = outline
        // SF Symbol fallback for asset-catalog miss in Debug builds.
        if statusItem.button?.image == nil {
            let fallback = NSImage(
                systemSymbolName: "cube.transparent",
                accessibilityDescription: "oMLX"
            )
            fallback?.isTemplate = true
            statusItem.button?.image = fallback
        }
        statusItem.behavior = []
        statusItem.menu = menu
        // This menu is state-driven by refreshMenuState(). If AppKit's
        // automatic target/action enabling stays on, stopped-server items
        // such as Web Dashboard and Chat can be re-enabled while opening.
        menu.autoenablesItems = false
        menu.delegate = self

        buildMenu()
        refreshMenuState()

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

        menu.addItem(.separator())

        statsParentItem = item(String(localized: "menubar.item.serving_stats",
                                      defaultValue: "Serving Stats",
                                      comment: "Menubar parent item opening the Serving Stats submenu"),
                               action: nil,
                               symbol: "chart.bar")
        statsSubmenu = NSMenu()
        statsParentItem.submenu = statsSubmenu
        menu.addItem(statsParentItem)
        rebuildStatsSubmenu()

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

        // Native Settings is the recovery surface for stopped/failed servers.
        // Web Dashboard / Chat open browser URLs against the live port, so
        // they stay gated on a healthy running server.
        let availability = MenubarController.menuAvailability(for: state)
        adminPanelItem.isEnabled = availability.settings
        webAdminItem.isEnabled = availability.webDashboard
        chatItem.isEnabled = availability.chat

        refreshUpdateMenuItem()

        // Icon swap — outline when not actively serving, filled otherwise
        let serving = state.isRunningLike
        statusItem.button?.image = serving ? iconFilled : iconOutline
        statusItem.button?.image?.isTemplate = true
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
        return URL(string: "http://\(host):\(port)")
    }

    private func startStatsPoller() {
        guard let baseURL = liveBaseURL(),
              let key = config.apiKey, !key.isEmpty else { return }
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
        let p = MenubarStatsPoller(baseURL: baseURL, apiKey: key)
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(statsDidUpdate(_:)),
            name: MenubarStatsPoller.didUpdateNotification,
            object: p
        )
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
            self?.recreateStatusItem() ?? NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        }
        watcher.scheduleInitialCheck(after: 3.0)
        self.visibilityWatcher = watcher
    }

    private func recreateStatusItem() -> NSStatusItem {
        NSStatusBar.system.removeStatusItem(statusItem)
        let new = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        new.button?.image = iconOutline
        new.button?.image?.isTemplate = true
        new.menu = menu
        statusItem = new
        return new
    }

    // MARK: - Notification handlers

    @objc private func serverStateChanged(_ note: Notification) {
        refreshMenuState()
        rebuildStatsSubmenu()
        refreshStatsPollerEndpoint()

        guard let server else { return }
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
        // Stats only need to redraw if the submenu is open or about to open;
        // menuWillOpen (NSMenuDelegate) handles the latter, so for now we
        // rebuild eagerly — the next render will pick up fresh values.
        rebuildStatsSubmenu()
    }

    @objc private func updateStateChanged(_ note: Notification) {
        refreshUpdateMenuItem()
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
        guard let url = URL(string: "http://\(host):\(port)/admin/chat") else { return }
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
        var comps = URLComponents()
        comps.scheme = "http"
        comps.host = host
        comps.port = port
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
        refreshMenuState()
        rebuildStatsSubmenu()
    }
}

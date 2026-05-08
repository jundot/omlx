// PR 4 — full menubar parity port. Mirrors the Python menu construction
// (app.py:1450-1700) and refresh strategy (menuWillOpen + per-second poll).
//
// Items, top-down:
//   • Status header                     (colored, non-clickable)
//   • Force Restart   (UNRESPONSIVE/ERROR only)
//   • Stop Server     (RUNNING / STARTING / STOPPING / UNRESPONSIVE)
//   • Start Server    (STOPPED / IDLE / FAILED)
//   • Serving Stats   (Session + All-Time submenu)
//   • Admin Panel     (enabled when running — opens AppView in PR 6, browser fallback now)
//   • Chat with oMLX  (enabled when running — opens /admin/chat in browser)
//   • Settings…       (Cmd-, → SwiftUI Settings scene; AppView replaces in PR 6)
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
    private let bootstrapError: Error?

    private var statusItem: NSStatusItem
    private let menu = NSMenu()

    private var statsPoller: MenubarStatsPoller?
    private var visibilityWatcher: MenubarVisibilityWatcher?

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
    private var chatItem: NSMenuItem!

    private let iconOutline: NSImage?
    private let iconFilled: NSImage?

    // MARK: - Init

    init(server: ServerProcess?, config: AppConfig, lastError: Error? = nil) {
        self.server = server
        self.config = config
        self.bootstrapError = lastError

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
        }

        startStatsPoller()
        startVisibilityWatcher()
    }

    // MARK: - Menu construction

    private func buildMenu() {
        menu.removeAllItems()

        statusHeader = NSMenuItem(title: "Server: …", action: nil, keyEquivalent: "")
        statusHeader.isEnabled = false
        menu.addItem(statusHeader)

        menu.addItem(.separator())

        restartItem = item("Force Restart",
                           action: #selector(forceRestartServer),
                           symbol: "arrow.clockwise.circle")
        menu.addItem(restartItem)

        stopItem = item("Stop Server",
                        action: #selector(stopServer),
                        symbol: "stop.circle")
        menu.addItem(stopItem)

        startItem = item("Start Server",
                         action: #selector(startServer),
                         symbol: "play.circle")
        menu.addItem(startItem)

        menu.addItem(.separator())

        statsParentItem = item("Serving Stats", action: nil, symbol: "chart.bar")
        statsSubmenu = NSMenu()
        statsParentItem.submenu = statsSubmenu
        menu.addItem(statsParentItem)
        rebuildStatsSubmenu()

        menu.addItem(.separator())

        adminPanelItem = item("Admin Panel",
                              action: #selector(openAdminPanel),
                              symbol: "globe")
        menu.addItem(adminPanelItem)

        chatItem = item("Chat with oMLX",
                        action: #selector(openChat),
                        symbol: "message")
        menu.addItem(chatItem)

        menu.addItem(.separator())

        let prefs = item("Settings…",
                         action: #selector(openSettings),
                         symbol: "gearshape",
                         keyEquivalent: ",")
        menu.addItem(prefs)

        let about = item("About oMLX-next",
                         action: #selector(showAbout),
                         symbol: "info.circle")
        menu.addItem(about)

        menu.addItem(.separator())

        let quit = item("Quit oMLX-next",
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
        let isRunning: Bool
        if case .running = state { isRunning = true } else { isRunning = false }
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

        // Admin Panel + Chat enabled when actually running (not unresponsive)
        adminPanelItem.isEnabled = isRunning
        chatItem.isEnabled = isRunning

        // Icon swap — outline when not actively serving, filled otherwise
        let serving = isRunning || isUnresponsive
        statusItem.button?.image = serving ? iconFilled : iconOutline
        statusItem.button?.image?.isTemplate = true
    }

    private func headerDisplay(_ state: ServerProcess.State) -> (String, NSColor) {
        switch state {
        case .stopped:
            if let err = bootstrapError {
                return ("Server: bootstrap failed (\(err))", .systemRed)
            }
            return ("Server: stopped", .secondaryLabelColor)
        case .starting:
            return ("Server: starting…", .systemBlue)
        case .running(let pid):
            return ("Server: running · pid \(pid) · :\(config.port)", .systemGreen)
        case .stopping:
            return ("Server: stopping…", .systemOrange)
        case .unresponsive(let pid):
            return ("Server: unresponsive · pid \(pid) (auto-recover or Force Restart)", .systemOrange)
        case .failed(let msg):
            return ("Server: failed — \(msg)", .systemRed)
        }
    }

    private func rebuildStatsSubmenu() {
        statsSubmenu.removeAllItems()

        let isRunning: Bool
        if case .running = server?.state { isRunning = true } else { isRunning = false }

        if !isRunning {
            statsSubmenu.addItem(disabled("Server is off"))
            return
        }
        let session = statsPoller?.sessionStats
        let alltime = statsPoller?.alltimeStats
        if session == nil && alltime == nil {
            statsSubmenu.addItem(disabled(statsPoller == nil
                                          ? "Set OMLX_API_KEY to enable stats"
                                          : "Loading stats…"))
            return
        }

        statsSubmenu.addItem(disabled("Session"))
        appendStat("Total Tokens Processed", compact(session?.totalPromptTokens))
        appendStat("Cached Tokens", compact(session?.totalCachedTokens))
        appendStat("Cache Efficiency", percent(session?.cacheEfficiency))
        appendStat("Avg PP Speed", tps(session?.avgPrefillTps))
        appendStat("Avg TG Speed", tps(session?.avgGenerationTps))

        statsSubmenu.addItem(.separator())

        statsSubmenu.addItem(disabled("All-Time"))
        appendStat("Total Tokens Processed", compact(alltime?.totalPromptTokens))
        appendStat("Cached Tokens", compact(alltime?.totalCachedTokens))
        appendStat("Cache Efficiency", percent(alltime?.cacheEfficiency))
        appendStat("Total Requests", compact(alltime?.totalRequests))
    }

    // MARK: - Pollers

    private func startStatsPoller() {
        guard let baseURL = config.baseURL,
              let key = config.apiKey, !key.isEmpty else { return }
        let p = MenubarStatsPoller(baseURL: baseURL, apiKey: key)
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(statsDidUpdate(_:)),
            name: MenubarStatsPoller.didUpdateNotification,
            object: p
        )
        p.start()
        self.statsPoller = p
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
    }

    @objc private func statsDidUpdate(_ note: Notification) {
        // Stats only need to redraw if the submenu is open or about to open;
        // menuWillOpen (NSMenuDelegate) handles the latter, so for now we
        // rebuild eagerly — the next render will pick up fresh values.
        rebuildStatsSubmenu()
    }

    // MARK: - Actions

    @objc private func startServer() {
        guard let server else { return }
        do {
            switch try server.start() {
            case .started, .alreadyRunning:
                break
            case .portConflict(let conflict):
                presentPortConflictAlert(conflict)
            }
        } catch {
            NSLog("oMLX-next: start failed — \(error)")
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
                NSLog("oMLX-next: force-restart failed — \(error)")
            }
        }
    }

    private func presentPortConflictAlert(_ conflict: PortConflict) {
        NSApp.activate(ignoringOtherApps: true)
        let alert = NSAlert()
        alert.messageText = "Port \(config.port) is in use."
        let pidStr = conflict.pid.map { "PID \($0)" } ?? "unknown PID"
        alert.informativeText = conflict.isOMLX
            ? "Another oMLX server is already running on this port (\(pidStr)). Stop it before starting a new instance, or change the port in Settings."
            : "Another process (\(pidStr)) is listening on port \(config.port). Choose a different port in Settings or terminate that process."
        alert.addButton(withTitle: "OK")
        alert.window.level = .floating
        alert.runModal()
    }

    @objc private func openAdminPanel() {
        // PR 6 wires this to the SwiftUI AppView. Until then, route to the
        // browser admin so the menu item isn't a dead end.
        guard let url = URL(string: "http://\(config.host):\(config.port)/admin/dashboard") else { return }
        NSWorkspace.shared.open(url)
    }

    @objc private func openChat() {
        guard let url = URL(string: "http://\(config.host):\(config.port)/admin/chat") else { return }
        NSWorkspace.shared.open(url)
    }

    @objc private func openSettings() {
        // Opens the SwiftUI Settings scene declared in oMLXApp. macOS 14+
        // exposes showSettingsWindow:; the older showPreferencesWindow:
        // is still routed by AppKit, so we try both.
        if !NSApp.sendAction(Selector(("showSettingsWindow:")), to: nil, from: self) {
            _ = NSApp.sendAction(Selector(("showPreferencesWindow:")), to: nil, from: self)
        }
    }

    @objc private func showAbout() {
        NSApp.activate(ignoringOtherApps: true)
        NSApp.orderFrontStandardAboutPanel(nil)
    }

    @objc private func quitApp() {
        NSApp.terminate(nil)
    }

    // MARK: - Helpers

    private func disabled(_ title: String) -> NSMenuItem {
        let it = NSMenuItem(title: title, action: nil, keyEquivalent: "")
        it.isEnabled = false
        return it
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

// MARK: - NSMenuDelegate

extension MenubarController: NSMenuDelegate {
    func menuWillOpen(_ menu: NSMenu) {
        refreshMenuState()
        rebuildStatsSubmenu()
    }
}

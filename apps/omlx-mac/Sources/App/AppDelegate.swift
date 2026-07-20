// Application delegate: sequences activation policy, menubar, server
// bootstrap, and signal handlers. The main AppView window is a SwiftUI
// `Window` scene declared in oMLXApp.swift — we no longer build it
// manually here.
//
// Boot flow
//   applicationWillFinishLaunching  → setActivationPolicy(.regular)
//                                     (Dock icon shows briefly during launch)
//   applicationDidFinishLaunching   → load AppConfig
//                                     → install NSWindow observers (drive
//                                       the dock-icon toggle)
//                                     → if first run (no settings.json):
//                                         • show Welcome window (wizard
//                                           persists config + spawns server
//                                           only after Start Server)
//                                     else (returning user):
//                                         • resolve PythonRuntime
//                                         • spawn ServerProcess
//                                         • create MenubarController
//                                         • install POSIX SignalHandlers
//                                         • flip to .accessory next tick
//                                           (Dock icon hides; menubar stays)
//   applicationWillTerminate        → await server.stop(timeout: 10)
//
// Dock-icon toggle
//   Any time an in-app NSWindow becomes main → .regular (Dock icon shows).
//   When the last visible app window closes → .accessory (Dock icon hides).
//   Server + menubar are untouched by the toggle.

import AppKit
import SwiftUI

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private(set) var server: ServerProcess?
    private var menubar: MenubarController?
    private var controlServer: AppControlServer?
    let services = AppServices()

    private var welcomeController: WelcomeWindowController?
    private var welcomeCloseObserver: NSObjectProtocol?

    /// Set true by `requestQuit()` to permit a real terminate. Cmd-Q / Dock
    /// Quit / "Quit oMLX" from the application menu all route through
    /// `applicationShouldTerminate`, which (when this flag is false) closes
    /// any visible app window instead of terminating — preserving the
    /// menubar status item + the running server. The menubar's own "Quit"
    /// item flips this flag before triggering termination.
    private var explicitQuitRequested: Bool = false

    /// Set true by `hideWindowsAndDropDockIcon()` so the willCloseNotification
    /// observer knows this close was app-initiated (Cmd-Q / Dock Quit) and
    /// should drop the Dock icon. When false, the close came from the user
    /// clicking the red traffic-light button — leave the Dock icon up so
    /// the user can click it to bring the window back.
    private var dropDockIconOnNextClose: Bool = false

    /// Appearance → "Show Dock Icon". While true, every `.accessory` drop is
    /// suppressed so the Dock icon stays up with no window open.
    private var dockIconAlwaysVisible: Bool {
        MenubarMetricPrefs.showDockIcon
    }

    /// Last value of the pref that was acted on, so the chatty
    /// UserDefaults notification only triggers policy work on real flips.
    private var lastAppliedDockIconPref: Bool?

    func requestQuit() {
        explicitQuitRequested = true
        NSApp.terminate(nil)
    }

    /// Cmd-Q / Dock → Quit path: hide every titled window AND set
    /// `.accessory` so the Dock icon vanishes. Server + menubar stay alive.
    func hideWindowsAndDropDockIcon() {
        dropDockIconOnNextClose = true
        var hidAny = false
        for win in NSApp.windows where win.styleMask.contains(.titled) && win.isVisible {
            win.close()
            if win.isVisible { win.orderOut(nil) }
            hidAny = hidAny || !win.isVisible
        }
        // If close() was vetoed and only orderOut hid the window,
        // willCloseNotification didn't fire — drop policy explicitly.
        let stillVisible = NSApp.windows.contains { $0.styleMask.contains(.titled) && $0.isVisible }
        if !stillVisible, !dockIconAlwaysVisible {
            NSApp.setActivationPolicy(.accessory)
        }
        dropDockIconOnNextClose = false
        _ = hidAny
    }

    /// Bring the main AppView window forward. If SwiftUI hasn't materialised
    /// the NSWindow yet (i.e. nobody opened it since launch), kick the
    /// `omlxapp://main` URL — the Window scene in oMLXApp.swift handles it
    /// via `.handlesExternalEvents(matching: ["main"])`.
    func presentAppView() {
        // Flip to .regular eagerly so the Dock icon shows in lockstep with
        // the window appearing. The `didBecomeMain` observer is a backup
        // for other paths (e.g. Welcome window), but on re-opening a hidden
        // SwiftUI Window the notification doesn't always fire (the existing
        // NSWindow is just ordered front rather than re-created), so we
        // can't rely on it here.
        if NSApp.activationPolicy() != .regular {
            NSApp.setActivationPolicy(.regular)
        }
        NSApp.activate(ignoringOtherApps: true)
        if let main = mainAppViewWindow() {
            // Also apply on every show: the observer fires only on
            // didBecomeMain, which may not run if the window was just
            // reordered without becoming main.
            main.titleVisibility = .hidden
            main.makeKeyAndOrderFront(nil)
            return
        }
        if let url = URL(string: "omlxapp://main") {
            NSWorkspace.shared.open(url)
        }
    }

    /// SwiftUI's `Window(id: "main")` tags its NSWindow with that identifier
    /// (the actual rawValue includes a stable prefix; substring match is
    /// stable across macOS revisions).
    private func mainAppViewWindow() -> NSWindow? {
        NSApp.windows.first { window in
            window.identifier?.rawValue.contains("main") == true
        }
    }

    nonisolated func applicationWillFinishLaunching(_ notification: Notification) {
        // Regular policy until the status item registers; we flip to Accessory
        // after creating the menubar (next runloop tick).
        DispatchQueue.main.async {
            NSApp.setActivationPolicy(.regular)
        }
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        installWindowObservers()
        // Seed without applying: launch flow (accessory flip / welcome)
        // owns the initial policy; only later real flips act.
        lastAppliedDockIconPref = dockIconAlwaysVisible
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(defaultsDidChange(_:)),
            name: UserDefaults.didChangeNotification,
            object: UserDefaults.standard
        )
        services.updates.setTerminateForUpdate { [weak self] in
            if let self {
                self.requestQuit()
            } else {
                NSApp.terminate(nil)
            }
        }
        services.updates.setPresentUpdateConfirmation { [weak self] in
            self?.presentAppView()
        }
        if !isRunningUnitTests {
            do {
                let cliResult = try ShellEnvWriter.ensureCLIShim()
                handleCLISetupResult(cliResult)
            } catch {
                NSLog("oMLX: CLI shim setup failed — \(error)")
            }
            startControlServer()
        }

        let config = AppConfig.load()
        services.updateConfig(config)

        if AppConfig.hasExistingConfig {
            // Returning user. AppConfig.load() picks the highest-priority
            // file (`~/.omlx/settings.json` first, Library config.json
            // second) and stamps `config.source` so future saves route to
            // the same file. No re-write needed here.
            bootstrapServer(config: config)
            scheduleAccessoryPolicyFlip()
        } else {
            // First run: show the wizard only. Do not create the menubar or
            // persist settings until the user clicks Start Server.
            NSApp.activate(ignoringOtherApps: true)
            presentWelcome()
        }
    }

    private func handleCLISetupResult(_ result: ShellEnvWriter.CLISetupResult) {
        guard case .needsShellPathPrompt(let reason) = result else { return }
        guard !ShellEnvWriter.shouldSuppressCLIPathPrompt() else { return }
        promptForShellPathExport(reason: reason)
    }

    private func promptForShellPathExport(reason: String) {
        let alert = NSAlert()
        alert.messageText = "Enable `omlx` in Terminal?"
        alert.informativeText = """
        oMLX could not create a public `omlx` command in /opt/homebrew/bin or /usr/local/bin.

        To make `omlx` available in new Terminal sessions, oMLX can add a small PATH block to your shell init file. This only happens if you choose Update Shell File.

        \(reason)
        """
        alert.addButton(withTitle: "Update Shell File")
        alert.addButton(withTitle: "Dismiss Now")
        alert.addButton(withTitle: "Don't Ask Again")
        alert.window.level = .floating

        switch alert.runModal() {
        case .alertFirstButtonReturn:
            do {
                try ShellEnvWriter.ensureShellPathExport()
            } catch {
                NSLog("oMLX: CLI shell path setup failed — \(error)")
            }
        case .alertThirdButtonReturn:
            ShellEnvWriter.suppressCLIPathPromptForever()
        default:
            break
        }
    }

    private var isRunningUnitTests: Bool {
        ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil
    }

    /// All three MenubarController construction sites (first-run, returning
    /// user success, returning user failure) capture the same `openAppView`
    /// closure and differ only in `server`/`lastError`.
    private func makeMenubar(
        server: ServerProcess?,
        config: AppConfig,
        lastError: Error? = nil
    ) -> MenubarController {
        MenubarController(
            server: server,
            config: config,
            updates: services.updates,
            lastError: lastError,
            client: services.client,
            openModelSettings: { [weak self] id in
                guard let self else { return }
                self.services.modelDetailID = id
                self.services.requestedSection = .models
                self.presentAppView()
            },
            openAppView: { [weak self] in self?.presentAppView() },
            openAppearanceSettings: { [weak self] in
                guard let self else { return }
                self.services.requestedSection = .appearance
                self.presentAppView()
            },
            requestQuit:  { [weak self] in self?.requestQuit() }
        )
    }

    /// UserDefaults writes can come from any thread; hop before touching
    /// the activation policy.
    @objc nonisolated private func defaultsDidChange(_ note: Notification) {
        Task { @MainActor in
            self.applyDockIconPreference()
        }
    }

    /// Applies a "Show Dock Icon" flip immediately: ON shows the icon right
    /// away; OFF returns to menubar-only unless a window is currently open
    /// (then the regular auto rule takes over on its next close).
    private func applyDockIconPreference() {
        let pref = dockIconAlwaysVisible
        guard pref != lastAppliedDockIconPref else { return }
        lastAppliedDockIconPref = pref

        if pref {
            if NSApp.activationPolicy() != .regular {
                NSApp.setActivationPolicy(.regular)
            }
            return
        }
        let anyVisible = NSApp.windows.contains { $0.isVisible && isAppOwnedWindow($0) }
        if !anyVisible, NSApp.activationPolicy() != .accessory {
            NSApp.setActivationPolicy(.accessory)
        }
    }

    private func bootstrapServer(config: AppConfig) {
        do {
            let runtime = try PythonRuntime.resolve()
            let server = ServerProcess(
                runtime: runtime,
                bindAddress: config.bindAddress,
                port: config.port,
                basePath: URL(fileURLWithPath: config.basePath, isDirectory: true)
            )
            self.server = server
            self.menubar = makeMenubar(server: server, config: config)
            services.bind(server: server)

            // Install signal handlers BEFORE the spawn so a fast crash of
            // the parent during startup still reaps any child we managed
            // to spawn.
            SignalHandlers.shared.install { [weak server] in
                server?.reapSync()
            }

            if config.autoStartOnLaunch {
                switch try server.start() {
                case .started, .alreadyRunning:
                    break
                case .portConflict:
                    // ServerProcess already posted .portConflictNotification +
                    // updated state to .failed; MenubarController will surface
                    // it on next click.
                    break
                }
            }
        } catch {
            // Surface the failure in the menubar header so the user has a
            // recovery affordance without needing to dig through logs.
            self.menubar = makeMenubar(server: nil, config: config, lastError: error)
            NSLog("oMLX: server bootstrap failed — \(error)")
        }
    }

    private func scheduleAccessoryPolicyFlip() {
        // Defer the policy flip so the status item has time to register
        // with WindowServer before we hide the Dock icon (mirrors
        // switchToAccessoryPolicy_ in app.py:324-327).
        DispatchQueue.main.async { [weak self] in
            guard self?.dockIconAlwaysVisible != true else { return }
            NSApp.setActivationPolicy(.accessory)
            NSApp.activate(ignoringOtherApps: true)
        }
    }

    // MARK: - Dock-icon toggle via NSWindow observers

    /// Wire NSWindow lifecycle notifications so the Dock icon follows the
    /// "any app window visible → .regular, none → .accessory" rule.
    /// Both the Welcome wizard and the SwiftUI main window participate; the
    /// menubar status item is not an NSWindow and is unaffected.
    ///
    /// Uses the selector-based observer API (not the closure-based one) so
    /// the non-Sendable Notification + NSWindow values don't need to cross
    /// an actor boundary. NSWindow.* notifications are delivered on the
    /// main thread per Apple's documented contract, so the AppDelegate's
    /// @MainActor methods receive them safely.
    private func installWindowObservers() {
        let center = NotificationCenter.default
        center.addObserver(self,
                           selector: #selector(windowDidBecomeMainNotification(_:)),
                           name: NSWindow.didBecomeMainNotification,
                           object: nil)
        center.addObserver(self,
                           selector: #selector(windowWillCloseNotification(_:)),
                           name: NSWindow.willCloseNotification,
                           object: nil)
    }

    @objc private func windowDidBecomeMainNotification(_ notif: Notification) {
        guard let win = notif.object as? NSWindow, isAppOwnedWindow(win) else { return }
        // Hide the SwiftUI Window scene's title text in the title bar
        // (the "oMLX" floating above the toolbar zone). The title string
        // is still used by the Window menu / Dock-icon right-click menu —
        // only the in-bar display is suppressed. Matches Settings.app's
        // chrome where the title bar is left to the per-screen big title
        // we render inside ContentScaffold.
        if win.identifier?.rawValue.contains("main") == true,
           win.titleVisibility != .hidden {
            win.titleVisibility = .hidden
        }
        if NSApp.activationPolicy() != .regular {
            NSApp.setActivationPolicy(.regular)
        }
    }

    @objc private func windowWillCloseNotification(_ notif: Notification) {
        guard let win = notif.object as? NSWindow, isAppOwnedWindow(win) else { return }
        let shouldDropDockIcon = dropDockIconOnNextClose
        // The closing window is still in NSApp.windows at notification time;
        // defer the visible-count check so it reflects post-close state.
        DispatchQueue.main.async {
            let stillVisible = NSApp.windows.contains { other in
                other !== win && other.isVisible && self.isAppOwnedWindow(other)
            }
            // Only drop to .accessory when the app initiated the close (Cmd-Q /
            // Dock Quit / Welcome wizard finish). Red-button close keeps the
            // Dock icon up so clicking it can re-open the window via
            // applicationShouldHandleReopen.
            if !stillVisible, shouldDropDockIcon, !self.dockIconAlwaysVisible {
                NSApp.setActivationPolicy(.accessory)
            }
        }
    }

    /// True for windows we own — excludes Sparkle's update windows, panel
    /// chrome from system services, etc. Heuristic: must be titled
    /// (so panel popovers don't count) and not excluded from the windows
    /// menu (so system status windows don't count).
    private func isAppOwnedWindow(_ win: NSWindow) -> Bool {
        guard win.styleMask.contains(.titled) else { return false }
        guard !win.isExcludedFromWindowsMenu else { return false }
        return true
    }

    // MARK: - Welcome wizard

    private func presentWelcome() {
        // First-run only — once `<basePath>/settings.json` exists,
        // `applicationDidFinishLaunching` takes the returning-user path and
        // this is never reached again.
        let controller = WelcomeWindowController(
            services: services,
            server: server,
            didFinish: { [weak self] _, finishedServer in
                // The wizard returns the spawned ServerProcess. Adopt it so
                // applicationWillTerminate can clean up correctly.
                guard let self else { return }
                self.server = finishedServer
                if let proc = finishedServer {
                    SignalHandlers.shared.install { [weak proc] in
                        proc?.reapSync()
                    }
                }
                self.menubar = self.makeMenubar(
                    server: finishedServer,
                    config: self.services.config
                )
            }
        )
        self.welcomeController = controller

        welcomeCloseObserver = NotificationCenter.default.addObserver(
            forName: WelcomeWindowController.willCloseNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.welcomeDidClose()
            }
        }

        controller.show()
    }

    private func welcomeDidClose() {
        if let observer = welcomeCloseObserver {
            NotificationCenter.default.removeObserver(observer)
            welcomeCloseObserver = nil
        }
        welcomeController = nil

        // Close before Start Server is cancellation: no menubar, no settings,
        // no base directory. Quit completely so relaunch shows Welcome again.
        guard server != nil else {
            explicitQuitRequested = true
            NSApp.terminate(nil)
            return
        }

        // The wizard spawned the server itself. Rebuild the menubar with the
        // running server state and switch to menubar-only mode.
        if let server, menubar != nil {
            self.menubar = makeMenubar(server: server, config: services.config)
        }
        scheduleAccessoryPolicyFlip()
    }

    func applicationWillTerminate(_ notification: Notification) {
        // Graceful stop. SIGKILL fallback is inside ServerProcess.stop().
        // We can't await indefinitely here — AppKit will eventually time
        // us out — so we run a short synchronous reap as belt-and-suspenders
        // (SignalHandlers also covers most external-kill paths).
        NotificationCenter.default.removeObserver(self)
        controlServer?.stop()
        controlServer = nil

        guard let server else { return }
        let group = DispatchGroup()
        group.enter()
        Task { @MainActor in
            await server.stop(timeout: 8)
            group.leave()
        }
        _ = group.wait(timeout: .now() + 9)
        server.reapSync(timeout: 1)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        // Menubar app — never quit on window close.
        false
    }

    /// Intercept terminate so Cmd-Q / Dock → Quit *only* close the window.
    /// The single real-quit path is the menubar status item's "Quit oMLX",
    /// which routes through `requestQuit()` to set the explicit flag.
    ///
    /// Notes:
    /// - We always cancel terminate when the explicit flag isn't set.
    ///   SwiftUI's Window scene appears to dismiss the window before
    ///   `applicationShouldTerminate` runs, so a "no visible windows"
    ///   guard fires when we'd really want to keep cancelling.
    /// - We use `close()`, not `performClose(_:)`. SwiftUI's window has a
    ///   delegate that vetoes `windowShouldClose:` in some cases — `close()`
    ///   bypasses that and reliably hides the window + fires
    ///   `willClose`/`didClose` so the Dock-icon observer drops to
    ///   `.accessory`.
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        if explicitQuitRequested { return .terminateNow }
        // Same close path used by the SwiftUI Cmd-Q command in oMLXApp.swift.
        hideWindowsAndDropDockIcon()
        return .terminateCancel
    }

    private func startControlServer() {
        let control = AppControlServer()
        control.handler = self
        do {
            try control.start()
            self.controlServer = control
        } catch {
            NSLog("oMLX: app-control server failed to start — \(error)")
        }
    }

    /// Dock icon click while no window is visible: bring the main window
    /// back. macOS calls this only when the user clicks the Dock icon of an
    /// already-running app whose windows are all hidden. With our policy
    /// of keeping the Dock icon up after a red-button close, this is the
    /// canonical "re-open" path.
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        if !flag {
            presentAppView()
        }
        return true
    }
}

extension AppDelegate: AppControlHandling {
    func handleAppControl(_ command: AppControlServer.Command) async -> AppControlServer.Response {
        guard let server else {
            return .failure(
                status: "unavailable",
                state: .stopped,
                server: nil,
                message: "Managed server is unavailable. Complete the oMLX first-run setup in the app."
            )
        }

        switch command {
        case .status:
            return .success(status: "ok", state: server.state, server: server)

        case .start:
            do {
                switch try server.start() {
                case .started:
                    return .success(status: "starting", state: server.state, server: server)
                case .alreadyRunning:
                    return .success(status: "running", state: server.state, server: server)
                case .portConflict(let conflict):
                    let pid = conflict.pid.map(String.init) ?? "unknown"
                    return .failure(
                        status: "port_conflict",
                        state: server.state,
                        server: server,
                        message: "Port \(server.port) is in use by PID \(pid)."
                    )
                }
            } catch {
                return .failure(
                    status: "error",
                    state: server.state,
                    server: server,
                    message: String(describing: error)
                )
            }

        case .stop:
            await server.stop()
            return .success(
                status: "stopped",
                state: server.state,
                server: server,
                message: "oMLX stopped"
            )

        case .restart:
            await server.stop()
            do {
                switch try server.start() {
                case .started:
                    return .success(status: "starting", state: server.state, server: server)
                case .alreadyRunning:
                    return .success(status: "running", state: server.state, server: server)
                case .portConflict(let conflict):
                    let pid = conflict.pid.map(String.init) ?? "unknown"
                    return .failure(
                        status: "port_conflict",
                        state: server.state,
                        server: server,
                        message: "Port \(server.port) is in use by PID \(pid)."
                    )
                }
            } catch {
                return .failure(
                    status: "error",
                    state: server.state,
                    server: server,
                    message: String(describing: error)
                )
            }
        }
    }
}

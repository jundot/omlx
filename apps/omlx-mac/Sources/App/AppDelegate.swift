// PR 5 (PR 10 update) — application delegate sequences activation policy +
// menubar + server bootstrap and installs POSIX signal handlers.
//
//   applicationWillFinishLaunching  → setActivationPolicy(.regular)
//   applicationDidFinishLaunching   → load AppConfig
//                                   → if first run (no config.json):
//                                       • create MenubarController without
//                                         a ServerProcess
//                                       • show Welcome window (the wizard
//                                         persists config.json + spawns the
//                                         server when the user clicks Start)
//                                       • flip to .accessory only after the
//                                         wizard window closes
//                                     else (returning user):
//                                       • resolve PythonRuntime
//                                       • spawn ServerProcess
//                                       • create MenubarController
//                                       • install SignalHandlers
//                                       • flip to .accessory next runloop tick
//   applicationWillTerminate        → await server.stop(timeout: 10)
//                                      (graceful SIGTERM → wait → SIGKILL inside)

import AppKit

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private(set) var server: ServerProcess?
    private var menubar: MenubarController?
    let services = AppServices()

    private var welcomeController: WelcomeWindowController?
    private var welcomeCloseObserver: NSObjectProtocol?

    nonisolated func applicationWillFinishLaunching(_ notification: Notification) {
        // Regular policy until the status item registers; we flip to Accessory
        // after creating the menubar (next runloop tick).
        DispatchQueue.main.async {
            NSApp.setActivationPolicy(.regular)
        }
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        let config = AppConfig.load()
        services.updateConfig(config)

        if AppConfig.configFileExists {
            bootstrapServer(config: config)
            scheduleAccessoryPolicyFlip()
        } else {
            // First run: stand up the menubar without a server, then run the
            // wizard. The wizard's "Start Server" creates a ServerProcess via
            // `services.bind(server:)`; AppDelegate adopts it back on close.
            self.menubar = MenubarController(server: nil, config: config)
            // Stay in .regular until the wizard closes so the user sees the
            // window in the Dock.
            NSApp.activate(ignoringOtherApps: true)
            presentWelcome()
        }
    }

    private func bootstrapServer(config: AppConfig) {
        do {
            let runtime = try PythonRuntime.resolve()
            let server = ServerProcess(
                runtime: runtime,
                host: config.host,
                port: config.port
            )
            self.server = server
            self.menubar = MenubarController(server: server, config: config)
            services.bind(server: server)

            // Install signal handlers BEFORE the spawn so a fast crash of
            // the parent during startup still reaps any child we managed
            // to spawn.
            SignalHandlers.shared.install { [weak server] in
                server?.reapSync()
            }

            switch try server.start() {
            case .started, .alreadyRunning:
                break
            case .portConflict:
                // ServerProcess already posted .portConflictNotification +
                // updated state to .failed; MenubarController will surface
                // it on next click.
                break
            }
        } catch {
            // Surface the failure in the menubar; in-app banner lands in PR 6.
            self.menubar = MenubarController(server: nil, config: config, lastError: error)
            NSLog("oMLX-next: server bootstrap failed — \(error)")
        }
    }

    private func scheduleAccessoryPolicyFlip() {
        // Defer the policy flip so the status item has time to register
        // with WindowServer before we hide the Dock icon (mirrors
        // switchToAccessoryPolicy_ in app.py:324-327).
        DispatchQueue.main.async {
            NSApp.setActivationPolicy(.accessory)
            NSApp.activate(ignoringOtherApps: true)
        }
    }

    private func presentWelcome() {
        let controller = WelcomeWindowController(
            services: services,
            server: server
        ) { [weak self] _, finishedServer in
            // The wizard returns the spawned ServerProcess. Adopt it so
            // applicationWillTerminate can clean up correctly.
            self?.server = finishedServer
            if let proc = finishedServer {
                SignalHandlers.shared.install { [weak proc] in
                    proc?.reapSync()
                }
            }
        }
        self.welcomeController = controller

        welcomeCloseObserver = NotificationCenter.default.addObserver(
            forName: WelcomeWindowController.willCloseNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            // Hop back into MainActor isolation to mutate AppDelegate state
            // safely under Swift Concurrency.
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

        // The wizard either spawned the server itself (success path) or the
        // user closed it without starting (skipped). Either way, drop the
        // app icon from the Dock and rebuild the menubar with whatever
        // state we ended up with.
        if let server, menubar != nil {
            self.menubar = MenubarController(
                server: server,
                config: services.config
            )
        }
        scheduleAccessoryPolicyFlip()
    }

    func applicationWillTerminate(_ notification: Notification) {
        // Graceful stop. SIGKILL fallback is inside ServerProcess.stop().
        // We can't await indefinitely here — AppKit will eventually time
        // us out — so we run a short synchronous reap as belt-and-suspenders
        // (SignalHandlers also covers most external-kill paths).
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
}

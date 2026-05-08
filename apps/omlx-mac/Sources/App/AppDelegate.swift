// PR 5 — application delegate now sequences activation policy + menubar +
// server bootstrap and installs POSIX signal handlers.
//
//   applicationWillFinishLaunching  → setActivationPolicy(.regular)
//   applicationDidFinishLaunching   → load AppConfig
//                                   → resolve PythonRuntime
//                                   → spawn ServerProcess
//                                   → create MenubarController
//                                   → install SignalHandlers (SIGTERM/INT/HUP/QUIT
//                                      synchronously reap the Python child)
//                                   → next runloop tick: setActivationPolicy(.accessory)
//   applicationWillTerminate        → await server.stop(timeout: 10)
//                                      (graceful SIGTERM → wait → SIGKILL inside)

import AppKit

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private(set) var server: ServerProcess?
    private var menubar: MenubarController?
    let services = AppServices()

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

        // Defer the policy flip so the status item has time to register
        // with WindowServer before we hide the Dock icon (mirrors
        // switchToAccessoryPolicy_ in app.py:324-327).
        DispatchQueue.main.async {
            NSApp.setActivationPolicy(.accessory)
            NSApp.activate(ignoringOtherApps: true)
        }
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

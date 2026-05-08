// PR 4 — application delegate now sequences activation policy + menubar +
// server bootstrap. Lifecycle invariants:
//
//   applicationWillFinishLaunching  → setActivationPolicy(.regular)
//                                      (must be set before WindowServer registers
//                                      our status item, see app.py:189-200)
//   applicationDidFinishLaunching   → load AppConfig
//                                   → resolve PythonRuntime
//                                   → spawn ServerProcess
//                                   → create MenubarController (registers status
//                                      item, starts stats poller + visibility watcher)
//                                   → on next runloop tick, switch policy to
//                                      .accessory (hides Dock icon)
//   applicationWillTerminate        → server.terminate()  (basic SIGTERM today;
//                                      PR 5 ports the SIGTERM→wait→SIGKILL chain
//                                      and the signal/atexit handlers).

import AppKit

final class AppDelegate: NSObject, NSApplicationDelegate {
    private(set) var server: ServerProcess?
    private var menubar: MenubarController?

    func applicationWillFinishLaunching(_ notification: Notification) {
        // Regular policy until the status item registers; we flip to Accessory
        // after creating the menubar (next runloop tick).
        NSApp.setActivationPolicy(.regular)
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        let config = AppConfig.load()

        do {
            let runtime = try PythonRuntime.resolve()
            let server = ServerProcess(
                runtime: runtime,
                host: config.host,
                port: config.port
            )
            self.server = server
            self.menubar = MenubarController(server: server, config: config)
            try server.start()
        } catch {
            // Surface the failure in the menubar; alert UX lands in PR 5.
            self.menubar = MenubarController(server: nil, config: config, lastError: error)
            NSLog("oMLX-next: server bootstrap failed — \(error)")
        }

        // Defer the policy flip by one runloop tick so the status item has
        // time to register with WindowServer before we hide the Dock icon
        // (mirrors switchToAccessoryPolicy_ in app.py:324-327).
        DispatchQueue.main.async {
            NSApp.setActivationPolicy(.accessory)
            NSApp.activate(ignoringOtherApps: true)
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        server?.terminate()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        // Menubar app — never quit on window close.
        false
    }
}

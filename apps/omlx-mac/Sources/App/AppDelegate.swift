// PR 2 — application delegate now owns the ServerProcess + MenubarController.
//
// Server starts at applicationDidFinishLaunching and terminates at
// applicationWillTerminate. Full SIGTERM→wait→SIGKILL sequencing + auto-restart
// land in PR 5.

import AppKit

final class AppDelegate: NSObject, NSApplicationDelegate {
    private(set) var server: ServerProcess?
    private var menubar: MenubarController?

    func applicationDidFinishLaunching(_ notification: Notification) {
        do {
            let runtime = try PythonRuntime.resolve()
            let server = ServerProcess(runtime: runtime)
            self.server = server
            self.menubar = MenubarController(server: server)
            try server.start()
        } catch {
            // Surface the failure in the menubar; a real alert UX lands in PR 5.
            self.menubar = MenubarController(server: nil, lastError: error)
            NSLog("oMLX-next: server bootstrap failed: \(error)")
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        server?.terminate()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        // Menubar app (LSUIElement) — never quit on window close.
        false
    }
}

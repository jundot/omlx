// PR 1 — minimal application delegate.
//
// Owns the menubar status item; full lifecycle (server spawn, welcome
// trigger, Sparkle, terminationHandler) lands in PR 2 / PR 5 / PR 11.

import AppKit

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var menubar: MenubarController?

    func applicationDidFinishLaunching(_ notification: Notification) {
        menubar = MenubarController()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        // We're a menubar app (LSUIElement) — closing the last window
        // must not terminate the process.
        false
    }
}

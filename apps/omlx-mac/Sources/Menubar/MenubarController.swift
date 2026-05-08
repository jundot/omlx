// PR 1 — status-item stub. A single Quit item proves end-to-end wiring
// (Info.plist LSUIElement, AppDelegate adoption, status bar visible).
//
// Full menubar parity (icon templates, dynamic menu, stats poll @1Hz,
// Bartender visibility watcher) lands in PR 4. See plan.md §4.

import AppKit

final class MenubarController: NSObject {
    private let statusItem: NSStatusItem

    override init() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        super.init()

        if let button = statusItem.button {
            button.image = NSImage(
                systemSymbolName: "cube.transparent",
                accessibilityDescription: "oMLX"
            )
            button.image?.isTemplate = true
        }

        let menu = NSMenu()
        let quit = NSMenuItem(
            title: "Quit oMLX-next",
            action: #selector(quitApp),
            keyEquivalent: "q"
        )
        quit.target = self
        menu.addItem(quit)
        statusItem.menu = menu
    }

    @objc private func quitApp() {
        NSApp.terminate(nil)
    }
}

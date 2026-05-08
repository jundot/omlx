// PR 2 — menubar still a stub, but now wired to ServerProcess so we can
// Start / Stop the spawned child. Full menubar parity (icon templates,
// dynamic menu, stats poll @1Hz, Bartender visibility watcher) lands in PR 4.

import AppKit

@MainActor
final class MenubarController: NSObject {
    private let statusItem: NSStatusItem
    private weak var server: ServerProcess?
    private let bootstrapError: Error?
    private var startItem: NSMenuItem?
    private var stopItem: NSMenuItem?
    private var statusHeader: NSMenuItem?

    init(server: ServerProcess?, lastError: Error? = nil) {
        self.statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        self.server = server
        self.bootstrapError = lastError
        super.init()

        if let button = statusItem.button {
            button.image = NSImage(
                systemSymbolName: "cube.transparent",
                accessibilityDescription: "oMLX"
            )
            button.image?.isTemplate = true
        }

        statusItem.menu = makeMenu()
        if let server {
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(serverStateChanged(_:)),
                name: ServerProcess.stateDidChangeNotification,
                object: server
            )
        }
        refreshMenuState()
    }

    @objc private func serverStateChanged(_ note: Notification) {
        refreshMenuState()
    }

    private func makeMenu() -> NSMenu {
        let menu = NSMenu()

        let header = NSMenuItem(title: "Server: …", action: nil, keyEquivalent: "")
        header.isEnabled = false
        menu.addItem(header)
        statusHeader = header

        menu.addItem(.separator())

        let start = NSMenuItem(
            title: "Start Server", action: #selector(startServer), keyEquivalent: ""
        )
        start.target = self
        menu.addItem(start)
        startItem = start

        let stop = NSMenuItem(
            title: "Stop Server", action: #selector(stopServer), keyEquivalent: ""
        )
        stop.target = self
        menu.addItem(stop)
        stopItem = stop

        menu.addItem(.separator())

        let quit = NSMenuItem(
            title: "Quit oMLX-next", action: #selector(quitApp), keyEquivalent: "q"
        )
        quit.target = self
        menu.addItem(quit)

        return menu
    }

    private func refreshMenuState() {
        let state = server?.state ?? .idle
        let header: String
        switch state {
        case .idle:
            header = bootstrapError.map { "Server: bootstrap failed (\($0))" }
                ?? "Server: idle"
        case .starting:                  header = "Server: starting…"
        case .running(let pid):          header = "Server: running (pid \(pid))"
        case .stopped:                   header = "Server: stopped"
        case .failed(let message):       header = "Server: failed — \(message)"
        }
        statusHeader?.title = header

        let running: Bool
        if case .running = state { running = true } else { running = false }
        startItem?.isHidden = running
        stopItem?.isHidden = !running
        startItem?.isEnabled = (server != nil) && !running
        stopItem?.isEnabled = running
    }

    @objc private func startServer() {
        guard let server else { return }
        do { try server.start() }
        catch { NSLog("oMLX-next: start failed: \(error)") }
    }

    @objc private func stopServer() {
        server?.terminate()
    }

    @objc private func quitApp() {
        NSApp.terminate(nil)
    }
}

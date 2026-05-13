// Hosts AppView in a manual NSWindow instead of SwiftUI's `Settings` scene.
//
// `Settings { AppView() }` looks tidy on paper but doesn't survive the
// .accessory activation policy this app runs under: `showSettingsWindow:`
// silently no-ops when the responder chain has no first responder, and
// even when it fires the window opens behind the foreground app.
//
// Owning the window ourselves gives a deterministic
// `activate → makeKeyAndOrderFront` path that works the same whether the
// menubar item, an external trigger, or a future RemoteTrigger calls in.

import AppKit
import SwiftUI

@MainActor
final class AppViewWindowController: NSWindowController, NSWindowDelegate {
    private let services: AppServices

    init(services: AppServices) {
        self.services = services

        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1140, height: 760),
            styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        window.title = "oMLX"
        window.minSize = NSSize(width: 880, height: 600)
        window.isReleasedWhenClosed = false
        window.titlebarAppearsTransparent = false
        window.collectionBehavior.insert(.fullScreenPrimary)
        window.center()

        let host = NSHostingController(
            rootView: AppView()
                .environmentObject(services)
        )
        window.contentViewController = host

        super.init(window: window)
        window.delegate = self
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { fatalError("AppViewWindowController is code-only") }

    /// Bring the AppView window to the front. Safe to call repeatedly —
    /// the window is reused across opens (isReleasedWhenClosed = false).
    func present() {
        guard let window else { return }
        if !window.isVisible {
            // Re-center if reopening from a closed state so it doesn't pop
            // up offscreen on a different display arrangement.
            window.center()
        }
        NSApp.activate(ignoringOtherApps: true)
        window.makeKeyAndOrderFront(nil)
    }
}

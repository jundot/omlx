// SwiftUI shell. The main AppView is a `Window` scene managed by SwiftUI
// (state restoration, autosave, opt-in lifecycle). AppDelegate stays in
// charge of the menubar + server bootstrap + Welcome wizard.
//
// Window lifecycle
//   • `.defaultLaunchBehavior(.suppressed)` keeps the window from appearing
//     at launch — we're a menubar-first app and the user opens it via the
//     status-item's "Admin Panel" command (or the Welcome wizard on first
//     run, which lives in its own manual NSWindow controller).
//   • `.handlesExternalEvents(matching: ["main"])` lets AppDelegate trigger
//     the window the FIRST time via `NSWorkspace.shared.open(omlxapp://main)`
//     when no NSWindow instance has been created yet. Subsequent shows
//     just `makeKeyAndOrderFront` the cached window.
//   • Dock-icon toggle (regular when visible, accessory when closed) is
//     handled by AppDelegate via NSWindow notification observers — not in
//     this file — so the welcome flow shares the same dock-icon logic.

import SwiftUI

@main
struct OMLXApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        // Empty title string keeps the toolbar zone free of "oMLX" text
        // (SwiftUI macOS 26 renders the Window title in the unified toolbar
        // regardless of NSWindow.titleVisibility). The Window menu / Dock
        // right-click menu show the bundle display name ("oMLX") as a
        // fallback when title is empty, so we don't lose the in-menu name.
        Window("", id: "main") {
            AppView()
                .environment(appDelegate.services)
        }
        .defaultLaunchBehavior(.suppressed)
        .handlesExternalEvents(matching: ["main"])
        .windowResizability(.contentMinSize)
        // Replace the system "Quit oMLX" command (Cmd-Q from the in-app
        // menu). Cmd-Q hides every visible window AND drops the Dock icon
        // — same path as Dock → Quit (`applicationShouldTerminate`). The
        // menubar status item's "Quit oMLX" remains the only path to fully
        // terminate.
        .commands {
            CommandGroup(replacing: .appTermination) {
                Button("Close Window") {
                    appDelegate.hideWindowsAndDropDockIcon()
                }
                .keyboardShortcut("q", modifiers: .command)
            }
        }
    }
}

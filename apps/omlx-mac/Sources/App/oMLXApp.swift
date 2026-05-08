// PR 1 (PR 6 update) — SwiftUI shell. The `Settings` scene now hosts the real
// `AppView` instead of a placeholder. macOS routes both `Cmd-,` and the
// menubar's `Admin Panel` item (via `showSettingsWindow:`) to this scene, so
// we get one place to drive every screen and one window for the user to find.

import SwiftUI

@main
struct OMLXApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        Settings {
            AppView()
                .frame(
                    minWidth: 880, idealWidth: 1140, maxWidth: .infinity,
                    minHeight: 600, idealHeight: 760, maxHeight: .infinity
                )
        }
    }
}

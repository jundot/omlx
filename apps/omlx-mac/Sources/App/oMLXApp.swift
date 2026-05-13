// SwiftUI shell. AppView no longer lives in the `Settings` scene because
// `showSettingsWindow:` is unreliable for `.accessory` apps — see
// `AppViewWindowController.swift`. The menubar's "Admin Panel" item now
// presents the window directly through AppDelegate, so this scene is just
// the placeholder SwiftUI's `App` protocol requires.

import SwiftUI

@main
struct OMLXApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        Settings { EmptyView() }
    }
}

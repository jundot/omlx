// PR 1 — minimal SwiftUI shell. Replaced by AppView (PR 6) and full
// scene wiring (PR 6+) over subsequent phases. See plan.md.

import SwiftUI

@main
struct OMLXApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        // Placeholder Settings scene; replaced by AppView in PR 6.
        Settings {
            VStack(spacing: 8) {
                Text("oMLX-next")
                    .font(.system(size: 17, weight: .semibold))
                Text("Configuration arrives in PR 6.")
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
            }
            .padding(40)
            .frame(minWidth: 360, minHeight: 200)
        }
    }
}

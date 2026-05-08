// PR 6 — placeholder. Real Status (system info, live stats, device telemetry)
// + the Updates section (per design v2: three-state status row, Channel popup,
// AutoCheck/AutoDownload toggles) land in PR 7. Sparkle wiring lands in PR 11.

import SwiftUI

struct StatusScreen: View {
    var body: some View {
        PlaceholderScreen(
            landsIn: .pr7,
            summary: "System info, live tok/s + cache stats, device telemetry, Updates section (Stable / Beta / Nightly)."
        )
    }
}

// PR 6 — placeholder. Real Server configuration UI (host/port/auth + endpoint
// list, Start/Stop/Restart hero) lands in PR 7.

import SwiftUI

struct ServerScreen: View {
    var body: some View {
        PlaceholderScreen(
            landsIn: .pr7,
            summary: "Listen address, auto-start, API-key gate, /v1 endpoint list, Start / Stop / Restart hero."
        )
    }
}

// PR 6 — placeholder. Version + license + credits + links land in PR 9.
// Inline updater UI does NOT live here — the Updates section is part of the
// Status screen per design v2. Sparkle wiring lands in PR 11.

import SwiftUI

struct AboutScreen: View {
    var body: some View {
        PlaceholderScreen(
            landsIn: .pr9,
            summary: "Build info, license, credits, project links. The updater lives on Status (design v2)."
        )
    }
}

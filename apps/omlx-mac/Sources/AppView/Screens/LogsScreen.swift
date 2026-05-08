// PR 6 — placeholder. Real log tail UI lands in PR 7.

import SwiftUI

struct LogsScreen: View {
    var body: some View {
        PlaceholderScreen(
            landsIn: .pr7,
            summary: "Tail of the bundled server's log with level filter and copy-to-clipboard."
        )
    }
}

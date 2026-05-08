// PR 6 — shared empty-content stub for sidebar items whose real surface lands
// in a later PR. Each `XScreen` view in this folder boils down to a
// `PlaceholderScreen(landsIn: .prN, summary: "…")` until its phase ships.

import SwiftUI

enum LandingPR: String, Sendable {
    case pr7  = "PR 7"
    case pr8  = "PR 8"
    case pr9  = "PR 9"
    case pr11 = "PR 11"

    var headline: String {
        switch self {
        case .pr7:  return "Lands in PR 7"
        case .pr8:  return "Lands in PR 8"
        case .pr9:  return "Lands in PR 9"
        case .pr11: return "Updater wires up in PR 11"
        }
    }
}

struct PlaceholderScreen: View {
    let landsIn: LandingPR
    let summary: String

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            SectionHeader(landsIn.headline)
            ListGroup {
                Row(label: summary, isLast: true)
            }
        }
        .frame(maxWidth: .infinity, alignment: .topLeading)
    }
}

#Preview("PlaceholderScreen — light") {
    PlaceholderScreen(
        landsIn: .pr7,
        summary: "Server / Status / Logs configuration UI is the first real consumer of the design system."
    )
    .padding(28)
    .frame(width: 720)
    .omlxThemed()
    .preferredColorScheme(.light)
}

#Preview("PlaceholderScreen — dark") {
    PlaceholderScreen(
        landsIn: .pr8,
        summary: "Active models, library, downloads + per-model Profiles / Basic / Advanced / Aliases."
    )
    .padding(28)
    .frame(width: 720)
    .omlxThemed()
    .preferredColorScheme(.dark)
}

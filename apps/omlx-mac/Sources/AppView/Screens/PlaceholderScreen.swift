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
        case .pr7:
            return String(localized: "placeholder.headline.pr7",
                          defaultValue: "Lands in PR 7",
                          comment: "Placeholder section header for a screen whose real content ships in PR 7")
        case .pr8:
            return String(localized: "placeholder.headline.pr8",
                          defaultValue: "Lands in PR 8",
                          comment: "Placeholder section header for a screen whose real content ships in PR 8")
        case .pr9:
            return String(localized: "placeholder.headline.pr9",
                          defaultValue: "Lands in PR 9",
                          comment: "Placeholder section header for a screen whose real content ships in PR 9")
        case .pr11:
            return String(localized: "placeholder.headline.pr11",
                          defaultValue: "Updater wires up in PR 11",
                          comment: "Placeholder section header for the Updates surface that wires up in PR 11")
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

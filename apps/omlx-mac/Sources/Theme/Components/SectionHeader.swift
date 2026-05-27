// PR 3 — small uppercase section header used above every ListGroup.

import SwiftUI

struct SectionHeader<Trailing: View>: View {
    let title: String
    let subtitle: String?
    let trailing: Trailing

    @Environment(\.omlxTheme) private var theme

    init(
        _ title: String,
        subtitle: String? = nil,
        @ViewBuilder trailing: () -> Trailing = { EmptyView() }
    ) {
        self.title = title
        self.subtitle = subtitle
        self.trailing = trailing()
    }

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.omlxText(11, weight: .semibold))
                    .foregroundStyle(theme.textSecondary)
                    .textCase(.uppercase)
                    .kerning(0.6)
                if let subtitle, !subtitle.isEmpty {
                    Text(subtitle)
                        .font(.omlxText(11))
                        .foregroundStyle(theme.textTertiary)
                }
            }
            Spacer(minLength: 0)
            trailing
        }
        .padding(.horizontal, 14)
        .padding(.top, 14)
        .padding(.bottom, 6)
    }
}

extension SectionHeader where Trailing == EmptyView {
    init(_ title: String, subtitle: String? = nil) {
        self.title = title
        self.subtitle = subtitle
        self.trailing = EmptyView()
    }
}

#Preview("SectionHeader") {
    VStack(alignment: .leading, spacing: 8) {
        SectionHeader("Network")
        SectionHeader("System", subtitle: "Apple Silicon · macOS 15.2")
        SectionHeader("Active Models") {
            Text("3 / 5")
                .font(.omlxText(11))
                .foregroundStyle(.secondary)
        }
    }
    .padding(24)
    .frame(width: 480)
    .omlxThemed()
}

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
                    .font(.omlxText(DesignTokens.FontSize.subheadline, weight: .semibold))
                    .foregroundStyle(theme.textSecondary)
                    .textCase(.uppercase)
                    .kerning(0.6)
                if let subtitle, !subtitle.isEmpty {
                    Text(subtitle)
                        .font(.omlxText(DesignTokens.FontSize.subheadline))
                        .foregroundStyle(theme.textTertiary)
                }
            }
            Spacer(minLength: 0)
            trailing
        }
        // 28 = card gutter (14) + row inset (14): header text lines up with
        // the row text inside the ListGroup below, like System Settings.
        .padding(.horizontal, 28)
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

/// Collapsible header for rarely changed settings (Server → Advanced,
/// Performance → Advanced). Same type treatment as `SectionHeader`, with a
/// chevron; the caller shows its rows only while `isExpanded` is true.
struct DisclosureSectionHeader: View {
    let title: String
    var subtitle: String? = nil
    @Binding var isExpanded: Bool

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        Button {
            isExpanded.toggle()
        } label: {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Image(systemName: "chevron.right")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(theme.textSecondary)
                        .rotationEffect(.degrees(isExpanded ? 90 : 0))
                        .animation(.easeOut(duration: 0.12), value: isExpanded)
                    Text(title)
                        .font(.omlxText(DesignTokens.FontSize.subheadline, weight: .semibold))
                        .foregroundStyle(theme.textSecondary)
                        .textCase(.uppercase)
                        .kerning(0.6)
                    Spacer()
                }
                if isExpanded, let subtitle, !subtitle.isEmpty {
                    Text(subtitle)
                        .font(.omlxText(DesignTokens.FontSize.subheadline))
                        .foregroundStyle(theme.textTertiary)
                        .multilineTextAlignment(.leading)
                }
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        // Matches SectionHeader's indent (card gutter + row inset).
        .padding(.horizontal, 28)
        .padding(.top, 22)
        .padding(.bottom, 8)
    }
}

#Preview("SectionHeader") {
    VStack(alignment: .leading, spacing: 8) {
        SectionHeader("Network")
        SectionHeader("System", subtitle: "Apple Silicon · macOS 15.2")
        SectionHeader("Active Models") {
            Text("3 / 5")
                .font(.omlxText(DesignTokens.FontSize.subheadline))
                .foregroundStyle(.secondary)
        }
    }
    .padding(24)
    .frame(width: 480)
    .omlxThemed()
}

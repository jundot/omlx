// PR 3 — labeled row, the workhorse layout inside `ListGroup`. Mirrors JSX `Row`.
// Two variants:
//   • `Row(label:sublabel:isLast:trailing:)` — most uses
//   • `Row(isLast:content:)` — free-form (chip strips, custom layouts)

import SwiftUI

struct Row<Trailing: View>: View {
    let label: String?
    let sublabel: String?
    let warning: String?
    let isLast: Bool
    let trailing: Trailing

    @Environment(\.omlxTheme) private var theme

    init(
        label: String? = nil,
        sublabel: String? = nil,
        warning: String? = nil,
        isLast: Bool = false,
        @ViewBuilder trailing: () -> Trailing
    ) {
        self.label = label
        self.sublabel = sublabel
        self.warning = warning
        self.isLast = isLast
        self.trailing = trailing()
    }

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            if let label {
                VStack(alignment: .leading, spacing: 2) {
                    Text(label)
                        .font(.omlxText(DesignTokens.FontSize.body, weight: .medium))
                        .foregroundStyle(theme.text)
                    if let sublabel, !sublabel.isEmpty {
                        Text(sublabel)
                            .font(.omlxText(11.5))
                            .foregroundStyle(theme.textSecondary)
                            .lineLimit(2)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    if let warning, !warning.isEmpty {
                        Text(warning)
                            .font(.omlxText(11.5))
                            .foregroundStyle(theme.warningText)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                Spacer(minLength: 12)
            }
            trailing
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .frame(minHeight: 36)
        .frame(maxWidth: .infinity, alignment: label == nil ? .leading : .center)
        .overlay(alignment: .bottom) {
            if !isLast {
                Rectangle()
                    .fill(theme.rowSep)
                    .frame(height: 0.5)
                    .padding(.horizontal, 14)
            }
        }
    }
}

/// Trailing on/off switch for a `Row`. One shared control so every screen
/// gets the same size: `.small`, matching System Settings rows — the default
/// regular switch reads oversized next to 13 pt row labels.
struct RowSwitch: View {
    @Binding var isOn: Bool

    init(isOn: Binding<Bool>) {
        self._isOn = isOn
    }

    var body: some View {
        Toggle("", isOn: $isOn)
            .labelsHidden()
            .toggleStyle(.switch)
            .controlSize(.small)
    }
}

extension Row where Trailing == EmptyView {
    /// Label-only row (no trailing slot).
    init(label: String, sublabel: String? = nil, isLast: Bool = false) {
        self.label = label
        self.sublabel = sublabel
        self.warning = nil
        self.isLast = isLast
        self.trailing = EmptyView()
    }
}

/// Free-form variant — caller provides the entire row body. Used for chip
/// strips, multi-line custom rows, etc.
struct FreeRow<Content: View>: View {
    let isLast: Bool
    let content: () -> Content

    @Environment(\.omlxTheme) private var theme

    init(isLast: Bool = false, @ViewBuilder content: @escaping () -> Content) {
        self.isLast = isLast
        self.content = content
    }

    var body: some View {
        content()
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .overlay(alignment: .bottom) {
                if !isLast {
                    Rectangle()
                        .fill(theme.rowSep)
                        .frame(height: 0.5)
                        .padding(.horizontal, 14)
                }
            }
    }
}

struct LinkRow<Icon: View>: View {
    let label: String
    let sublabel: String
    let url: URL
    var isLast: Bool = false

    private let iconView: Icon

    @Environment(\.omlxTheme) private var theme

    init(
        label: String,
        sublabel: String,
        url: URL,
        isLast: Bool = false,
        @ViewBuilder iconView: () -> Icon
    ) {
        self.label = label
        self.sublabel = sublabel
        self.url = url
        self.isLast = isLast
        self.iconView = iconView()
    }

    init(
        label: String,
        sublabel: String,
        icon: String,
        url: URL,
        isLast: Bool = false
    ) where Icon == AnyView {
        self.label = label
        self.sublabel = sublabel
        self.url = url
        self.isLast = isLast
        self.iconView = AnyView(
            Image(systemName: icon)
                .font(.system(size: 14))
                .frame(width: 18, alignment: .center)
        )
    }

    @State private var hovering = false

    // The whole row is one button (System Settings convention for rows
    // that open a URL); the trailing ↗ is a passive indicator, not a
    // separate control. Hover feedback replaces FreeRow so the highlight
    // spans the full row width.
    var body: some View {
        Button {
            NSWorkspace.shared.open(url)
        } label: {
            HStack(spacing: 10) {
                iconView
                    .foregroundStyle(theme.textSecondary)

                VStack(alignment: .leading, spacing: 2) {
                    Text(label)
                        .font(.omlxText(DesignTokens.FontSize.body, weight: .medium))
                        .foregroundStyle(theme.text)
                    Text(sublabel)
                        .font(.omlxText(DesignTokens.FontSize.subheadline))
                        .foregroundStyle(theme.textSecondary)
                        .lineLimit(2)
                }

                Spacer(minLength: 8)

                Image(systemName: "arrow.up.right.square")
                    .font(.system(size: 14))
                    .foregroundStyle(theme.textSecondary)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .background(hovering ? theme.hoverBg : .clear)
        .onHover { hovering = $0 }
        .overlay(alignment: .bottom) {
            if !isLast {
                Rectangle()
                    .fill(theme.rowSep)
                    .frame(height: 0.5)
                    .padding(.horizontal, 14)
            }
        }
    }
}

/// Row that opens a sub-page, System Settings style: the trailing value
/// summarizes the sub-page and the chevron marks the row as navigation.
struct NavRow<Icon: View>: View {
    let label: String
    let value: String?
    var isLast: Bool = false
    var isDimmed: Bool = false
    let action: () -> Void

    private let icon: Icon

    @Environment(\.omlxTheme) private var theme
    @State private var hovering = false

    init(
        label: String,
        value: String? = nil,
        isLast: Bool = false,
        isDimmed: Bool = false,
        action: @escaping () -> Void,
        @ViewBuilder icon: () -> Icon
    ) {
        self.label = label
        self.value = value
        self.isLast = isLast
        self.isDimmed = isDimmed
        self.action = action
        self.icon = icon()
    }

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                icon
                Text(label)
                    .font(.omlxText(DesignTokens.FontSize.body, weight: .medium))
                    .foregroundStyle(theme.text)
                    .lineLimit(1)
                Spacer(minLength: 12)
                if let value, !value.isEmpty {
                    Text(value)
                        .font(.omlxText(DesignTokens.FontSize.callout))
                        .foregroundStyle(theme.textSecondary)
                        .lineLimit(1)
                        .truncationMode(.tail)
                }
                Image(systemName: "chevron.right")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(theme.textTertiary)
            }
            .opacity(isDimmed ? 0.5 : 1)
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .frame(minHeight: 36)
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .background(hovering ? theme.hoverBg : .clear)
        .onHover { hovering = $0 }
        .overlay(alignment: .bottom) {
            if !isLast {
                Rectangle()
                    .fill(theme.rowSep)
                    .frame(height: 0.5)
                    .padding(.horizontal, 14)
            }
        }
    }
}

#Preview("Row in ListGroup") {
    @Previewable @State var autoStart = true
    @Previewable @State var requireKey = false

    return ListGroup {
        Row(
            label: "Listen Address",
            sublabel: "Default 8000. Restart server to apply."
        ) {
            CodeChip(value: "127.0.0.1:8000")
        }
        Row(label: "Auto-start on launch") {
            RowSwitch(isOn: $autoStart)
        }
        Row(
            label: "Require API Key",
            sublabel: "Reject unauthenticated /v1 requests",
            isLast: true
        ) {
            RowSwitch(isOn: $requireKey)
        }
    }
    .padding(.vertical, 14)
    .frame(width: 560)
    .omlxThemed()
}

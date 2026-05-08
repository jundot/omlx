// PR 3 — button styles for primary / destructive / plain / regular.
//
// Use:
//   Button("Save") { … }
//     .buttonStyle(.omlx(.primary))
//
// The plain kind is the JSX "kind=plain" — borderless action label, e.g. the
// chevron-only row buttons in screens.

import SwiftUI

struct OMLXButtonStyle: ButtonStyle {
    enum Kind: Sendable { case primary, destructive, normal, plain }
    enum Size: Sendable { case small, regular }

    let kind: Kind
    let size: Size

    @Environment(\.omlxTheme) private var theme

    func makeBody(configuration: Configuration) -> some View {
        let labelFont = Font.omlxText(size == .small ? 11.5 : 13, weight: .medium)
        let hPad: CGFloat = size == .small ? 10 : 12
        let vPad: CGFloat = size == .small ? 4 : 6

        return configuration.label
            .font(labelFont)
            .padding(.horizontal, hPad)
            .padding(.vertical, vPad)
            .foregroundStyle(foreground(configuration))
            .background(background(configuration))
            .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
            .overlay(border(configuration))
            .opacity(configuration.isPressed ? 0.78 : 1.0)
            .contentShape(Rectangle())
    }

    @ViewBuilder
    private func background(_ cfg: Configuration) -> some View {
        switch kind {
        case .primary:
            theme.accent
        case .destructive:
            theme.redDot
        case .normal:
            theme.controlBg
        case .plain:
            cfg.isPressed ? theme.hoverBg : Color.clear
        }
    }

    private func foreground(_ cfg: Configuration) -> Color {
        switch kind {
        case .primary, .destructive: return theme.accentText
        case .normal: return theme.text
        case .plain:  return theme.text
        }
    }

    @ViewBuilder
    private func border(_ cfg: Configuration) -> some View {
        if kind == .normal {
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .strokeBorder(theme.inputBorder, lineWidth: 0.5)
        }
    }
}

extension ButtonStyle where Self == OMLXButtonStyle {
    static func omlx(
        _ kind: OMLXButtonStyle.Kind = .normal,
        size: OMLXButtonStyle.Size = .regular
    ) -> OMLXButtonStyle {
        OMLXButtonStyle(kind: kind, size: size)
    }
}

#Preview("Buttons") {
    VStack(alignment: .leading, spacing: 14) {
        HStack(spacing: 8) {
            Button("Save") {}.buttonStyle(.omlx(.primary))
            Button("Save") {}.buttonStyle(.omlx(.normal))
            Button("Delete") {}.buttonStyle(.omlx(.destructive))
            Button("Cancel") {}.buttonStyle(.omlx(.plain))
        }
        HStack(spacing: 8) {
            Button("Load") {}.buttonStyle(.omlx(.primary, size: .small))
            Button("Unload") {}.buttonStyle(.omlx(.normal, size: .small))
            Button { } label: {
                Image(systemName: "trash")
            }.buttonStyle(.omlx(.plain, size: .small))
        }
    }
    .padding(24)
    .omlxThemed()
}

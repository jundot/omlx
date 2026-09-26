// Button styles for primary / destructive / normal / plain.
//
// Use:
//   Button("Save") { … }
//     .buttonStyle(.omlx(.primary))
//
// primary / destructive / normal delegate to the native bordered styles so
// buttons share bezel metrics, fonts, and disabled/pressed treatment with
// the native fields and pickers they sit next to. plain stays a quiet
// custom label (text-colored, hover-highlight only) because the native
// borderless style would accent-tint the many icon-only row buttons.

import SwiftUI

struct OMLXButtonStyle: PrimitiveButtonStyle {
    enum Kind: Sendable { case primary, destructive, normal, plain }
    enum Size: Sendable { case small, regular }

    let kind: Kind
    let size: Size

    @Environment(\.omlxTheme) private var theme

    @ViewBuilder
    func makeBody(configuration: Configuration) -> some View {
        let button = Button(configuration)
            .controlSize(size == .small ? .small : .regular)
        switch kind {
        case .primary:
            button.buttonStyle(.borderedProminent)
        case .destructive:
            button.buttonStyle(.borderedProminent).tint(theme.redDot)
        case .normal:
            button.buttonStyle(.bordered)
        case .plain:
            button.buttonStyle(QuietButtonStyle(theme: theme, size: size))
        }
    }
}

/// The former custom "plain" rendering: label in text color, hover-style
/// highlight while pressed, dimmed when disabled.
private struct QuietButtonStyle: ButtonStyle {
    let theme: OMLXTheme
    let size: OMLXButtonStyle.Size

    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.omlxText(size == .small ? 11.5 : 13, weight: .medium))
            .padding(.horizontal, size == .small ? 10 : 12)
            .padding(.vertical, size == .small ? 4 : 6)
            .foregroundStyle(theme.text)
            .background(configuration.isPressed ? theme.hoverBg : Color.clear)
            .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
            .opacity(isEnabled ? (configuration.isPressed ? 0.78 : 1.0) : 0.45)
            .contentShape(Rectangle())
    }
}

extension PrimitiveButtonStyle where Self == OMLXButtonStyle {
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

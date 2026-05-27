// PR 3 — text field with focus ring matching the JSX `TextInput` / `BoxedInput`.

import SwiftUI

struct TextInput: View {
    @Binding var text: String
    var placeholder: String = ""
    var isSecure: Bool = false
    var mono: Bool = false
    var suffix: String? = nil
    var width: CGFloat? = nil

    @Environment(\.omlxTheme) private var theme
    @FocusState private var isFocused: Bool

    var body: some View {
        HStack(spacing: 6) {
            field
                .focused($isFocused)
                .textFieldStyle(.plain)
                .font(mono ? .omlxMono(13, weight: .medium)
                            : .omlxText(13, weight: .medium))
                .foregroundStyle(theme.text)
            if let suffix {
                Text(suffix)
                    .font(.omlxText(11))
                    .foregroundStyle(theme.textSecondary)
            }
        }
        .padding(.horizontal, 10)
        .frame(height: 28)
        .frame(maxWidth: width)
        .background(theme.inputBg)
        .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 7, style: .continuous)
                .strokeBorder(
                    isFocused ? theme.inputBorderFocus : theme.inputBorder,
                    lineWidth: 0.5
                )
        )
        // Focus glow — softer than NSFocusRing but consistent with design.
        .overlay(
            RoundedRectangle(cornerRadius: 7, style: .continuous)
                .stroke(theme.accent.opacity(isFocused ? 0.20 : 0), lineWidth: 3)
                .padding(-2)
                .allowsHitTesting(false)
        )
        .animation(.easeOut(duration: 0.08), value: isFocused)
    }

    @ViewBuilder
    private var field: some View {
        if isSecure {
            SecureField(placeholder, text: $text)
        } else {
            TextField(placeholder, text: $text)
        }
    }
}

#Preview("TextInput") {
    @Previewable @State var port = "8080"
    @Previewable @State var pwd = "sk-omlx-2k4j8"
    @Previewable @State var alias = ""
    return VStack(alignment: .leading, spacing: 14) {
        TextInput(text: $port, placeholder: "Port", mono: true, width: 110)
        TextInput(text: $pwd, placeholder: "Admin password", isSecure: true, width: 200)
        TextInput(text: $alias, placeholder: "model-id-suffix", mono: true,
                  suffix: "alias", width: 240)
    }
    .padding(24)
    .omlxThemed()
}

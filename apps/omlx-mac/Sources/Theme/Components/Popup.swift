// PR 3 — dropdown picker styled to match the JSX `Popup`.

import SwiftUI

struct PopupOption<Value: Hashable>: Identifiable {
    let value: Value
    let label: String
    var id: Value { value }
}

struct Popup<Value: Hashable>: View {
    @Binding var selection: Value
    let options: [PopupOption<Value>]
    let width: CGFloat?

    @Environment(\.omlxTheme) private var theme

    init(selection: Binding<Value>, width: CGFloat? = nil, options: [PopupOption<Value>]) {
        self._selection = selection
        self.options = options
        self.width = width
    }

    init(
        selection: Binding<Value>,
        width: CGFloat? = nil,
        options: [(Value, String)]
    ) {
        self._selection = selection
        self.options = options.map { PopupOption(value: $0.0, label: $0.1) }
        self.width = width
    }

    var body: some View {
        Menu {
            ForEach(options) { opt in
                Button {
                    selection = opt.value
                } label: {
                    HStack {
                        Text(opt.label)
                        if opt.value == selection {
                            Spacer()
                            Image(systemName: "checkmark")
                        }
                    }
                }
            }
        } label: {
            HStack(spacing: 8) {
                Text(currentLabel)
                    .font(.omlxText(13, weight: .medium))
                    .foregroundStyle(theme.text)
                    .lineLimit(1)
                    .truncationMode(.tail)
                Spacer(minLength: 4)
                Image(systemName: "chevron.up.chevron.down")
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(theme.textSecondary)
            }
            .padding(.horizontal, 10)
            .frame(height: 28)
            .frame(maxWidth: width)
            .background(theme.inputBg)
            .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 7, style: .continuous)
                    .strokeBorder(theme.inputBorder, lineWidth: 0.5)
            )
            .contentShape(Rectangle())
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
    }

    private var currentLabel: String {
        options.first(where: { $0.value == selection })?.label ?? "—"
    }
}

#Preview("Popup") {
    @Previewable @State var host = "127.0.0.1"
    @Previewable @State var quant = "q4"
    return VStack(alignment: .leading, spacing: 14) {
        Popup(selection: $host, width: 220, options: [
            ("127.0.0.1", "127.0.0.1 (Local only)"),
            ("0.0.0.0", "0.0.0.0 (All networks)"),
            ("localhost", "localhost"),
        ])
        Popup(selection: $quant, width: 120, options: [
            ("auto", "Auto"), ("q4", "q4"), ("q5", "q5"), ("q6", "q6"), ("q8", "q8"), ("fp16", "fp16"),
        ])
    }
    .padding(24)
    .omlxThemed()
}

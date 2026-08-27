// PR 3 — dropdown picker styled to match the JSX `Popup`.

import SwiftUI

struct PopupOption<Value: Hashable>: Identifiable {
    let value: Value
    let label: String
    var id: Value { value }
}

struct Popup<Value: Hashable>: View {
    @Binding var selection: Value
    var titleKey: LocalizedStringKey
    let options: [PopupOption<Value>]
    let width: CGFloat?
    let fillsWidth: Bool

    @Environment(\.omlxTheme) private var theme

    init(_ titleKey: LocalizedStringKey = "", selection: Binding<Value>, width: CGFloat? = nil, fillsWidth: Bool = false, options: [PopupOption<Value>]) {
        self.titleKey = titleKey
        self._selection = selection
        self.options = options
        self.width = width
        self.fillsWidth = fillsWidth
    }

    init(_ titleKey: LocalizedStringKey = "", selection: Binding<Value>, width: CGFloat? = nil, fillsWidth: Bool = false, options: [(Value, String)]) {
        self.titleKey = titleKey
        self._selection = selection
        self.options = options.map { PopupOption(value: $0.0, label: $0.1) }
        self.width = width
        self.fillsWidth = fillsWidth
    }

    var body: some View {
        if fillsWidth, let width {
            Menu {
                ForEach(options) { opt in
                    Button {
                        selection = opt.value
                    } label: {
                        if opt.value == selection {
                            Label(opt.label, systemImage: "checkmark")
                        } else {
                            Text(opt.label)
                        }
                    }
                }
            } label: {
                HStack(spacing: 8) {
                    Text(selectedOption?.label ?? "")
                        .lineLimit(1)
                    Spacer(minLength: 0)
                    Image(systemName: "chevron.up.chevron.down")
                        .font(.system(size: 10, weight: .semibold))
                }
                .font(.omlxText(13, weight: .medium))
                .foregroundStyle(theme.text)
                .padding(.horizontal, 10)
                .padding(.vertical, 4)
                .frame(width: width)
                .background(theme.controlBg)
                .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                .overlay {
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .strokeBorder(theme.inputBorder, lineWidth: 0.5)
                }
                .contentShape(Rectangle())
            }
            .menuStyle(.borderlessButton)
            .accessibilityValue(selectedOption?.label ?? "")
        } else {
            Picker(titleKey, selection: $selection) {
                ForEach(options) { opt in
                    Text(opt.label)
                        .tag(opt.value)
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .frame(maxWidth: width)
        }
    }

    private var selectedOption: PopupOption<Value>? {
        options.first { $0.value == selection }
    }
}

#Preview("Popup") {
    @Previewable @State var host = "127.0.0.1"
    @Previewable @State var quant = "q4"

    VStack(alignment: .leading, spacing: 14) {
        Popup(selection: $host, width: 220, options: [
            ("127.0.0.1", "127.0.0.1 (Local only)"),
            ("0.0.0.0", "0.0.0.0 (IPv4 only)"),
            ("::", "0.0.0.0 & :: (All Networks)"),
            ("localhost", "localhost"),
        ])
        Popup(selection: $quant, width: 120, options: [
            ("auto", "Auto"), ("q4", "q4"), ("q5", "q5"), ("q6", "q6"), ("q8", "q8"), ("fp16", "fp16"),
        ])
    }
    .padding(24)
    .omlxThemed()
}

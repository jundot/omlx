// PR 3 — dropdown picker. Renders the native macOS menu picker so selects
// share one bezel, height, and font with the rest of the system controls.

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

    init(_ titleKey: LocalizedStringKey = "", selection: Binding<Value>, width: CGFloat? = nil, options: [PopupOption<Value>]) {
        self.titleKey = titleKey
        self._selection = selection
        self.options = options
        self.width = width
    }

    init(_ titleKey: LocalizedStringKey = "", selection: Binding<Value>, width: CGFloat? = nil, options: [(Value, String)]) {
        self.titleKey = titleKey
        self._selection = selection
        self.options = options.map { PopupOption(value: $0.0, label: $0.1) }
        self.width = width
    }

    var body: some View {
        Picker(titleKey, selection: $selection) {
            ForEach(options) { opt in
                Text(opt.label)
                    .tag(opt.value)
            }
        }
        .labelsHidden()
        .pickerStyle(.menu)
        // The native popup bezel hugs its label, and a bare `maxWidth` frame
        // would center it — leaving air on both sides. Pin it trailing so
        // select bezels end flush with the other controls in the column;
        // `width` stays the cap that keeps long labels from sprawling.
        .frame(maxWidth: width, alignment: .trailing)
    }
}

#Preview("Popup") {
    @Previewable @State var host = "127.0.0.1"
    @Previewable @State var quant = "q4"

    VStack(alignment: .leading, spacing: 14) {
        Popup(selection: $host, width: .controlMedium, options: [
            ("127.0.0.1", "127.0.0.1 (Local only)"),
            ("0.0.0.0", "0.0.0.0 (IPv4 only)"),
            ("::", "0.0.0.0 & :: (All Networks)"),
            ("localhost", "localhost"),
        ])
        Popup(selection: $quant, width: .controlCompact, options: [
            ("auto", "Auto"), ("q4", "q4"), ("q5", "q5"), ("q6", "q6"), ("q8", "q8"), ("fp16", "fp16"),
        ])
    }
    .padding(24)
    .omlxThemed()
}

// PR 3 — text field with focus ring matching the JSX `TextInput` / `BoxedInput`.

import SwiftUI

struct TextInput: View {
    @Binding var text: String
    var titleKey: LocalizedStringKey
    var placeholder: String
    var isSecure: Bool
    var mono: Bool
    var isNumeric: Bool
    var range: ClosedRange<Double>?
    var step: Double?
    var suffix: String?
    var width: CGFloat?

    @Environment(\.omlxTheme) private var theme

    init(_ titleKey: LocalizedStringKey = "", text: Binding<String>, placeholder: String = "", isSecure: Bool = false, mono: Bool = false, isNumeric: Bool = false, range: ClosedRange<Double>? = nil, step: Double? = nil, suffix: String? = nil, width: CGFloat? = nil) {
        self._text = text
        self.titleKey = titleKey
        self.placeholder = placeholder
        self.isSecure = isSecure
        self.mono = mono
        self.isNumeric = isNumeric
        self.range = range
        self.step = step
        self.suffix = suffix
        self.width = width
    }

    private var textAsDouble: Binding<Double> {
        Binding(
            get: { Double(text) ?? 0 },
            set: { text = $0.formatted(.number.grouping(.never)) }
        )
    }

    var body: some View {
        field
            .lineLimit(1)
            .textFieldStyle(.roundedBorder)
            .font(mono ? .omlxMono(13, weight: .medium) : .omlxText(13, weight: .medium))
            .foregroundStyle(theme.text)
            .overlay(alignment: .trailing) {
                HStack(spacing: 0) {
                    if let suffix {
                        Text(suffix)
                            .font(.omlxText(11))
                            .foregroundStyle(theme.textSecondary)
                            .padding(.horizontal, 10)
                    }
                    if isNumeric {
                        stepper
                    }
                }
            }
            .frame(maxWidth: width)
    }

    private var field: some View {
        Group {
            if isSecure {
                SecureField(titleKey, text: $text, prompt: Text(placeholder))
            } else {
                TextField(titleKey, text: $text, prompt: Text(placeholder))
            }
        }
    }

    private var stepper: some View {
        Group {
            switch (range, step) {
            case let (range?, step?):
                Stepper(titleKey, value: textAsDouble, in: range, step: step)
            case let (range?, nil):
                Stepper(titleKey, value: textAsDouble, in: range)
            default:
                Stepper(titleKey, value: textAsDouble)
            }
        }
        .labelsHidden()
        .controlSize(.small)
        .padding(.trailing, 1)
    }
}

#Preview("TextInput") {
    @Previewable @State var port = "8000"
    @Previewable @State var qty = "1024"
    @Previewable @State var temperature = "0.3"
    @Previewable @State var pwd = "sk-omlx-2k4j8"
    @Previewable @State var showKey = false
    @Previewable @State var alias = ""
    VStack(alignment: .leading, spacing: 14) {
        TextInput(text: $port, placeholder: "Port", mono: true, width: 110)
        TextInput(text: $qty, placeholder: "Quantity", isNumeric: true, suffix: "qty", width: 160)
        TextInput(text: $temperature, placeholder: "Temperature", isNumeric: true, range: 0...2, step: 0.1, width: 160)
        HStack {
            TextInput(text: $pwd, placeholder: "Admin password", isSecure: !showKey, width: 200)
            Button {
                showKey.toggle()
            } label: {
                Image(systemName: "eye")
                    .symbolVariant(showKey ? .slash : .none)
            }
            .buttonStyle(.plain)
        }
        TextInput(text: $alias, placeholder: "model-id-suffix", mono: true,
                  suffix: "alias", width: 240)
    }
    .padding(24)
    .omlxThemed()
}

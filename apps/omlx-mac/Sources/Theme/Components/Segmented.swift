// PR 3 — segmented control matching the JSX `Segmented`.

import SwiftUI

struct Segmented<Value: Hashable>: View {
    @Binding var selection: Value
    let options: [(value: Value, label: String)]

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        HStack(spacing: 0) {
            ForEach(options.indices, id: \.self) { i in
                let opt = options[i]
                let isSelected = opt.value == selection
                Button {
                    selection = opt.value
                } label: {
                    Text(opt.label)
                        .font(.omlxText(11.5, weight: .medium))
                        .foregroundStyle(isSelected ? theme.text : theme.textSecondary)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 4)
                        .frame(maxWidth: .infinity)
                        .background(
                            RoundedRectangle(cornerRadius: 5, style: .continuous)
                                .fill(isSelected ? theme.controlBg : Color.clear)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 5, style: .continuous)
                                        .strokeBorder(
                                            isSelected ? theme.inputBorder : Color.clear,
                                            lineWidth: 0.5
                                        )
                                )
                        )
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
        }
        .padding(2)
        .background(theme.inputBg)
        .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
    }
}

#Preview("Segmented") {
    @Previewable @State var mode = "local"
    @Previewable @State var scope = "session"
    return VStack(spacing: 14) {
        Segmented(selection: $mode, options: [
            ("cloud", "Cloud"), ("local", "Local"),
        ])
        .frame(width: 180)

        Segmented(selection: $scope, options: [
            ("session", "Session"), ("alltime", "All Time"),
        ])
        .frame(width: 180)
    }
    .padding(24)
    .omlxThemed()
}

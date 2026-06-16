// PR 3 — segmented control matching the JSX `Segmented`.

import SwiftUI

struct Segmented<Value: Hashable>: View {
    @Binding var selection: Value
    let titleKey: LocalizedStringKey = ""
    let options: [(value: Value, label: String)]

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        Picker(titleKey, selection: $selection) {
            ForEach(options, id: \.value) { opt in
                Text(opt.label)
                    .tag(opt.value)
            }
        }
        .pickerStyle(.segmented)
        .labelsHidden()
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

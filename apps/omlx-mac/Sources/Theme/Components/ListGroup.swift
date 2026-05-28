// PR 3 — rounded card that hosts a stack of `Row`s. Mirrors JSX `ListGroup`.

import SwiftUI

struct ListGroup<Content: View>: View {
    let content: () -> Content

    @Environment(\.omlxTheme) private var theme

    init(@ViewBuilder content: @escaping () -> Content) {
        self.content = content
    }

    var body: some View {
        VStack(spacing: 0) {
            content()
        }
        .background(theme.groupBg)
        .clipShape(RoundedRectangle(cornerRadius: theme.cornerRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: theme.cornerRadius, style: .continuous)
                .strokeBorder(theme.groupBorder, lineWidth: 0.5)
        )
        .padding(.horizontal, 14)
        .padding(.bottom, 6)
    }
}

#Preview("ListGroup") {
    ListGroup {
        Text("Row 1").frame(maxWidth: .infinity, alignment: .leading).padding(12)
        Divider()
        Text("Row 2").frame(maxWidth: .infinity, alignment: .leading).padding(12)
        Divider()
        Text("Row 3").frame(maxWidth: .infinity, alignment: .leading).padding(12)
    }
    .padding(.vertical, 14)
    .frame(width: 520)
    .omlxThemed()
}

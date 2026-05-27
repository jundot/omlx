// PR 3 — labeled row, the workhorse layout inside `ListGroup`. Mirrors JSX `Row`.
// Two variants:
//   • `Row(label:sublabel:isLast:trailing:)` — most uses
//   • `Row(isLast:content:)` — free-form (chip strips, custom layouts)

import SwiftUI

struct Row<Trailing: View>: View {
    let label: String?
    let sublabel: String?
    let isLast: Bool
    let trailing: Trailing

    @Environment(\.omlxTheme) private var theme

    init(
        label: String? = nil,
        sublabel: String? = nil,
        isLast: Bool = false,
        @ViewBuilder trailing: () -> Trailing
    ) {
        self.label = label
        self.sublabel = sublabel
        self.isLast = isLast
        self.trailing = trailing()
    }

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            if let label {
                VStack(alignment: .leading, spacing: 2) {
                    Text(label)
                        .font(.omlxText(13, weight: .medium))
                        .foregroundStyle(theme.text)
                    if let sublabel, !sublabel.isEmpty {
                        Text(sublabel)
                            .font(.omlxText(11.5))
                            .foregroundStyle(theme.textSecondary)
                            .lineLimit(2)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                Spacer(minLength: 12)
            }
            trailing
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .frame(minHeight: 36)
        .frame(maxWidth: .infinity, alignment: label == nil ? .leading : .center)
        .overlay(alignment: .bottom) {
            if !isLast {
                Rectangle()
                    .fill(theme.rowSep)
                    .frame(height: 0.5)
                    .padding(.leading, 14)
            }
        }
    }
}

extension Row where Trailing == EmptyView {
    /// Label-only row (no trailing slot).
    init(label: String, sublabel: String? = nil, isLast: Bool = false) {
        self.label = label
        self.sublabel = sublabel
        self.isLast = isLast
        self.trailing = EmptyView()
    }
}

/// Free-form variant — caller provides the entire row body. Used for chip
/// strips, multi-line custom rows, etc.
struct FreeRow<Content: View>: View {
    let isLast: Bool
    let content: () -> Content

    @Environment(\.omlxTheme) private var theme

    init(isLast: Bool = false, @ViewBuilder content: @escaping () -> Content) {
        self.isLast = isLast
        self.content = content
    }

    var body: some View {
        content()
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .overlay(alignment: .bottom) {
                if !isLast {
                    Rectangle()
                        .fill(theme.rowSep)
                        .frame(height: 0.5)
                        .padding(.leading, 14)
                }
            }
    }
}

#Preview("Row in ListGroup") {
    @Previewable @State var autoStart = true
    @Previewable @State var requireKey = false

    return ListGroup {
        Row(
            label: "Listen Address",
            sublabel: "Default 8080. Restart server to apply."
        ) {
            CodeChip(value: "127.0.0.1:8080")
        }
        Row(label: "Auto-start on launch") {
            Toggle("", isOn: $autoStart).labelsHidden().toggleStyle(.switch)
        }
        Row(
            label: "Require API Key",
            sublabel: "Reject unauthenticated /v1 requests",
            isLast: true
        ) {
            Toggle("", isOn: $requireKey).labelsHidden().toggleStyle(.switch)
        }
    }
    .padding(.vertical, 14)
    .frame(width: 560)
    .omlxThemed()
}

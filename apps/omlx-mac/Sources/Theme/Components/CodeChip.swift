// PR 3 — inline monospaced chip with click-to-copy. Used for endpoint URLs,
// shell commands, and aliases throughout the design.

import SwiftUI
import AppKit

struct CodeChip: View {
    let value: String
    var maxWidth: CGFloat? = nil

    @Environment(\.omlxTheme) private var theme
    @State private var copied = false

    var body: some View {
        Button(action: copy) {
            HStack(spacing: 6) {
                Text(value)
                    .font(.omlxMono(11.5, weight: .medium))
                    .foregroundStyle(theme.text)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Image(systemName: copied ? "checkmark" : "doc.on.doc")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(copied ? theme.successText : theme.textSecondary)
                    .animation(.easeOut(duration: 0.12), value: copied)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .frame(maxWidth: maxWidth, alignment: .leading)
            .background(theme.codeBg)
            .clipShape(RoundedRectangle(cornerRadius: 5, style: .continuous))
        }
        .buttonStyle(.plain)
        .help("Click to copy")
    }

    private func copy() {
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(value, forType: .string)
        copied = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.4) {
            copied = false
        }
    }
}

struct CopyIconButton: View {
    let value: String

    @Environment(\.omlxTheme) private var theme
    @State private var copied = false

    private var helpText: String {
        String(localized: "common.copy_model_id",
               defaultValue: "Copy model ID",
               comment: "Tooltip and accessibility label for a button that copies a model identifier")
    }

    var body: some View {
        Button(action: copy) {
            Image(systemName: copied ? "checkmark" : "doc.on.doc")
                .font(.system(size: 10.5, weight: .medium))
                .foregroundStyle(copied ? theme.successText : theme.textSecondary)
                .frame(width: 22, height: 22)
                .contentShape(Rectangle())
                .animation(.easeOut(duration: 0.12), value: copied)
        }
        .buttonStyle(.plain)
        .help(helpText)
        .accessibilityLabel(Text(helpText))
    }

    private func copy() {
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(value, forType: .string)
        copied = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.4) {
            copied = false
        }
    }
}

#Preview("CodeChip") {
    VStack(alignment: .leading, spacing: 10) {
        CodeChip(value: "http://127.0.0.1:8000/v1")
        CodeChip(value: "http://127.0.0.1:8000/health")
        CodeChip(value: "OPENAI_BASE_URL=http://127.0.0.1:8000/v1", maxWidth: 280)
        CopyIconButton(value: "mlx-community/Qwen3-8B-4bit")
    }
    .padding(24)
    .omlxThemed()
}

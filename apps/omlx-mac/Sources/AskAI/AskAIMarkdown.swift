// MarkdownUI theme for Ask AI answers: compact type for a floating panel,
// no page background so the glass shows through, and a copy button on
// every fenced block and block quote.

import MarkdownUI
import SwiftUI

extension Theme {
    @MainActor static let askAI = Theme.gitHub
        .text {
            ForegroundColor(Color(nsColor: .labelColor))
            FontSize(14)
        }
        .code {
            FontFamilyVariant(.monospaced)
            FontSize(.em(0.88))
            BackgroundColor(Color(nsColor: .labelColor).opacity(0.08))
        }
        .heading1 { configuration in
            configuration.label
                .markdownMargin(top: 16, bottom: 8)
                .markdownTextStyle {
                    FontWeight(.semibold)
                    FontSize(.em(1.4))
                }
        }
        .heading2 { configuration in
            configuration.label
                .markdownMargin(top: 14, bottom: 8)
                .markdownTextStyle {
                    FontWeight(.semibold)
                    FontSize(.em(1.25))
                }
        }
        .heading3 { configuration in
            configuration.label
                .markdownMargin(top: 12, bottom: 6)
                .markdownTextStyle {
                    FontWeight(.semibold)
                    FontSize(.em(1.1))
                }
        }
        .paragraph { configuration in
            configuration.label
                .fixedSize(horizontal: false, vertical: true)
                .relativeLineSpacing(.em(0.2))
                .markdownMargin(top: 0, bottom: 10)
        }
        .blockquote { configuration in
            AskAIBlockquote(configuration: configuration)
        }
        .codeBlock { configuration in
            AskAICodeBlock(configuration: configuration)
        }
        .table { configuration in
            configuration.label
                .fixedSize(horizontal: false, vertical: true)
                .markdownTableBorderStyle(.init(color: Color(nsColor: .separatorColor)))
                .markdownTableBackgroundStyle(
                    .alternatingRows(Color.clear, Color(nsColor: .labelColor).opacity(0.04))
                )
                .markdownMargin(top: 0, bottom: 10)
        }
}

private struct AskAICodeBlock: View {
    let configuration: CodeBlockConfiguration

    @Environment(\.omlxTheme) private var theme

    private var language: String {
        let lang = configuration.language?.trimmingCharacters(in: .whitespaces) ?? ""
        return lang.isEmpty ? "text" : lang
    }

    private var copyValue: String {
        var content = configuration.content
        while content.hasSuffix("\n") { content.removeLast() }
        return content
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 6) {
                Text(language)
                    .font(.omlxMono(DesignTokens.FontSize.subheadline))
                    .foregroundStyle(theme.textSecondary)
                Spacer(minLength: 0)
                CopyIconButton(value: copyValue, helpText: AskAIStrings.copyBlock)
            }
            .padding(.leading, 12)
            .padding(.trailing, 4)
            .padding(.vertical, 2)
            Rectangle()
                .fill(theme.separator)
                .frame(height: 0.5)
            ScrollView(.horizontal) {
                configuration.label
                    .fixedSize(horizontal: false, vertical: true)
                    .relativeLineSpacing(.em(0.225))
                    .markdownTextStyle {
                        FontFamilyVariant(.monospaced)
                        FontSize(.em(0.88))
                    }
                    .padding(12)
            }
        }
        .background(theme.codeBg)
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .markdownMargin(top: 0, bottom: 10)
    }
}

private struct AskAIBlockquote: View {
    let configuration: BlockConfiguration

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            RoundedRectangle(cornerRadius: 2)
                .fill(theme.separator)
                .frame(width: 3)
            configuration.label
                .markdownTextStyle { ForegroundColor(Color(nsColor: .secondaryLabelColor)) }
                .padding(.leading, 12)
            Spacer(minLength: 8)
            CopyIconButton(value: configuration.content.renderPlainText(),
                           helpText: AskAIStrings.copyBlock)
        }
        .fixedSize(horizontal: false, vertical: true)
        .markdownMargin(top: 0, bottom: 10)
    }
}

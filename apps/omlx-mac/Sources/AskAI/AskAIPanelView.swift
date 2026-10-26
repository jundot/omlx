// Content of the Ask AI panel: prompt field, preset suggestions, the
// conversation, and a footer with the target model and key hints.

import MarkdownUI
import SwiftUI

enum AskAIStrings {
    static var copyBlock: String {
        String(localized: "askai.copy.block", defaultValue: "Copy",
               comment: "Tooltip on the copy button of a code block or quote in an Ask AI answer")
    }
    static var copyAnswer: String {
        String(localized: "askai.copy.answer", defaultValue: "Copy answer",
               comment: "Tooltip on the button that copies a whole Ask AI answer")
    }
    static var copied: String {
        String(localized: "askai.copied", defaultValue: "Copied to clipboard",
               comment: "Notice after an Ask AI answer was copied")
    }
    static var openSettings: String {
        String(localized: "askai.open_settings", defaultValue: "Open Settings",
               comment: "Button in the Ask AI panel that opens the Ask AI settings")
    }
}

struct AskAIPanelView: View {
    @Bindable var controller: AskAIController
    let chrome: AskAIPanelChrome

    @Environment(\.omlxTheme) private var theme
    @State private var inputHeight: CGFloat = 22
    @State private var transcriptHeight: CGFloat = 0
    /// Panel height minus the transcript: input, attachments, notices,
    /// suggestions, and footer. The transcript gets what is left.
    @State private var chromeHeight: CGFloat = 100
    @State private var scrollPosition = ScrollPosition(edge: .bottom)
    /// Follows streamed text until the user scrolls up, then again once
    /// they scroll back to the bottom.
    @State private var followsBottom = true

    private var conversation: AskAIConversation { controller.conversation }

    var body: some View {
        let suggestions = controller.visibleSuggestions
        VStack(spacing: 0) {
            inputRow(suggestionsVisible: !suggestions.isEmpty)
            if !controller.attachments.isEmpty {
                AttachmentStrip(images: controller.attachments) { controller.removeAttachment($0) }
                    .padding(.horizontal, 16)
                    .padding(.bottom, 12)
            }
            if !suggestions.isEmpty {
                separator
                SuggestionList(presets: suggestions, selected: controller.selectedSuggestion) { index in
                    controller.acceptSuggestion(at: index)
                }
            }
            if let notice = controller.notice {
                separator
                noticeRow(notice)
            }
            if !conversation.isEmpty {
                separator
                transcript
            }
            separator
            footer
        }
        .frame(width: controller.panelWidth)
        .appGlass(.regular, in: RoundedRectangle(cornerRadius: AskAIPanelController.cornerRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: AskAIPanelController.cornerRadius, style: .continuous)
                .strokeBorder(theme.separator.opacity(0.6), lineWidth: 0.5)
        )
        .overlay(alignment: .leading) {
            AskAIPanelResizeHandle(edge: .leading, chrome: chrome).frame(width: 6)
        }
        .overlay(alignment: .trailing) {
            AskAIPanelResizeHandle(edge: .trailing, chrome: chrome).frame(width: 6)
        }
        .fixedSize(horizontal: false, vertical: true)
        .onGeometryChange(for: CGFloat.self) { $0.size.height } action: { total in
            chrome.heightChanged(total)
            let rest = total - (conversation.isEmpty ? 0 : transcriptFrameHeight)
            if abs(rest - chromeHeight) > 0.5 { chromeHeight = rest }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }

    private var separator: some View {
        Rectangle().fill(theme.separator).frame(height: 0.5)
    }

    // MARK: Input

    private func inputRow(suggestionsVisible: Bool) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "text.bubble")
                .font(.system(size: 16, weight: .medium))
                .foregroundStyle(theme.textSecondary)
                .frame(width: 24, height: 24)
                .overlay { AskAIPanelDragHandle(chrome: chrome) }
                .padding(.top, -1)
            ZStack(alignment: .topLeading) {
                if controller.draft.isEmpty {
                    Text(String(localized: "askai.input.placeholder",
                                defaultValue: "Ask anything, or type / for a preset",
                                comment: "Placeholder in the Ask AI prompt field"))
                        .font(.system(size: 15))
                        .foregroundStyle(theme.textTertiary)
                        .padding(.top, 2)
                        .allowsHitTesting(false)
                }
                AskAIInputView(
                    text: $controller.draft,
                    height: $inputHeight,
                    suggestionsVisible: suggestionsVisible,
                    focus: controller.inputFocus,
                    onCommand: controller.handle,
                    onPasteImages: controller.attachImages
                )
                .frame(height: inputHeight)
            }
            if conversation.isStreaming {
                Button {
                    controller.handle(.stop)
                } label: {
                    Image(systemName: "stop.circle.fill")
                        .font(.system(size: 16))
                        .foregroundStyle(theme.textSecondary)
                }
                .buttonStyle(.plain)
                .help(String(localized: "askai.stop.help", defaultValue: "Stop (⌘.)",
                             comment: "Tooltip on the stop button while an Ask AI answer streams"))
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
    }

    private func noticeRow(_ notice: AskAIController.Notice) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "exclamationmark.circle")
                .foregroundStyle(theme.warningText)
            Text(notice.text)
                .font(.omlxText(DesignTokens.FontSize.callout))
                .foregroundStyle(theme.text)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 8)
            if notice.offersSettings {
                Button(AskAIStrings.openSettings) { controller.openSettings() }
                    .buttonStyle(.omlx(.normal, size: .small))
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }

    // MARK: Transcript

    private var transcriptFrameHeight: CGFloat {
        min(transcriptHeight, max(controller.maxPanelHeight - chromeHeight, 80))
    }

    private var transcript: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                ForEach(conversation.turns) { turn in
                    TurnView(turn: turn, autoCopied: conversation.autoCopiedTurnID == turn.id)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .onGeometryChange(for: CGFloat.self) { $0.size.height } action: { transcriptHeight = $0 }
        }
        .scrollPosition($scrollPosition)
        .onScrollGeometryChange(for: ScrollGeometry.self) { $0 } action: { old, new in
            // Content or viewport size changed: growth, not a user scroll.
            if new.contentSize != old.contentSize || new.containerSize != old.containerSize {
                if followsBottom { scrollPosition.scrollTo(edge: .bottom) }
                return
            }
            if new.visibleRect.maxY >= new.contentSize.height - 24 {
                followsBottom = true
            } else if new.contentOffset.y < old.contentOffset.y - 0.5 {
                followsBottom = false
            }
        }
        .onChange(of: conversation.turns.count) {
            followsBottom = true
            scrollPosition.scrollTo(edge: .bottom)
        }
        .frame(height: transcriptFrameHeight)
        // Keeps the scroller clear of the edge resize strips.
        .padding(.horizontal, 6)
    }

    // MARK: Footer

    private var footer: some View {
        HStack(spacing: 12) {
            if let target = controller.pendingTarget {
                HStack(spacing: 6) {
                    if let command = target.command {
                        Text("/\(command)")
                            .font(.omlxMono(DesignTokens.FontSize.subheadline, weight: .medium))
                            .foregroundStyle(theme.accent)
                    }
                    Text(target.model)
                        .foregroundStyle(theme.textSecondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }
            Spacer(minLength: 8)
            if controller.showsCopied {
                Label(AskAIStrings.copied, systemImage: "checkmark")
                    .foregroundStyle(theme.successText)
            }
            Text(hintText)
                .foregroundStyle(theme.textTertiary)
            Button {
                controller.openSettings()
            } label: {
                Image(systemName: "gearshape")
            }
            .buttonStyle(.plain)
            .foregroundStyle(theme.textTertiary)
            .help(AskAIStrings.openSettings)
        }
        .font(.omlxText(DesignTokens.FontSize.subheadline))
        .padding(.horizontal, 16)
        .padding(.vertical, 7)
    }

    private var hintText: String {
        if conversation.isStreaming {
            return String(localized: "askai.hint.streaming", defaultValue: "Esc Stop",
                          comment: "Key hint in the Ask AI footer while an answer streams")
        }
        if conversation.isEmpty {
            return String(localized: "askai.hint.idle", defaultValue: "⌘↩ Send   Esc Close",
                          comment: "Key hints in the Ask AI footer before the first question")
        }
        return String(localized: "askai.hint.conversation",
                      defaultValue: "⌘↩ Follow up   ⌘N New   ⇧⌘C Copy",
                      comment: "Key hints in the Ask AI footer after an answer")
    }
}

// MARK: - Attachments

private struct AttachmentStrip: View {
    let images: [AskAIImage]
    let onRemove: (UUID) -> Void

    var body: some View {
        HStack(spacing: 8) {
            ForEach(images) { image in
                ImageThumbnail(image: image, size: 56)
                    .overlay(alignment: .topTrailing) {
                        Button {
                            onRemove(image.id)
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .font(.system(size: 14))
                                .symbolRenderingMode(.palette)
                                .foregroundStyle(.white, Color.black.opacity(0.6))
                        }
                        .buttonStyle(.plain)
                        .padding(3)
                        .help(String(localized: "askai.attachment.remove",
                                     defaultValue: "Remove image",
                                     comment: "Tooltip on the button that removes a pasted image"))
                    }
            }
            Spacer(minLength: 0)
        }
    }
}

private struct ImageThumbnail: View {
    let image: AskAIImage
    let size: CGFloat

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        Image(decorative: image.thumbnail, scale: 1)
            .resizable()
            .scaledToFill()
            .frame(width: size, height: size)
            .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .strokeBorder(theme.separator, lineWidth: 0.5)
            )
    }
}

// MARK: - Suggestions

private struct SuggestionList: View {
    let presets: [AskAIPreset]
    let selected: Int
    let onPick: (Int) -> Void

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(spacing: 2) {
            ForEach(Array(presets.enumerated()), id: \.element.id) { index, preset in
                HStack(spacing: 10) {
                    Text("/\(preset.command)")
                        .font(.omlxMono(DesignTokens.FontSize.body, weight: .medium))
                        .foregroundStyle(theme.text)
                    Text(preset.model)
                        .font(.omlxText(DesignTokens.FontSize.callout))
                        .foregroundStyle(theme.textSecondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Text(preset.promptPreview)
                        .font(.omlxText(DesignTokens.FontSize.callout))
                        .foregroundStyle(theme.textTertiary)
                        .lineLimit(1)
                    Spacer(minLength: 0)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background(index == selected ? theme.accentSoft : .clear)
                .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                .contentShape(Rectangle())
                .onTapGesture { onPick(index) }
            }
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 6)
    }
}

// MARK: - Turn

private struct TurnView: View {
    let turn: AskAIConversation.Turn
    let autoCopied: Bool

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !turn.images.isEmpty {
                HStack(spacing: 6) {
                    ForEach(turn.images) { image in
                        ImageThumbnail(image: image, size: 40)
                    }
                }
            }
            if !turn.question.isEmpty {
                Text(turn.question)
                    .font(.omlxText(DesignTokens.FontSize.body))
                    .foregroundStyle(theme.textSecondary)
                    .lineLimit(4)
                    .textSelection(.enabled)
            }

            if !turn.reasoning.isEmpty {
                ThinkingView(text: turn.reasoning, isThinking: turn.isStreaming && turn.answer.isEmpty)
            }

            if !turn.answer.isEmpty {
                Markdown(turn.answer)
                    .markdownTheme(.askAI)
                    .textSelection(.enabled)
            } else if turn.isStreaming && turn.reasoning.isEmpty {
                ProgressView()
                    .controlSize(.small)
            }

            if let error = turn.error {
                // Some server errors append the full model list.
                Label(error, systemImage: "exclamationmark.triangle.fill")
                    .font(.omlxText(DesignTokens.FontSize.callout))
                    .foregroundStyle(theme.redDot)
                    .lineLimit(3)
                    .help(error)
            }

            if !turn.isStreaming && !turn.answer.isEmpty {
                HStack(spacing: 8) {
                    if turn.wasStopped {
                        Text(String(localized: "askai.turn.stopped", defaultValue: "Stopped",
                                    comment: "Marker under an Ask AI answer that was stopped early"))
                            .foregroundStyle(theme.textTertiary)
                    }
                    if autoCopied {
                        Label(AskAIStrings.copied, systemImage: "checkmark")
                            .foregroundStyle(theme.successText)
                    }
                    Spacer(minLength: 0)
                    CopyIconButton(
                        value: turn.answer.trimmingCharacters(in: .whitespacesAndNewlines),
                        helpText: AskAIStrings.copyAnswer
                    )
                }
                .font(.omlxText(DesignTokens.FontSize.subheadline))
            }
        }
    }
}

private struct ThinkingView: View {
    let text: String
    let isThinking: Bool

    @State private var expanded = false
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        DisclosureGroup(isExpanded: $expanded) {
            Text(text)
                .font(.omlxText(DesignTokens.FontSize.callout))
                .foregroundStyle(theme.textSecondary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, 4)
        } label: {
            HStack(spacing: 6) {
                Text(isThinking
                     ? String(localized: "askai.thinking.active", defaultValue: "Thinking…",
                              comment: "Label while the model is still reasoning")
                     : String(localized: "askai.thinking.done", defaultValue: "Thought process",
                              comment: "Label of the collapsed reasoning section after the answer"))
                if isThinking {
                    ProgressView().controlSize(.mini)
                }
            }
            .font(.omlxText(DesignTokens.FontSize.callout, weight: .medium))
            .foregroundStyle(theme.textSecondary)
        }
    }
}

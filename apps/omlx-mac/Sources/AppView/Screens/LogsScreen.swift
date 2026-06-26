// PR 7 — Logs screen. Tails `/admin/api/logs` and renders the result in a
// monospaced ScrollView with a Lines popup, file selector, refresh button
// and copy-all. The view auto-refreshes every 5 s while visible.

import SwiftUI
import AppKit

struct LogsScreen: View {
    @Environment(AppServices.self) private var services
    @State private var vm = LogsScreenVM()

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            SectionHeader(String(localized: "logs.section.title",
                                  defaultValue: "Server Logs",
                                  comment: "Section header above the log tail pane on the Logs screen"),
                          subtitle: vm.subtitle) {
                Button(String(localized: "common.copy",
                              defaultValue: "Copy",
                              comment: "Button label to copy the visible log text to the pasteboard")) {
                    vm.copyToPasteboard()
                }
                              .buttonStyle(.omlx(.normal, size: .small))
                              .disabled(vm.lines == 0 || vm.logText.isEmpty)
            }

            if vm.availableFiles.count > 1 {
                ListGroup {
                    Row(label: String(localized: "logs.row.file.label",
                                      defaultValue: "Log file",
                                      comment: "Row label for the log file selector popup on the Logs screen"),
                        isLast: true) {
                        Popup(
                            selection: $vm.selectedFile,
                            width: 220,
                            options: vm.fileOptions
                        )
                    }
                }
            }

            LogPane(text: vm.logText, isEmpty: vm.logText.isEmpty, isLoading: vm.isLoading)
                .padding(.horizontal, 14)
                .padding(.top, vm.availableFiles.count > 1 ? 0 : 6)
                .padding(.bottom, 8)
                .frame(maxWidth: .infinity, maxHeight: .infinity)

            if let error = vm.lastError {
                Text(error)
                    .font(.omlxText(11))
                    .foregroundStyle(.red)
                    .padding(.horizontal, 18)
                    .padding(.top, 4)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                logLinesPicker
                reloadButton
            }
        }
        .task(id: vm.refreshKey) {
            await vm.start(client: services.client)
        }
        .onChange(of: vm.lines) { _, _ in vm.bumpRefreshKey() }
        .onChange(of: vm.selectedFile) { _, _ in vm.bumpRefreshKey() }
        .onDisappear { vm.stop() }
    }

    @ViewBuilder
    private var logLinesPicker: some View {
        let lineOptions = [
            (100,
             String(localized: "logs.lines.100",
                    defaultValue: "Last 100",
                    comment: "Popup option to show the most recent 100 log lines")),
            (500,
             String(localized: "logs.lines.500",
                    defaultValue: "Last 500",
                    comment: "Popup option to show the most recent 500 log lines")),
            (1000,
             String(localized: "logs.lines.1000",
                    defaultValue: "Last 1,000",
                    comment: "Popup option to show the most recent 1,000 log lines")),
            (5000,
             String(localized: "logs.lines.5000",
                    defaultValue: "Last 5,000",
                    comment: "Popup option to show the most recent 5,000 log lines")),
            (20000,
             String(localized: "logs.lines.20000",
                    defaultValue: "Last 20,000",
                    comment: "Popup option to show the most recent 20,000 log lines")),
        ]

        Popup(
            selection: $vm.lines,
            width: 110,
            options: lineOptions
        )
    }

    @ViewBuilder
    private var reloadButton: some View {
        Button {
            Task { await vm.reload() }
        } label: {
            Image(systemName: "arrow.clockwise")
        }
        .disabled(vm.isLoading)
    }
}

// MARK: - Log pane

private struct LogPane: View {
    let text: String
    let isEmpty: Bool
    let isLoading: Bool

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        ZStack {
            // NSTextView handles tens of thousands of lines effortlessly,
            // unlike SwiftUI's Text which lays out the entire string up
            // front on every render. The bridge below keeps selection,
            // smooth scrolling, and Cmd+A intact.
            LogTextView(
                text: text,
                fgColor: NSColor(theme.text),
                bgColor: NSColor(theme.codeBg)
            )
            if isEmpty && !isLoading {
                Text(String(localized: "logs.empty",
                            defaultValue: "No log entries.",
                            comment: "Empty-state text shown inside the log pane when the server has no log entries"))
                    .font(.omlxText(12))
                    .foregroundStyle(theme.textTertiary)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(.vertical, 36)
            }
        }
        .frame(minHeight: 360, idealHeight: 480, maxHeight: .infinity)
        .background(theme.codeBg)
        .clipShape(RoundedRectangle(cornerRadius: theme.cornerRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: theme.cornerRadius, style: .continuous)
                .strokeBorder(theme.groupBorder, lineWidth: 0.5)
        )
    }
}

/// AppKit-backed monospaced text view wrapped in a scroll view. Reads its
/// content from `text` and follows the bottom only when the user is already
/// pinned there — preserving the scroll position when they've scrolled up
/// to inspect a specific entry.
private struct LogTextView: NSViewRepresentable {
    let text: String
    let fgColor: NSColor
    let bgColor: NSColor

    func makeNSView(context: Context) -> NSScrollView {
        let scroll = NSTextView.scrollableTextView()
        scroll.hasVerticalScroller = true
        scroll.hasHorizontalScroller = false
        scroll.autohidesScrollers = true
        scroll.borderType = .noBorder
        scroll.drawsBackground = false

        guard let textView = scroll.documentView as? NSTextView else {
            return scroll
        }
        textView.isEditable = false
        textView.isSelectable = true
        textView.isRichText = false
        textView.allowsUndo = false
        textView.usesFindBar = true
        textView.isIncrementalSearchingEnabled = true
        textView.drawsBackground = true
        textView.backgroundColor = bgColor
        textView.textContainerInset = NSSize(width: 12, height: 10)
        textView.font = NSFont.monospacedSystemFont(ofSize: 11.5, weight: .regular)
        textView.textColor = fgColor
        textView.isAutomaticQuoteSubstitutionEnabled = false
        textView.isAutomaticDashSubstitutionEnabled = false
        textView.isAutomaticTextReplacementEnabled = false
        textView.isAutomaticSpellingCorrectionEnabled = false
        // Log lines are often wider than the view; soft-wrap to width so
        // there's no horizontal scrollbar but long messages still readable.
        if let container = textView.textContainer {
            container.widthTracksTextView = true
            container.lineFragmentPadding = 0
        }
        textView.string = text
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {
        guard let textView = scroll.documentView as? NSTextView else { return }

        textView.backgroundColor = bgColor
        textView.textColor = fgColor
        if textView.string == text { return }

        // Decide whether to follow the tail before mutating the string.
        let wasPinnedToBottom: Bool = {
            let clip = scroll.contentView
            let visibleMaxY = clip.documentVisibleRect.maxY
            let documentMaxY = clip.documentRect.maxY
            // 40 pt slack — keeps "follow the tail" feel even if the user
            // nudged the scrollwheel a hair.
            return documentMaxY - visibleMaxY < 40
        }()

        textView.string = text

        if wasPinnedToBottom {
            DispatchQueue.main.async {
                textView.scrollToEndOfDocument(nil)
            }
        }
    }
}

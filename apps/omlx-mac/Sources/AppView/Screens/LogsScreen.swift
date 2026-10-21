// Logs screen. Same viewer as the web UI: Table lists one row per record with
// a detail card for the selected row, Raw shows the text as served.

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

            LogControls(vm: vm)

            Group {
                switch vm.viewMode {
                case .table: LogTable(vm: vm)
                case .raw: LogRawPane(vm: vm)
                }
            }
            .frame(minHeight: 360, idealHeight: 480, maxHeight: .infinity)
            .padding(.horizontal, 14)
            .padding(.bottom, 8)

            FooterBar(error: vm.lastError)
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
            width: .controlCompact,
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

// MARK: - Controls

private struct LogControls: View {
    @Bindable var vm: LogsScreenVM

    var body: some View {
        ListGroup {
            if vm.availableFiles.count > 1 {
                Row(label: String(localized: "logs.row.file.label",
                                  defaultValue: "Log file",
                                  comment: "Row label for the log file selector popup on the Logs screen")) {
                    Popup(selection: $vm.selectedFile, width: .controlMedium, options: vm.fileOptions)
                }
            }

            Row(label: String(localized: "logs.row.view.label",
                              defaultValue: "View",
                              comment: "Row label for the control that switches the Logs screen between the table and the raw text")) {
                Segmented(
                    selection: $vm.viewMode,
                    titleKey: "logs.row.view.label",
                    options: [
                        (value: LogViewMode.table,
                         label: String(localized: "logs.view.table",
                                       defaultValue: "Table",
                                       comment: "Logs screen view option: one row per log record")),
                        (value: LogViewMode.raw,
                         label: String(localized: "logs.view.raw",
                                       defaultValue: "Raw",
                                       comment: "Logs screen view option: the log text as the server sent it")),
                    ]
                )
                .frame(width: 160)
            }

            Row(label: String(localized: "logs.row.level.label",
                              defaultValue: "Minimum level",
                              comment: "Row label for the segmented level filter on the Logs screen")) {
                Segmented(
                    selection: $vm.minLevel,
                    titleKey: "logs.row.level.label",
                    options: LogLevel.allCases.map { (value: $0, label: $0.rawValue) }
                )
                .help(String(localized: "logs.row.level.help",
                             defaultValue: "Filters this view only. It does not change the level the log file saves.",
                             comment: "Tooltip on the log level filter explaining what it does"))
            }

            Row(label: String(localized: "logs.row.autoscroll.label",
                              defaultValue: "Auto-scroll",
                              comment: "Row label for the switch that follows the newest log line"),
                isLast: true) {
                Toggle("", isOn: $vm.autoScroll)
                    .toggleStyle(.switch)
                    .labelsHidden()
            }
        }
    }
}

// MARK: - Table

private enum LogColumn {
    static let time: CGFloat = 92
    static let level: CGFloat = 72
    static let module: CGFloat = 150
    static let rowHeight: CGFloat = 28
}

private struct LogTable: View {
    @Bindable var vm: LogsScreenVM

    @Environment(AppServices.self) private var services
    @Environment(\.omlxTheme) private var theme
    @State private var position = ScrollPosition(idType: Int.self)
    /// Detail card height, kept across selections and launches.
    @AppStorage("logs.detailHeight") private var savedDetailHeight: Double = 220
    /// Live height during a drag; saved once the drag ends.
    @State private var dragHeight: Double?
    @State private var dragStartHeight: Double?

    private static let handleHeight: CGFloat = 12
    private static let minDetailHeight: Double = 120
    private static let minListHeight: Double = 160

    var body: some View {
        GeometryReader { geometry in
            VStack(spacing: 0) {
                if let row = vm.selectedRow {
                    LogDetailCard(row: row, onOpen: { services.requestedSection = $0 },
                                  onClose: { vm.selectedRowID = nil })
                        .frame(height: clampedDetailHeight(dragHeight ?? savedDetailHeight,
                                                           total: geometry.size.height))
                    SplitHandle()
                        .frame(height: Self.handleHeight)
                        .gesture(
                            // Global space: the handle moves with the card, so
                            // a local translation would shrink as it moves.
                            DragGesture(minimumDistance: 1, coordinateSpace: .global)
                                .onChanged { value in
                                    let start = dragStartHeight ?? clampedDetailHeight(
                                        savedDetailHeight, total: geometry.size.height)
                                    dragStartHeight = start
                                    dragHeight = clampedDetailHeight(
                                        start + value.translation.height,
                                        total: geometry.size.height)
                                }
                                .onEnded { _ in
                                    if let dragHeight { savedDetailHeight = dragHeight }
                                    dragHeight = nil
                                    dragStartHeight = nil
                                }
                        )
                }
                listPane
            }
        }
    }

    private func clampedDetailHeight(_ height: Double, total: CGFloat) -> Double {
        let upper = max(Self.minDetailHeight,
                        Double(total - Self.handleHeight) - Self.minListHeight)
        return min(max(height, Self.minDetailHeight), upper)
    }

    private var listPane: some View {
        VStack(spacing: 0) {
            LogColumnHeader()
            Divider().overlay(theme.groupBorder)
            if vm.rows.isEmpty {
                if !vm.isLoading {
                    Text(vm.emptyMessage)
                        .font(.omlxText(DesignTokens.FontSize.body))
                        .foregroundStyle(theme.textTertiary)
                        .padding(.vertical, 36)
                }
                Spacer(minLength: 0)
            } else {
                rowList
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(theme.codeBg)
        .clipShape(RoundedRectangle(cornerRadius: theme.cornerRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: theme.cornerRadius, style: .continuous)
                .strokeBorder(theme.groupBorder, lineWidth: 0.5)
        )
    }

    private var rowList: some View {
        ScrollView {
            LazyVStack(spacing: 0) {
                ForEach(vm.rows) { row in
                    LogRowView(row: row, isSelected: row.id == vm.selectedRowID) {
                        vm.toggleSelection(row)
                    }
                    Divider().overlay(theme.groupBorder)
                }
            }
            .scrollTargetLayout()
        }
        // The position tracks the top row, so a refresh that drops lines off
        // the front keeps the reader's place while auto-scroll is off.
        .scrollPosition($position, anchor: .top)
        .onAppear { scrollToEdge() }
        .onChange(of: vm.rows.last?.id) { _, _ in
            if vm.autoScroll { position.scrollTo(edge: .bottom) }
        }
        .onChange(of: vm.minLevel) { _, _ in scrollToEdge() }
    }

    private func scrollToEdge() {
        position.scrollTo(edge: vm.autoScroll ? .bottom : .top)
    }
}

/// Drag area between the detail card and the list. The capsule marks it, and
/// the pointer turns into the row-resize arrows.
private struct SplitHandle: View {
    @Environment(\.omlxTheme) private var theme
    @State private var hovering = false

    var body: some View {
        Capsule()
            .fill(theme.textTertiary.opacity(hovering ? 0.9 : 0.45))
            .frame(width: 36, height: 4)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .contentShape(Rectangle())
            .onHover { hovering = $0 }
            .pointerStyle(.rowResize)
            .accessibilityLabel(String(localized: "logs.detail.resize",
                                       defaultValue: "Resize log entry",
                                       comment: "Accessibility label of the handle that resizes the log detail card"))
    }
}

private struct LogColumnHeader: View {
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        HStack(spacing: 8) {
            Text(String(localized: "logs.col.time", defaultValue: "Time",
                        comment: "Logs table column header: when the line was written"))
                .frame(width: LogColumn.time, alignment: .leading)
            Text(String(localized: "logs.col.level", defaultValue: "Level",
                        comment: "Logs table column header: log level"))
                .frame(width: LogColumn.level, alignment: .leading)
            Text(String(localized: "logs.col.module", defaultValue: "Module",
                        comment: "Logs table column header: the logger name"))
                .frame(width: LogColumn.module, alignment: .leading)
            Text(String(localized: "logs.col.message", defaultValue: "Message",
                        comment: "Logs table column header: the log message"))
            Spacer(minLength: 0)
        }
        .font(.omlxText(DesignTokens.FontSize.callout, weight: .semibold))
        .foregroundStyle(theme.textTertiary)
        .lineLimit(1)
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }
}

private struct LogRowView: View {
    let row: LogRow
    let isSelected: Bool
    let onSelect: () -> Void

    @Environment(\.omlxTheme) private var theme

    private var record: LogRecord { row.record }

    var body: some View {
        HStack(spacing: 8) {
            Text(Self.clockTime(record.time))
                .font(.omlxMono(DesignTokens.FontSize.callout))
                .foregroundStyle(theme.textTertiary)
                .frame(width: LogColumn.time, alignment: .leading)
                .help(record.level == nil
                      ? String(localized: "logs.row.fragment.help",
                               defaultValue: "The window starts inside this record",
                               comment: "Tooltip on a log row whose header line is above the fetched window")
                      : record.time)
            LevelBadge(level: record.level)
                .frame(width: LogColumn.level, alignment: .leading)
            Text(record.module)
                .font(.omlxMono(DesignTokens.FontSize.callout))
                .foregroundStyle(theme.textSecondary)
                .truncationMode(.middle)
                .frame(width: LogColumn.module, alignment: .leading)
            trailing
        }
        .lineLimit(1)
        .padding(.horizontal, 12)
        .frame(height: LogColumn.rowHeight)
        .background(isSelected ? theme.selBg : Color.clear)
        .contentShape(Rectangle())
        .onTapGesture(perform: onSelect)
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
    }

    /// The message and its badges. Memory chips show only when they fit; the
    /// detail card always has them.
    @ViewBuilder
    private var trailing: some View {
        if let memory = LogParser.memoryGuard(String(record.firstLine)) {
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 8) {
                    message.frame(idealWidth: 120, maxWidth: .infinity, alignment: .leading)
                    badges
                    LogMemoryChips(memory: memory)
                }
                messageAndBadges
            }
        } else {
            messageAndBadges
        }
    }

    private var messageAndBadges: some View {
        HStack(spacing: 8) {
            message
            badges
            Spacer(minLength: 0)
        }
    }

    private var message: some View {
        Text(record.firstLine)
            .font(.omlxText(DesignTokens.FontSize.body))
            .foregroundStyle(theme.text)
            .truncationMode(.tail)
    }

    @ViewBuilder
    private var badges: some View {
        if row.times.count > 1 {
            CountBadge(text: "×\(row.times.count)", tint: Color(nsColor: .systemOrange))
        }
        if record.lineCount > 1 {
            CountBadge(text: "≡\(record.lineCount)", tint: nil)
        }
    }

    /// `yyyy-MM-dd HH:mm:ss,SSS` -> `HH:mm:ss.SSS`; the detail card has the date.
    private static func clockTime(_ stamp: String) -> String {
        guard stamp.count == 23 else { return "-" }
        return String(stamp.dropFirst(11)).replacingOccurrences(of: ",", with: ".")
    }
}

private struct LevelBadge: View {
    let level: LogLevel?

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        let color = tint
        Text(level?.rawValue ?? "-")
            .font(.omlxMono(DesignTokens.FontSize.callout, weight: .semibold))
            .foregroundStyle(color)
            .padding(.horizontal, 5)
            .padding(.vertical, 1)
            .background(color.opacity(0.14), in: RoundedRectangle(cornerRadius: 4, style: .continuous))
    }

    private var tint: Color {
        switch level {
        case .critical: Color(nsColor: .systemPurple)
        case .error: Color(nsColor: .systemRed)
        case .warning: Color(nsColor: .systemOrange)
        case .info: Color(nsColor: .systemBlue)
        case .debug: Color(nsColor: .systemTeal)
        case .trace, nil: theme.textTertiary
        }
    }
}

private struct CountBadge: View {
    let text: String
    /// Nil draws the muted style.
    let tint: Color?

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        Text(text)
            .font(.omlxText(DesignTokens.FontSize.callout, weight: .semibold))
            .foregroundStyle(tint ?? theme.textTertiary)
            .padding(.horizontal, 4)
            .background((tint?.opacity(0.16) ?? theme.controlBg),
                        in: RoundedRectangle(cornerRadius: 5, style: .continuous))
            .fixedSize()
    }
}

private struct LogMemoryChips: View {
    let memory: [LogMemoryField: String]

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        HStack(spacing: 4) {
            ForEach(LogMemoryField.allCases.filter { memory[$0] != nil }, id: \.self) { field in
                HStack(spacing: 3) {
                    Text(field.label).foregroundStyle(theme.textTertiary)
                    Text("\(memory[field] ?? "") GB").fontWeight(.semibold)
                }
                .padding(.horizontal, 4)
                .background(theme.controlBg, in: RoundedRectangle(cornerRadius: 5, style: .continuous))
            }
        }
        .font(.omlxText(DesignTokens.FontSize.callout))
        .fixedSize()
    }
}

private extension LogMemoryField {
    var label: String {
        switch self {
        case .usage:
            String(localized: "logs.memory.usage", defaultValue: "Usage",
                   comment: "Chip label on a memory-guard log row: process memory in use")
        case .watermark:
            String(localized: "logs.memory.watermark", defaultValue: "Hard watermark",
                   comment: "Chip label on a memory-guard log row: the abort threshold")
        case .ceiling:
            String(localized: "logs.memory.ceiling", defaultValue: "Ceiling",
                   comment: "Chip label on a memory-guard log row: the memory ceiling")
        case .peak:
            String(localized: "logs.memory.peak", defaultValue: "Predicted peak",
                   comment: "Chip label on a memory-guard log row: the predicted prefill peak")
        }
    }
}

// MARK: - Detail card

private struct LogDetailCard: View {
    let row: LogRow
    let onOpen: (AppSection) -> Void
    let onClose: () -> Void

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Text(String(localized: "logs.detail.title",
                            defaultValue: "Entry",
                            comment: "Title of the card that shows one log record in full"))
                    .font(.omlxText(DesignTokens.FontSize.callout, weight: .semibold))
                    .foregroundStyle(theme.textSecondary)
                Text(row.record.time)
                    .font(.omlxMono(DesignTokens.FontSize.callout))
                    .foregroundStyle(theme.textSecondary)
                LevelBadge(level: row.record.level)
                Text(row.record.module)
                    .font(.omlxMono(DesignTokens.FontSize.callout))
                    .foregroundStyle(theme.textTertiary)
                    .lineLimit(1)
                Spacer(minLength: 8)
                Button(String(localized: "logs.detail.close",
                              defaultValue: "Close",
                              comment: "Button that closes the log record detail card")) {
                    onClose()
                }
                .buttonStyle(.omlx(.plain, size: .small))
                .keyboardShortcut(.escape, modifiers: [])
            }

            // Fills the card, so dragging the split divider shows more lines.
            LogTextView(text: row.record.message, followTail: false, edgeKey: String(row.id),
                        fgColor: NSColor(theme.text), bgColor: NSColor(theme.inputBg))
                .frame(minHeight: 40, maxHeight: .infinity)
                .clipShape(RoundedRectangle(cornerRadius: theme.rowRadius, style: .continuous))

            if let memory = LogParser.memoryGuard(String(row.record.firstLine)) {
                HStack(spacing: 8) {
                    LogMemoryChips(memory: memory)
                    Button(String(localized: "logs.action.raise_tier",
                                  defaultValue: "Raise memory_guard_tier",
                                  comment: "Button in the log detail card that opens the Performance settings")) {
                        onOpen(.performance)
                    }
                    Button(String(localized: "logs.action.reduce_context",
                                  defaultValue: "Reduce context",
                                  comment: "Button in the log detail card that opens the Server settings with the default context window")) {
                        onOpen(.server)
                    }
                }
                .font(.omlxText(DesignTokens.FontSize.callout))
                .buttonStyle(.link)
            }

            if row.times.count > 1 {
                Text(String(localized: "logs.detail.occurrences",
                            defaultValue: "Occurrences (\(row.times.count))",
                            comment: "Heading above the list of times a repeated log line appeared; the placeholder is how many times"))
                    .font(.omlxText(DesignTokens.FontSize.callout, weight: .medium))
                    .foregroundStyle(theme.textSecondary)
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 3) {
                        ForEach(Array(row.times.prefix(LogParser.occurrenceLimit).enumerated()),
                                id: \.offset) { _, time in
                            Text(time)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 96)
                .font(.omlxMono(DesignTokens.FontSize.callout))
                .foregroundStyle(theme.textTertiary)
                if row.times.count > LogParser.occurrenceLimit {
                    Text(String(localized: "logs.detail.occurrences_more",
                                defaultValue: "…and \(row.times.count - LogParser.occurrenceLimit) more",
                                comment: "Line under the occurrence list when the list is capped; the placeholder is how many occurrences are not listed"))
                        .font(.omlxMono(DesignTokens.FontSize.callout))
                        .foregroundStyle(theme.textTertiary)
                }
            }
        }
        .padding(10)
        .background(theme.groupBg)
        .clipShape(RoundedRectangle(cornerRadius: theme.cornerRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: theme.cornerRadius, style: .continuous)
                .strokeBorder(theme.groupBorder, lineWidth: 0.5)
        )
    }
}

// MARK: - Raw

private struct LogRawPane: View {
    @Bindable var vm: LogsScreenVM

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        ZStack {
            LogTextView(text: vm.rawText, followTail: vm.autoScroll, edgeKey: vm.minLevel.rawValue,
                        fgColor: NSColor(theme.text), bgColor: NSColor(theme.codeBg))
            if vm.rawText.isEmpty && !vm.isLoading {
                Text(vm.emptyMessage)
                    .font(.omlxText(DesignTokens.FontSize.body))
                    .foregroundStyle(theme.textTertiary)
                    .padding(.vertical, 36)
            }
        }
        .background(theme.codeBg)
        .clipShape(RoundedRectangle(cornerRadius: theme.cornerRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: theme.cornerRadius, style: .continuous)
                .strokeBorder(theme.groupBorder, lineWidth: 0.5)
        )
    }
}

/// AppKit-backed monospaced text view: NSTextView lays out tens of thousands
/// of lines cheaply and keeps selection and the Find bar. With `followTail` a
/// text change scrolls to the end; a change of `edgeKey` scrolls to the end
/// with `followTail` and to the top without it.
private struct LogTextView: NSViewRepresentable {
    let text: String
    let followTail: Bool
    let edgeKey: String
    let fgColor: NSColor
    let bgColor: NSColor

    final class Coordinator {
        var edgeKey = ""
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

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
        textView.font = NSFont.monospacedSystemFont(ofSize: 12, weight: .regular)
        textView.textColor = fgColor
        textView.isAutomaticQuoteSubstitutionEnabled = false
        textView.isAutomaticDashSubstitutionEnabled = false
        textView.isAutomaticTextReplacementEnabled = false
        textView.isAutomaticSpellingCorrectionEnabled = false
        // Log lines are often wider than the view; soft-wrap them to its width.
        if let container = textView.textContainer {
            container.widthTracksTextView = true
            container.lineFragmentPadding = 0
        }
        textView.string = text
        context.coordinator.edgeKey = edgeKey
        if followTail {
            DispatchQueue.main.async { textView.scrollToEndOfDocument(nil) }
        }
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {
        guard let textView = scroll.documentView as? NSTextView else { return }

        textView.backgroundColor = bgColor
        textView.textColor = fgColor
        let edgeChanged = context.coordinator.edgeKey != edgeKey
        context.coordinator.edgeKey = edgeKey
        if textView.string != text {
            textView.string = text
        } else if !edgeChanged {
            return
        }

        if followTail {
            DispatchQueue.main.async { textView.scrollToEndOfDocument(nil) }
        } else if edgeChanged {
            DispatchQueue.main.async { textView.scrollToBeginningOfDocument(nil) }
        }
    }
}

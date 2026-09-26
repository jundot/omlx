// Logs screen. The web console's log viewer, in the app's own dress: a filter
// row (minimum level + auto-scroll), a list of records with fixed columns, and
// a detail card for the row you click — which is where a repeated line shows
// every occurrence behind its ×N badge.
//
// The list is records, not raw text: `LogRecords.swift` owns the parsing and
// the aggregation, `LogsScreenVM` owns the tail and the selection.

import SwiftUI
import AppKit

struct LogsScreen: View {
    @Environment(AppServices.self) private var services
    @Environment(\.omlxTheme) private var theme
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

            LogPane(vm: vm)
                .padding(.horizontal, 14)
                .padding(.bottom, 8)
                .frame(maxWidth: .infinity, maxHeight: .infinity)

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
        Popup(
            selection: $vm.lines,
            width: .controlCompact,
            options: [
                (100, String(localized: "logs.lines.100", defaultValue: "Last 100",
                             comment: "Popup option to show the most recent 100 log lines")),
                (500, String(localized: "logs.lines.500", defaultValue: "Last 500",
                             comment: "Popup option to show the most recent 500 log lines")),
                (1000, String(localized: "logs.lines.1000", defaultValue: "Last 1,000",
                              comment: "Popup option to show the most recent 1,000 log lines")),
                (5000, String(localized: "logs.lines.5000", defaultValue: "Last 5,000",
                              comment: "Popup option to show the most recent 5,000 log lines")),
                (20000, String(localized: "logs.lines.20000", defaultValue: "Last 20,000",
                               comment: "Popup option to show the most recent 20,000 log lines")),
            ]
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

/// The filter row: the six severities (the web console puts the same six above
/// its list), the file selector, and auto-scroll.
private struct LogControls: View {
    @Bindable var vm: LogsScreenVM

    private var levels: [LogLevel] { [.trace, .debug, .info, .warning, .error, .critical] }

    var body: some View {
        ListGroup {
            if vm.availableFiles.count > 1 {
                Row(label: String(localized: "logs.row.file.label",
                                  defaultValue: "Log file",
                                  comment: "Row label for the log file selector popup on the Logs screen")) {
                    Popup(selection: $vm.selectedFile, width: .controlMedium, options: vm.fileOptions)
                }
            }

            Row(label: String(localized: "logs.row.level.label",
                              defaultValue: "Minimum level",
                              comment: "Row label for the segmented level filter on the Logs screen")) {
                Segmented(
                    selection: $vm.minLevel,
                    titleKey: "logs.row.level.label",
                    options: levels.map { (value: $0, label: $0.rawValue) }
                )
                .help(String(localized: "logs.row.level.help",
                             defaultValue: "Hide everything below this level",
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

// MARK: - Log pane

/// The two sizes this screen uses. The log screen predates the shared numeric
/// tokens; keeping them here means the app PR stays independent of the web
/// console series, and both stay at or above the 12pt floor.
// LogTypeScale, LogRecordBody, LogDetailCard and LogRowView are internal
// rather than private so the offscreen drawing tests (LogsDrawingTests) can
// render the shapes the screen actually draws, not a copy of them.
enum LogTypeScale {
    /// Time, level and module columns.
    static let meta: CGFloat = 12
    /// The message itself; the app's body size.
    static let body: CGFloat = 13
    /// A record's box — the detail card's body and the empty pane's fragment:
    /// tall enough for a dozen lines, scrolled past that instead of growing
    /// with the record.
    static let recordBodyHeight: CGFloat = 200
    /// The detail card's occurrence list: the same idea, one line per time.
    static let occurrencesHeight: CGFloat = 120
    /// Height the row list keeps when a record's card is open above it.
    static let listMinHeight: CGFloat = 200
}

private struct LogPane: View {
    @Bindable var vm: LogsScreenVM

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(spacing: 8) {
            if let selected = vm.selectedRow {
                LogDetailCard(row: selected) { vm.clearSelection() }
            }
            list
                // A selected record must not push the list out of the pane: the
                // card scrolls inside its own boxes, the list keeps its rows.
                .frame(minHeight: LogTypeScale.listMinHeight)
        }
        .frame(minHeight: 360, idealHeight: 480, maxHeight: .infinity)
    }

    @ViewBuilder
    private var list: some View {
        ZStack {
            if vm.rows.isEmpty {
                LogEmptyPane(fragment: vm.leadingFragment, message: vm.emptyStateMessage)
            } else {
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 0) {
                            ForEach(vm.rows) { row in
                                LogRowView(row: row, isSelected: row.id == vm.selectedRowID) {
                                    vm.select(row)
                                }
                                .id(row.id)
                                Divider().overlay(theme.groupBorder)
                            }
                        }
                        .padding(.vertical, 4)
                    }
                    // Following the tail: a new row scrolls into view unless the
                    // reader has switched it off (the button in the filter row).
                    // The key is the window itself, not the last row's id: ids
                    // are parse indices, so a window that keeps its record count
                    // — any homogeneous log under a fixed `lines` — never
                    // changes one and the list stopped scrolling after the
                    // first render.
                    .onChange(of: vm.logText) { _, _ in
                        guard vm.autoScroll, let id = vm.rows.last?.id else { return }
                        withAnimation(.easeOut(duration: 0.18)) {
                            proxy.scrollTo(id, anchor: .bottom)
                        }
                    }
                    .onAppear {
                        guard vm.autoScroll, let id = vm.rows.last?.id else { return }
                        proxy.scrollTo(id, anchor: .bottom)
                    }
                }
            }
        }
        .frame(maxHeight: .infinity)
        .background(theme.codeBg)
        .clipShape(RoundedRectangle(cornerRadius: theme.cornerRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: theme.cornerRadius, style: .continuous)
                .strokeBorder(theme.groupBorder, lineWidth: 0.5)
        )
    }
}

// MARK: - Detail card

/// A record's text in its own bordered, scrolled box. The detail card and the
/// empty pane (a window that begins inside a record) both draw one: a record
/// can carry tens of thousands of continuation lines, so unbounded it made the
/// pane 300,000pt tall and pushed everything else off the screen. The box is a
/// fixed height and the record scrolls inside it.
struct LogRecordBody: View {
    let text: String

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        ScrollView {
            Text(text)
                .font(.omlxMono(LogTypeScale.meta))
                .foregroundStyle(theme.text)
                .textSelection(.enabled)
                .multilineTextAlignment(.leading)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .frame(maxHeight: LogTypeScale.recordBodyHeight, alignment: .top)
        .padding(8)
        .background(theme.inputBg)
        .clipShape(RoundedRectangle(cornerRadius: theme.rowRadius, style: .continuous))
    }
}

/// What the pane shows when it has no row to list. Usually the level rail hid
/// every record, but the window can also have begun inside one: those lines
/// have no row and no level, so they are drawn here rather than dropped, under
/// the message that names the real reason.
private struct LogEmptyPane: View {
    let fragment: LogRecord?
    let message: String

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if let fragment {
                LogRecordBody(text: fragment.fullMessage)
            }
            Text(message)
                .font(.omlxText(LogTypeScale.body))
                .foregroundStyle(theme.textTertiary)
                .frame(maxWidth: .infinity, alignment: .center)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 36)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// The record behind a row: its header, the message in full, and — for a
/// repeated line — every time it appeared, which is what the row's ×N badge
/// stands for. Same content as the web console's detail panel.
struct LogDetailCard: View {
    let row: LogRow
    let onClose: () -> Void

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Image(systemName: "text.magnifyingglass")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(theme.textSecondary)
                Text(String(localized: "logs.detail.title",
                            defaultValue: "Entry",
                            comment: "Title of the card that shows one log record in full"))
                    .font(.omlxText(LogTypeScale.meta, weight: .semibold))
                    .foregroundStyle(theme.textSecondary)
                Text(row.record.time)
                    .font(.omlxMono(LogTypeScale.meta))
                    .foregroundStyle(theme.textSecondary)
                    .monospacedDigit()
                LevelBadge(level: row.record.level)
                if !row.record.module.isEmpty {
                    Text(row.record.module)
                        .font(.omlxMono(LogTypeScale.meta))
                        .foregroundStyle(theme.textTertiary)
                        .lineLimit(1)
                }
                Spacer(minLength: 8)
                Button {
                    onClose()
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "xmark")
                            .font(.system(size: 9, weight: .semibold))
                        Text(String(localized: "logs.detail.close",
                                    defaultValue: "Close",
                                    comment: "Button that closes the log record detail card"))
                            .font(.omlxText(LogTypeScale.meta))
                    }
                }
                .buttonStyle(.omlx(.plain, size: .small))
                .keyboardShortcut(.escape, modifiers: [])
            }

            // The card is a fixed shape: the record scrolls inside its box.
            LogRecordBody(text: row.record.renderedFullMessage)
            if row.record.hiddenContinuationLines > 0 {
                MoreLinesNote(hidden: row.record.hiddenContinuationLines)
            }

            if row.isRepeated {
                VStack(alignment: .leading, spacing: 3) {
                    Text(String(localized: "logs.detail.occurrences",
                                defaultValue: "Occurrences (\(row.count.formatted()))",
                                comment: "Heading above the list of times a repeated log line appeared; the placeholder is how many times"))
                        .font(.omlxText(LogTypeScale.meta, weight: .medium))
                        .foregroundStyle(theme.textSecondary)
                    // Windowed and scrolled for the same reason as the body
                    // above: a warning can repeat tens of thousands of times.
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 3) {
                            ForEach(row.renderedOccurrences) { occurrence in
                                HStack(spacing: 6) {
                                    Text(occurrence.time)
                                        .font(.omlxMono(LogTypeScale.meta))
                                        .foregroundStyle(theme.textTertiary)
                                        .monospacedDigit()
                                    if !occurrence.requestID.isEmpty {
                                        Text(occurrence.requestID)
                                            .font(.omlxMono(LogTypeScale.meta))
                                            .foregroundStyle(theme.textTertiary)
                                            .lineLimit(1)
                                    }
                                }
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .frame(maxHeight: LogTypeScale.occurrencesHeight, alignment: .top)
                    if row.hiddenOccurrences > 0 {
                        Text(String(localized: "logs.detail.occurrences_more",
                                    defaultValue: "…and \(row.hiddenOccurrences.formatted()) more",
                                    comment: "Line under the occurrence list when the list is capped; the placeholder is how many occurrences are not listed"))
                            .font(.omlxMono(LogTypeScale.meta))
                            .foregroundStyle(theme.textTertiary)
                    }
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

/// One line saying how much of a long record is not drawn. The full text is one
/// Copy away, so the note points there rather than pretending the record ended.
private struct MoreLinesNote: View {
    let hidden: Int

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        Text(String(localized: "logs.detail.lines_more",
                    defaultValue: "…and \(hidden.formatted()) more lines",
                    comment: "Line under a capped log record body; the placeholder is how many lines are not drawn. The screen's Copy holds the whole log."))
            .font(.omlxMono(LogTypeScale.meta))
            .foregroundStyle(theme.textTertiary)
    }
}

// MARK: - Rows

/// A level as a tinted badge, the way the console's tables show state: the
/// colour carries the severity, the label spells it out.
private struct LevelBadge: View {
    let level: LogLevel
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        Text(level.label)
            .font(.omlxMono(LogTypeScale.meta, weight: .semibold))
            .foregroundStyle(LogPalette.color(level, theme: theme))
            .padding(.horizontal, 5)
            .padding(.vertical, 1)
            .background(
                RoundedRectangle(cornerRadius: 4, style: .continuous)
                    .fill(LogPalette.color(level, theme: theme).opacity(0.14))
            )
            .frame(width: 38, alignment: .leading)
    }
}

/// One severity's colour, in one place: the row, the badge and the detail card
/// all read the same mapping, so a level is painted one way wherever it lands.
///
/// Six levels, six marks, one meaning each. The HIG's rule is why CRITICAL is
/// not a second red: a colour that means two things means neither, so red stays
/// ERROR's alone and CRITICAL takes the platform's purple. TRACE is the
/// quietest level rather than a hue, and keeps the tertiary label colour.
enum LogPalette {
    static func color(_ level: LogLevel, theme: OMLXTheme) -> Color {
        switch level {
        case .critical: theme.purpleDot
        case .error: theme.redDot
        case .warning: theme.amberDot
        case .info: theme.blueDot
        case .debug: theme.tealDot
        case .trace, .other: theme.textTertiary
        }
    }
}

/// One row: time and level in fixed columns so the eye can run down them, module
/// next, then the message — which wraps and keeps its continuation lines
/// (tracebacks) as one block. A repeated record carries its ×N badge, and the
/// whole row opens the detail card.
struct LogRowView: View {
    let row: LogRow
    let isSelected: Bool
    let onSelect: () -> Void

    @Environment(\.omlxTheme) private var theme
    @State private var expanded = false

    private var record: LogRecord { row.record }

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Text(Self.clockTime(record.time))
                .font(.omlxMono(LogTypeScale.meta))
                .foregroundStyle(theme.textTertiary)
                .frame(width: 132, alignment: .leading)
                .monospacedDigit()

            LevelBadge(level: record.level)

            VStack(alignment: .leading, spacing: 2) {
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    if !record.module.isEmpty {
                        Text(record.module)
                            .font(.omlxMono(LogTypeScale.meta))
                            .foregroundStyle(theme.textSecondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .frame(width: 150, alignment: .leading)
                    }
                    Text(record.message)
                        .font(.omlxText(LogTypeScale.body))
                        .foregroundStyle(theme.text)
                        .textSelection(.enabled)
                        .multilineTextAlignment(.leading)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    if row.isRepeated {
                        Text("×\(row.count.formatted())")
                            .font(.omlxText(LogTypeScale.meta, weight: .semibold))
                            .foregroundStyle(theme.textSecondary)
                            .padding(.horizontal, 4)
                            .padding(.vertical, 0.5)
                            .background(
                                RoundedRectangle(cornerRadius: 4, style: .continuous)
                                    .fill(theme.controlBg)
                            )
                            .help(String(localized: "logs.row.repeat.help",
                                         defaultValue: "This line repeats; open the row to see every occurrence",
                                         comment: "Tooltip on the ×N badge of a repeated log line"))
                    }
                }

                if !record.continuation.isEmpty {
                    if expanded {
                        // The first few lines only: the card below holds the whole
                        // record, and a row that drew all of it filled the pane.
                        Text(record.renderedInlineContinuation.joined(separator: "\n"))
                            .font(.omlxMono(LogTypeScale.meta))
                            .foregroundStyle(theme.textSecondary)
                            .textSelection(.enabled)
                            .multilineTextAlignment(.leading)
                            .fixedSize(horizontal: false, vertical: true)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        if record.hiddenInlineContinuationLines > 0 {
                            MoreLinesNote(hidden: record.hiddenInlineContinuationLines)
                        }
                    } else {
                        Button {
                            expanded = true
                        } label: {
                            Text(String(localized: "logs.more_lines",
                                        defaultValue: "≡ \(record.continuation.count.formatted()) more lines",
                                        comment: "Button under a log record that reveals its continuation lines; the placeholder is how many lines are hidden"))
                                .font(.omlxText(LogTypeScale.meta))
                                .foregroundStyle(theme.textTertiary)
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 5)
        .background(isSelected ? theme.selBg : Color.clear)
        .contentShape(Rectangle())
        .onTapGesture(perform: onSelect)
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(isSelected ? [.isSelected] : [])
    }

    /// The server stamps `yyyy-MM-dd HH:mm:ss,SSS`; the column shows the clock
    /// time alone and falls back to the raw stamp when it does not parse.
    private static func clockTime(_ stamp: String) -> String {
        guard let parsed = stampParser.date(from: stamp) else { return stamp }
        return clockFormatter.string(from: parsed)
    }

    private static let stampParser: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss,SSS"
        formatter.locale = Locale(identifier: "en_US_POSIX")
        return formatter
    }()

    private static let clockFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss.SSS"
        return formatter
    }()
}

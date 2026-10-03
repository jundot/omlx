import SwiftUI

/// Which empty pane the screen is showing. The three situations used to share
/// one message, and the one a reader of a window that began inside a record hit
/// told them to lower a level filter that had hidden nothing.
enum LogsEmptyState: Equatable {
    /// The server returned no text at all.
    case noEntries
    /// The window is the tail of a record whose header scrolled out of it:
    /// there is no level for the rail to hide and no row to list.
    case startsInsideRecord
    /// The text holds header records; the rail is hiding all of them.
    case filteredOut
}

@MainActor
@Observable
final class LogsScreenVM {
    var lines: Int = 100
    var selectedFile: String = ""
    var logText: String = ""
    /// Parsed once per refresh: the screen renders records, not the raw text.
    private(set) var records: [LogRecord] = []
    /// Lowest level the list shows. `trace` keeps everything. The rail is the
    /// list's own filter, so a change re-aggregates it.
    var minLevel: LogLevel = .info { didSet { aggregate() } }
    /// The row whose occurrences are open in the detail card, if any.
    var selectedRowID: Int?
    /// Follow the tail: when it is on, a refresh scrolls to the newest row.
    var autoScroll: Bool = true

    /// The list the screen renders: identical consecutive records are one row.
    /// Aggregated once per refresh (and once per filter change) rather than on
    /// every read: the body read it several times per draw, and a 20,000-line
    /// refresh is 20,000 records to walk each time.
    private(set) var rows: [LogRow] = []
    var selectedRow: LogRow? { rows.first { $0.id == selectedRowID } }
    var availableFiles: [String] = []
    var lastError: String?
    private(set) var isLoading: Bool = false
    private(set) var totalLines: Int = 0

    /// Feed a freshly fetched window to the screen. Separate from `tick()` so
    /// the window can also be applied without a server (see the tests).
    func applyLogText(_ text: String) {
        logText = text
        reparse()
    }

    private func reparse() {
        // The window is the tail of a file that keeps growing, so a record's
        // index here changes as lines arrive. The open row is carried across
        // the refresh by its header, not by the index it held in the old window,
        // and it stays open while it only gains continuation lines: the newest
        // record is still being written when the refresh lands.
        let openRecord = selectedRow?.record
        records = LogParser.parse(logText)
        rows = LogRows.aggregate(records, minLevel: minLevel)
        guard let openRecord else { return }
        selectedRowID = rows.first { $0.record.continues(openRecord) }?.id
    }

    private func aggregate() {
        rows = LogRows.aggregate(records, minLevel: minLevel)
        // The rail is the list's own filter: a row it hides is not open, and
        // keeping its id would open a different record when the filter returns.
        if let id = selectedRowID, !rows.contains(where: { $0.id == id }) {
            selectedRowID = nil
        }
    }
    private(set) var refreshKey: Int = 0

    @ObservationIgnored
    private weak var client: OMLXClient?
    @ObservationIgnored
    private var pollTask: Task<Void, Never>?

    var subtitle: String {
        guard !logText.isEmpty else { return "" }
        return String(localized: "logs.subtitle.line_count",
                      defaultValue: "Lines: \(totalLines.formatted())",
                      comment: "Section header subtitle on the Logs screen; placeholder is the total number of log lines")
    }

    var fileOptions: [(String, String)] {
        availableFiles.map { name in
            let label = name == "server.log"
                ? String(localized: "logs.file.current",
                         defaultValue: "server.log (current)",
                         comment: "Popup label for the active server log file in the Logs screen file selector")
                : name
            return (name, label)
        }
    }

    /// The leading lines of a window that began inside a record. The parser
    /// keeps them as a display-only record (see `LogRecord.isFragment`); there
    /// is no row to hold them, so the empty pane draws them.
    var leadingFragment: LogRecord? {
        records.first { $0.isFragment }
    }

    /// Which empty pane this is, decided from the parse rather than from the
    /// rows, so a window that begins inside a record can never be told to lower
    /// the level filter: there is no level for the filter to have hidden.
    var emptyState: LogsEmptyState {
        guard !records.isEmpty else { return .noEntries }
        return records.contains { $0.isHeader } ? .filteredOut : .startsInsideRecord
    }

    var emptyStateMessage: String {
        switch emptyState {
        case .noEntries:
            String(localized: "logs.empty",
                   defaultValue: "No log entries.",
                   comment: "Empty-state text shown inside the log pane when the server has no log entries")
        case .startsInsideRecord:
            String(localized: "logs.empty.partial",
                   defaultValue: "This window starts inside a log entry. Refresh to see the newest lines.",
                   comment: "Empty-state text shown when the fetched log window begins in the middle of a record whose header line is above the window")
        case .filteredOut:
            String(localized: "logs.empty.filtered",
                   defaultValue: "No lines at this level. Lower the level filter to see more.",
                   comment: "Empty-state text shown when the level filter hides every log line")
        }
    }

    func start(client: OMLXClient) async {
        self.client = client
        pollTask?.cancel()
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                await self.tick()
                try? await Task.sleep(for: .seconds(5))
            }
        }
    }

    func stop() {
        pollTask?.cancel()
        pollTask = nil
    }

    func reload() async {
        await tick()
    }

    func bumpRefreshKey() {
        refreshKey &+= 1
    }

    func select(_ row: LogRow) {
        selectedRowID = selectedRowID == row.id ? nil : row.id
    }

    func clearSelection() {
        selectedRowID = nil
    }

    func copyToPasteboard() {
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(logText, forType: .string)
    }

    private func tick() async {
        guard let client else { return }
        isLoading = true
        defer { isLoading = false }

        let file = selectedFile.isEmpty ? nil : selectedFile
        do {
            let dto = try await client.getLogs(lines: lines, file: file)
            self.applyLogText(dto.logs)
            self.totalLines = dto.totalLines
            self.availableFiles = dto.availableFiles
            if selectedFile.isEmpty {
                self.selectedFile = dto.logFile
            }
            self.lastError = nil
        } catch {
            self.lastError = error.omlxDescription
        }
    }

}

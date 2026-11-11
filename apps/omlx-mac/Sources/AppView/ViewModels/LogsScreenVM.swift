import SwiftUI

enum LogViewMode: Sendable {
    case table, raw
}

@MainActor
@Observable
final class LogsScreenVM {
    // Defaults match the web UI log viewer.
    var lines: Int = 500
    var selectedFile: String = "" {
        didSet { if selectedFile != oldValue { selectedRowID = nil } }
    }
    var viewMode: LogViewMode = .table
    var minLevel: LogLevel = .trace {
        didSet {
            selectedRowID = nil
            rebuildRows()
        }
    }
    var autoScroll = true
    var selectedRowID: Int?
    private(set) var logText: String = ""
    private(set) var rawText: String = ""
    private(set) var rows: [LogRow] = []
    var availableFiles: [String] = []
    var lastError: String?
    private(set) var isLoading: Bool = false
    private(set) var totalLines: Int = 0
    private(set) var refreshKey: Int = 0

    @ObservationIgnored
    private var records: [LogRecord] = []
    @ObservationIgnored
    private weak var client: OMLXClient?
    @ObservationIgnored
    private var pollTask: Task<Void, Never>?

    var selectedRow: LogRow? {
        rows.first { $0.id == selectedRowID }
    }

    var subtitle: String {
        guard !logText.isEmpty else { return "" }
        return String(localized: "logs.subtitle.line_count",
                      defaultValue: "Lines: \(totalLines)",
                      comment: "Section header subtitle on the Logs screen; placeholder is the total number of log lines")
    }

    var emptyMessage: String {
        totalLines > 0
            ? String(localized: "logs.empty.filtered",
                     defaultValue: "No lines at this level. Lower the level filter to see more.",
                     comment: "Empty-state text shown when the level filter hides every log line")
            : String(localized: "logs.empty",
                     defaultValue: "No log entries.",
                     comment: "Empty-state text shown inside the log pane when the server has no log entries")
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

    func toggleSelection(_ row: LogRow) {
        selectedRowID = selectedRowID == row.id ? nil : row.id
    }

    func copyToPasteboard() {
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(viewMode == .raw ? rawText : logText, forType: .string)
    }

    func apply(logs: String, totalLines: Int) {
        logText = logs
        self.totalLines = totalLines
        records = LogParser.parse(logs, totalLines: totalLines)
        rebuildRows()
    }

    private func rebuildRows() {
        rows = LogParser.rows(records, minLevel: minLevel)
        rawText = LogParser.filter(logText, minLevel: minLevel)
        if let id = selectedRowID, !rows.contains(where: { $0.id == id }) {
            selectedRowID = nil
        }
    }

    private func tick() async {
        guard let client else { return }
        isLoading = true
        defer { isLoading = false }

        let file = selectedFile.isEmpty ? nil : selectedFile
        do {
            let dto = try await client.getLogs(lines: lines, file: file)
            apply(logs: dto.logs, totalLines: dto.totalLines)
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

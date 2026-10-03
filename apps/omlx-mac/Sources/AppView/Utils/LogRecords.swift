// Log line parsing for the Logs screen.
//
// The server writes `%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s`;
// a traceback or a parameter dump continues on following lines that carry no
// timestamp. Reading that as one raw string is what made the screen hard to
// scan, so the screen works on records instead: one entry per header line with
// its continuation lines attached, a level with its own colour, and the module
// kept separate from the message.
//
// A fetched window is the tail of a file that keeps growing, so it can begin in
// the middle of a record whose header line is above it. Those leading lines are
// kept too (see `LogRecord.isFragment`): dropping them left a pane that held
// text with nothing to show.
//
// Pure value types and pure functions: no AppKit, no SwiftUI, no I/O, so the
// contract is testable on its own (see LogRecordsTests).

import Foundation

/// The severities the server can emit, in increasing order.
enum LogLevel: String, CaseIterable, Sendable {
    case trace = "TRACE"
    case debug = "DEBUG"
    case info = "INFO"
    case warning = "WARNING"
    case error = "ERROR"
    case critical = "CRITICAL"
    case other = ""

    /// Short label for the level column. `other` is a record whose level word
    /// is none of the six; it is parsed, not listed.
    var label: String {
        switch self {
        case .trace: "TRC"
        case .debug: "DBG"
        case .info: "INF"
        case .warning: "WRN"
        case .error: "ERR"
        case .critical: "CRT"
        case .other: "—"
        }
    }

    var rank: Int {
        switch self {
        case .trace: 0
        case .debug: 1
        case .info: 2
        case .warning: 3
        case .error: 4
        case .critical: 5
        case .other: -1
        }
    }
}

/// One log entry: the header line plus any continuation lines.
struct LogRecord: Identifiable, Sendable {
    let id: Int
    let time: String
    let level: LogLevel
    let module: String
    let message: String
    /// Continuation lines (tracebacks, parameter dumps), already trimmed.
    let continuation: [String]
    let requestID: String

    var isHeader: Bool { level != .other }

    /// True when the window began inside this record: its header line is above
    /// the fetched tail, so it was read from continuation lines alone and has
    /// no time, level, module or request id. The parser stamps a time only from
    /// a header, so an empty one is what marks the fragment.
    ///
    /// It is a display-only record: the list has no row for it (there is no
    /// level for the rail to compare and nothing to aggregate on), and the
    /// empty pane draws its lines rather than dropping them.
    var isFragment: Bool { time.isEmpty }

    /// Whether `earlier` is this same entry, seen again in a later window: the
    /// header is identical and the continuation only grew. The header alone is
    /// not enough to match on — an unrelated entry can repeat it — so what was
    /// open has to still be the start of what is here now.
    func continues(_ earlier: LogRecord) -> Bool {
        time == earlier.time
            && level == earlier.level
            && module == earlier.module
            && message == earlier.message
            && requestID == earlier.requestID
            && continuation.starts(with: earlier.continuation)
    }

    /// The header line plus `lines`: the whole record for Copy, or only the
    /// capped continuation the card draws.
    private func joined(_ lines: [String]) -> String {
        lines.isEmpty ? message : ([message] + lines).joined(separator: "\n")
    }

    /// The message a row shows, continuation included, for the detail pane.
    var fullMessage: String { joined(continuation) }

    /// What the detail card draws: the header line plus the capped continuation.
    var renderedFullMessage: String { joined(renderedContinuation) }

    /// The continuation lines the detail card draws, capped.
    var renderedContinuation: [String] { LogRenderLimits.capped(continuation, LogRenderLimits.continuationLines) }

    /// Continuation lines the cap leaves out; 0 when the record is short enough.
    var hiddenContinuationLines: Int { LogRenderLimits.hidden(continuation.count, LogRenderLimits.continuationLines) }

    /// The continuation lines a row draws when expanded in place.
    var renderedInlineContinuation: [String] { LogRenderLimits.capped(continuation, LogRenderLimits.inlineContinuationLines) }

    /// Continuation lines hidden behind `renderedInlineContinuation`.
    var hiddenInlineContinuationLines: Int { LogRenderLimits.hidden(continuation.count, LogRenderLimits.inlineContinuationLines) }
}

/// How much of a record the screen lays out at once. A record can carry tens of
/// thousands of continuation lines (a dumped parameter list, a traceback with a
/// frame per layer) and a repeated line can have as many occurrences; drawing
/// every line and every occurrence in one pass is what froze the screen. The
/// numbers live next to the parsing so the row and the detail card cannot drift
/// apart, and the whole record stays available: `fullMessage` is untouched and
/// the screen's Copy puts the log text on the pasteboard.
enum LogRenderLimits {
    /// Continuation lines the detail card draws (they scroll inside its box).
    static let continuationLines = 200
    /// Occurrences listed behind a row's ×N badge.
    static let occurrences = 200
    /// Continuation lines a *row* draws when it is expanded in place. The card is
    /// where a whole traceback belongs; a row that drew one made the list a
    /// single 600pt row with nothing else in view, so the row shows the first
    /// few lines and counts the rest behind the same "≡ N more lines" note.
    static let inlineContinuationLines = 4

    /// The first `limit` items a bounded list draws.
    static func capped<T>(_ items: [T], _ limit: Int) -> [T] { Array(items.prefix(limit)) }

    /// How many items a `limit`-capped list leaves out; 0 when it fits.
    static func hidden(_ count: Int, _ limit: Int) -> Int { max(0, count - limit) }
}

/// One occurrence of a repeated record: when it happened and which request it
/// belonged to. The web console shows the same list behind a row's ×N badge.
struct LogOccurrence: Sendable, Equatable, Identifiable {
    /// Position inside the run. `time` and `requestID` alone can repeat within
    /// one run — a burst of identical lines lands in the same millisecond — and
    /// a `ForEach` needs unique ids.
    let index: Int
    let time: String
    let requestID: String
    var id: Int { index }
}

/// One row of the list. A row is usually a single record; consecutive records
/// that repeat — same level, module and message — become one row with a count,
/// which is what keeps a polling loop from burying everything else.
struct LogRow: Identifiable, Sendable {
    let record: LogRecord
    let occurrences: [LogOccurrence]

    var id: Int { record.id }
    var count: Int { occurrences.count }
    var isRepeated: Bool { count > 1 }

    /// The occurrences the detail card lists, capped (see `LogRenderLimits`).
    var renderedOccurrences: [LogOccurrence] { LogRenderLimits.capped(occurrences, LogRenderLimits.occurrences) }

    /// Occurrences the cap leaves out; 0 when the group is short enough.
    var hiddenOccurrences: Int { LogRenderLimits.hidden(occurrences.count, LogRenderLimits.occurrences) }
}

enum LogRows {
    /// Runs of identical records collapse from `WARNING` up; `INFO` and below
    /// are the traffic of a busy server and stay one row each. A record the
    /// level filter hides ends a run, because what is left is not consecutive.
    static let aggregateFrom: LogLevel = .warning

    /// A record whose level word is none of the six is parsed but never listed:
    /// the server's logger registers no level beyond them — 82k lines of real
    /// server.log hold none — and a row without a severity is only noise.
    ///
    /// A fragment is not listed either, for a different reason: it has no
    /// header at all (see `LogRecord.isFragment`) and the pane draws it itself.
    static func aggregate(_ records: [LogRecord], minLevel: LogLevel) -> [LogRow] {
        var rows: [LogRow] = []
        var current: LogRecord?
        var times: [LogOccurrence] = []

        func closeRun() {
            guard let first = current else { return }
            rows.append(LogRow(record: first, occurrences: times))
        }

        for record in records {
            // A fragment has no header to aggregate on and no level to compare,
            // so it is not a row: the empty pane draws its lines instead.
            guard record.isHeader else { continue }
            guard record.level.rank >= minLevel.rank else {
                // A hidden record ends the run: what remains is not consecutive.
                closeRun()
                current = nil
                continue
            }
            if let run = current,
               run.level.rank >= aggregateFrom.rank,
               run.level == record.level,
               run.module == record.module,
               run.message == record.message {
                times.append(LogOccurrence(index: times.count, time: record.time, requestID: record.requestID))
                continue
            }
            closeRun()
            current = record
            times = [LogOccurrence(index: 0, time: record.time, requestID: record.requestID)]
        }
        closeRun()
        return rows
    }
}

enum LogParser {
    /// One pass over the text, with the open record's continuation lines
    /// collected in an array and handed over when the record closes. Appending
    /// to `records.last.continuation` instead would copy every line collected so
    /// far on every line of a 20,000-line parameter dump — quadratic, and the
    /// screen hung on the refresh that carried one.
    static func parse(_ text: String) -> [LogRecord] {
        var records: [LogRecord] = []
        var id = 0
        // The record being read: its header fields plus the lines so far.
        var open: (id: Int, time: String, level: LogLevel, module: String,
                   message: String, requestID: String)?
        var continuation: [String] = []
        // Lines read before the window's first header. The window is the tail of
        // a file that keeps growing, so it can begin in the middle of a record
        // whose header scrolled out; those lines are kept as one fragment.
        var leadingFragment: [String] = []

        func closeOpen() {
            guard let record = open else { return }
            records.append(LogRecord(id: record.id, time: record.time, level: record.level,
                                     module: record.module, message: record.message,
                                     continuation: continuation, requestID: record.requestID))
            open = nil
            continuation = []
        }

        func closeLeadingFragment() {
            guard !leadingFragment.isEmpty else { return }
            // The first line carries the message and the rest the continuation,
            // so the fragment draws, selects and copies like any other record.
            records.append(LogRecord(id: id, time: "", level: .other, module: "",
                                     message: leadingFragment[0],
                                     continuation: Array(leadingFragment.dropFirst()),
                                     requestID: ""))
            id += 1
            leadingFragment = []
        }

        for rawLine in text.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = String(rawLine)
            if let match = header(line) {
                // The fragment is emitted before the first header, so the window
                // keeps its order and the fragment takes the first id.
                closeLeadingFragment()
                closeOpen()
                open = (id, match.time, match.level, match.module, match.message, match.requestID)
                id += 1
            } else if !line.trimmingCharacters(in: .whitespaces).isEmpty {
                // A continuation line belongs to the record above it; before the
                // first header it belongs to the record the window cut off.
                if open != nil {
                    continuation.append(line)
                } else {
                    leadingFragment.append(line)
                }
            }
        }
        closeOpen()
        closeLeadingFragment()
        return records
    }

    /// `2026-09-21 00:56:04,543 - omlx.server - WARNING - [-] - …`
    ///
    /// The fields are separated by ` - `; the stamp, an uppercase level word
    /// and a bracketed request id are what tell a header from a continuation
    /// line that happens to contain the separator.
    private static func header(_ line: String) -> (time: String, level: LogLevel, module: String,
                                                   message: String, requestID: String)? {
        let fields = line.split(separator: " - ", maxSplits: 4, omittingEmptySubsequences: false)
        guard fields.count == 5, isStamp(fields[0]), !fields[1].isEmpty,
              !fields[2].isEmpty, fields[2].allSatisfy({ $0.isASCII && $0.isUppercase }),
              fields[3].hasPrefix("["), fields[3].hasSuffix("]"),
              !fields[3].dropFirst().dropLast().contains("]")
        else { return nil }
        return (String(fields[0]),
                LogLevel(rawValue: String(fields[2])) ?? .other,
                String(fields[1]).trimmingCharacters(in: .whitespaces),
                String(fields[4]),
                String(fields[3].dropFirst().dropLast()))
    }

    /// `yyyy-MM-dd HH:mm:ss,SSS`: two date halves, then the clock. Fixed shape,
    /// so a continuation line that merely holds the separator is not a header.
    private static func isStamp(_ text: Substring) -> Bool {
        let halves = text.split(separator: " ", omittingEmptySubsequences: false)
        return halves.count == 2
            && halves[0].split(separator: "-").map(\.count) == [4, 2, 2]
            && halves[0].allSatisfy { $0.isNumber || $0 == "-" }
            && !halves[1].isEmpty
            && halves[1].allSatisfy { $0.isNumber || $0 == ":" || $0 == "," }
    }
}

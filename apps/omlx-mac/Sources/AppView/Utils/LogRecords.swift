// Record view of the Logs screen. Same rules as the web UI's logs.js.

import Foundation

enum LogLevel: String, CaseIterable, Comparable, Sendable {
    case trace = "TRACE"
    case debug = "DEBUG"
    case info = "INFO"
    case warning = "WARNING"
    case error = "ERROR"
    case critical = "CRITICAL"

    static func < (lhs: LogLevel, rhs: LogLevel) -> Bool {
        allCases.firstIndex(of: lhs)! < allCases.firstIndex(of: rhs)!
    }
}

/// One header line and the lines under it.
struct LogRecord: Sendable {
    /// Absolute line number in the file, so it stays the same while the
    /// record is in the window.
    let line: Int
    let time: String
    let module: String
    /// Nil for the lines before the first header: the window cut that record.
    let level: LogLevel?
    var message: String
    var lineCount = 1

    var firstLine: Substring { message.prefix { $0 != "\n" } }
}

struct LogRow: Identifiable, Sendable {
    let record: LogRecord
    /// One time per consecutive identical record.
    var times: [String]

    var id: Int { record.line }
}

enum LogMemoryField: CaseIterable, Sendable {
    case usage, watermark, ceiling, peak
}

enum LogParser {
    static let aggregateFrom = LogLevel.warning
    static let occurrenceLimit = 200

    private static let stamp = Array("0000-00-00 00:00:00,000".utf8)

    /// `totalLines` is the line count of the file that `text` is the tail of.
    static func parse(_ text: String, totalLines: Int) -> [LogRecord] {
        var lines = text.split(separator: "\n", omittingEmptySubsequences: false)
        if lines.last?.isEmpty == true { lines.removeLast() }
        let first = max(0, totalLines - lines.count)
        var records: [LogRecord] = []
        var open: LogRecord?
        for (index, line) in lines.enumerated() {
            if let header = header(line) {
                if let open { records.append(open) }
                open = LogRecord(line: first + index, time: String(header.time),
                                 module: String(header.module), level: header.level,
                                 message: String(header.message))
            } else if open != nil {
                open?.message.append("\n")
                open?.message.append(contentsOf: line)
                open?.lineCount += 1
            } else {
                open = LogRecord(line: first + index, time: "", module: "", level: nil,
                                 message: String(line))
            }
        }
        if let open { records.append(open) }
        return records
    }

    /// Consecutive identical records from WARNING up become one row. A hidden
    /// record ends a run.
    static func rows(_ records: [LogRecord], minLevel: LogLevel) -> [LogRow] {
        var rows: [LogRow] = []
        var runOpen = false
        for record in records {
            if let level = record.level, level < minLevel {
                runOpen = false
                continue
            }
            if runOpen, let level = record.level, level >= aggregateFrom,
               let last = rows.last?.record, last.level == level,
               last.module == record.module, last.message == record.message {
                rows[rows.count - 1].times.append(record.time)
                continue
            }
            rows.append(LogRow(record: record, times: [record.time]))
            runOpen = true
        }
        return rows
    }

    /// The window without the records below `minLevel`. Lines before the first
    /// header are kept.
    static func filter(_ text: String, minLevel: LogLevel) -> String {
        guard minLevel > .trace else { return text }
        var visible = true
        return text.split(separator: "\n", omittingEmptySubsequences: false).filter { line in
            if let level = header(line)?.level { visible = level >= minLevel }
            return visible
        }.joined(separator: "\n")
    }

    /// The numbers in a memory-guard message, as the server wrote them.
    static func memoryGuard(_ message: String) -> [LogMemoryField: String]? {
        let markers = ["process memory limit exceeded", "Prefill would require",
                       "Hard memory pressure", "due to memory pressure", "raise memory_guard_tier"]
        guard markers.contains(where: { message.contains($0) }) else { return nil }

        func number(_ regex: Regex<(Substring, Substring)>) -> String? {
            message.firstMatch(of: regex).map { String($0.1) }
        }
        var fields: [LogMemoryField: String] = [:]
        fields[.usage] = number(/\(usage ([\d.]+) GB/) ?? number(/\(current ([\d.]+) GB/)
        fields[.watermark] = number(/abort threshold \(hard watermark\) ([\d.]+) GB/)
        fields[.ceiling] = number(/\b[a-z_\/]+ ceiling (?:is )?([\d.]+) GB/)
        fields[.peak] = number(/require ~?([\d.]+) GB peak/)
        return fields.isEmpty ? nil : fields
    }

    /// `asctime - name - LEVEL - [request_id] - message`, request id optional.
    private static func header(_ line: Substring)
        -> (time: Substring, module: Substring, level: LogLevel, message: Substring)?
    {
        let fields = line.split(separator: " - ", maxSplits: 3, omittingEmptySubsequences: false)
        guard fields.count == 4,
              fields[0].utf8.elementsEqual(stamp, by: { byte, mask in
                  mask == UInt8(ascii: "0") ? (0x30...0x39).contains(byte) : byte == mask
              }),
              let level = LogLevel(rawValue: String(fields[2]))
        else { return nil }
        var message = fields[3]
        if message.hasPrefix("["), let end = message.range(of: "] - ") {
            message = message[end.upperBound...]
        }
        return (fields[0], fields[1], level, message)
    }
}

import XCTest

@testable import oMLX

final class LogRecordsTests: XCTestCase {
    private func line(_ level: String, _ message: String, second: Int = 0) -> String {
        "2026-09-24 00:00:\(String(format: "%02d", second)),000 - omlx.server - \(level) - [-] - \(message)"
    }

    func testRecordsKeepTheirFileLineAsTheWindowSlides() throws {
        let file = [
            "Traceback (most recent call last):",
            line("ERROR", "boom", second: 1),
            "  frame one",
            line("INFO", "queued", second: 2),
            "2026-09-24 00:00:03,000 - omlx.engine - WARNING - no request id",
        ]
        let records = LogParser.parse(file.joined(separator: "\n") + "\n", totalLines: 10)

        XCTAssertEqual(records.map(\.line), [5, 6, 8, 9])
        XCTAssertNil(records[0].level, "lines before the first header are kept as a fragment")
        XCTAssertEqual(records[1].message, "boom\n  frame one")
        XCTAssertEqual(records[1].lineCount, 2)
        XCTAssertEqual(records[3].level, .warning)
        XCTAssertEqual(records[3].module, "omlx.engine")
        XCTAssertEqual(records[3].message, "no request id")

        let grown = file + [line("INFO", "next", second: 4), line("INFO", "more", second: 5)]
        let later = LogParser.parse(grown.suffix(5).joined(separator: "\n") + "\n", totalLines: 12)
        XCTAssertEqual(later.map(\.line), [7, 8, 9, 10, 11])
    }

    func testRowsCollapseRepeatedWarningsOnly() {
        let text = [
            line("WARNING", "slow"), line("WARNING", "slow"), line("DEBUG", "tick"),
            line("WARNING", "slow"), line("INFO", "ok"), line("INFO", "ok"),
        ].joined(separator: "\n")
        let records = LogParser.parse(text, totalLines: 6)

        let all = LogParser.rows(records, minLevel: .trace)
        XCTAssertEqual(all.map(\.times.count), [2, 1, 1, 1, 1])

        // A hidden record still ends the run.
        let info = LogParser.rows(records, minLevel: .info)
        XCTAssertEqual(info.map(\.times.count), [2, 1, 1, 1])

        let fragment = LogParser.parse("tail of a traceback\n" + line("INFO", "ok"), totalLines: 2)
        XCTAssertEqual(LogParser.rows(fragment, minLevel: .error).count, 1)
    }

    func testRawFilterDropsARecordWithItsLines() {
        let text = ["cut off", line("DEBUG", "quiet"), "  detail", line("ERROR", "loud")]
            .joined(separator: "\n")
        XCTAssertEqual(LogParser.filter(text, minLevel: .error), "cut off\n" + line("ERROR", "loud"))
        XCTAssertEqual(LogParser.filter(text, minLevel: .trace), text)
    }

    func testMemoryGuardNumbers() {
        let abort = "Request aborted: process memory limit exceeded (usage 91.2 GB, abort "
            + "threshold (hard watermark) 88.0 GB, dynamic ceiling 96.0 GB). Retry later."
        XCTAssertEqual(LogParser.memoryGuard(abort),
                       [.usage: "91.2", .watermark: "88.0", .ceiling: "96.0"])

        let prefill = "Prefill would require ~41.20 GB peak (current 30.00 GB + KV+SDPA "
            + "11.20 GB) but dynamic ceiling is 40.00 GB (usage 29.1 GB, ceiling 40.0 GB)."
        XCTAssertEqual(LogParser.memoryGuard(prefill),
                       [.usage: "29.1", .ceiling: "40.00", .peak: "41.20"])

        XCTAssertNil(LogParser.memoryGuard("Loaded model in 3.1 GB"))
    }
}

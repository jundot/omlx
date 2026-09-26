// The drawing layer, offscreen.
//
// `LogRecordsTests` pins the caps: `LogRenderLimits`, the model's capped views,
// the hidden counts. Those are properties of the values. What they exist for is
// the other end — the height the screen actually asks for when the real
// `LogDetailCard` and `LogRowView` are handed a record with tens of thousands
// of lines and a line repeated tens of thousands of times. Before the caps,
// that shape measured 2.72 s and a 360,000 pt column offscreen; the model
// staying capped says nothing about a view that stopped reading the capped
// views (an eager stack, a dropped `maxHeight`), and this is where that turns
// red instead of freezing the screen.

import SwiftUI
import XCTest

@testable import oMLX

@MainActor
final class LogsDrawingTests: XCTestCase {

    /// The pane the card and the rows sit in, roughly: wide enough that the
    /// notes under the boxes stay on one line, narrow enough to be the real
    /// thing on a laptop.
    private let width: CGFloat = 720

    // MARK: - Bounded shapes

    /// Every shape the pane draws is bounded: a bigger record cannot buy a
    /// taller card, box or row. Both records of a pair are past the relevant
    /// cap, so the two measurements have to agree.
    func testEveryShapeStaysBounded() throws {
        // (name, huge record, record just past the cap, bound in points)
        let cases: [(String, AnyView, AnyView, CGFloat)] = [
            // 454pt measured: header, the record box (its fixed height plus
            // padding), the note, and the occurrence box (its fixed height, its
            // heading and its note) — the type scale's own numbers, nothing
            // that scales with the record.
            ("detail card",
             AnyView(card(row(continuations: 20_000, occurrences: 20_000))),
             AnyView(card(row(continuations: 400, occurrences: 400))),
             520),
            // The box is the type scale's height plus its own padding and
            // nothing else: 8pt a side.
            ("record box",
             AnyView(LogRecordBody(text: record(continuations: 20_000).renderedFullMessage)),
             AnyView(LogRecordBody(text: record(continuations: LogRenderLimits
                 .continuationLines).renderedFullMessage)),
             LogTypeScale.recordBodyHeight + 32),
            // 43pt measured: one message line and the note under it.
            ("row",
             AnyView(rowView(row(continuations: 20_000, occurrences: 1))),
             AnyView(rowView(row(continuations: 4, occurrences: 1))),
             120),
        ]
        for (name, huge, near, bound) in cases {
            let tall = try height(of: huge)
            let shorter = try height(of: near)
            XCTAssertEqual(tall, shorter, accuracy: 1,
                           "the \(name) grew with the record: \(tall)pt vs \(shorter)pt")
            XCTAssertLessThan(tall, bound,
                              "the \(name) asked for \(tall)pt at \(width)pt wide")
        }
    }

    /// The time the same shape takes offscreen, on the pattern of
    /// `testALongParameterDumpParsesInOnePass`: one pass, one generous bound,
    /// far above what this costs (5–13 ms measured, warm) and below what it
    /// cost uncapped (2.72 s).
    func testTheCardRendersOffscreenInOnePass() {
        // Warm the render path: fonts, locale, the app's theme.
        _ = render(of: card(row(continuations: 4, occurrences: 4)))

        let passes = (0..<3).compactMap { _ in
            render(of: card(row(continuations: 20_000, occurrences: 20_000)))?.seconds
        }
        let fastest = passes.min()
        XCTAssertNotNil(fastest)
        XCTAssertLessThan(fastest ?? .infinity, 1.5,
                          "the card took \(passes.map { String(format: "%.3f", $0) })s offscreen")
    }

    // MARK: - Scaffolding

    /// Render `content` at the width the pane lays it out at: the height it
    /// asks for and how long the render took. Nil when the render produced no
    /// image.
    private func render<V: View>(of content: V) -> (height: CGFloat, seconds: TimeInterval)? {
        let renderer = ImageRenderer(content: content.frame(width: width))
        renderer.scale = 1
        let started = Date()
        guard let image = renderer.nsImage else { return nil }
        return (image.size.height, Date().timeIntervalSince(started))
    }

    private func height<V: View>(of content: V) throws -> CGFloat {
        try XCTUnwrap(render(of: content)).height
    }

    private func card(_ row: LogRow) -> some View {
        LogDetailCard(row: row) {}
            .omlxThemed()
    }

    private func rowView(_ row: LogRow) -> some View {
        LogRowView(row: row, isSelected: false) {}
            .omlxThemed()
    }

    private func record(continuations: Int) -> LogRecord {
        LogRecord(id: 1,
                  time: "2026-09-21 00:56:04,543",
                  level: .error,
                  module: "omlx.server",
                  message: "request failed",
                  continuation: (0..<continuations).map { "frame \($0): something went wrong" },
                  requestID: "abc123")
    }

    private func row(continuations: Int, occurrences: Int) -> LogRow {
        LogRow(record: record(continuations: continuations),
               occurrences: (0..<occurrences).map {
                   LogOccurrence(index: $0,
                                 time: "2026-09-21 00:00:00,000",
                                 requestID: "abc123")
               })
    }
}

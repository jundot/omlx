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

    // MARK: - Detail card

    /// The card is a fixed shape: whatever the record carries, it draws a
    /// bounded box of it and counts the rest. 20,000 lines repeated 20,000
    /// times is the shape that froze the pane before.
    func testTheCardDrawsARecordAndItsOccurrencesBounded() throws {
        let huge = try height(of: card(row(continuations: 20_000, occurrences: 20_000)))
        let smaller = try height(of: card(row(continuations: 400, occurrences: 400)))

        // Both records are past both caps, so the card draws the same shape
        // for them: a bigger record cannot buy a taller card.
        XCTAssertEqual(huge, smaller, accuracy: 1,
                       "the card grew with the record: \(huge)pt vs \(smaller)pt")
        // 454pt measured: header, the record box (its fixed height plus
        // padding), the note, and the occurrence box (its fixed height, its
        // heading and its note) — the type scale's own numbers, nothing that
        // scales with the record.
        XCTAssertLessThan(huge, 520,
                          "the card asked for \(huge)pt at \(width)pt wide")
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

    /// The record box itself: the fixed-height, scrolling box the card and the
    /// empty pane draw a record into.
    func testTheRecordBoxStaysAtItsFixedHeight() throws {
        let huge = try height(of: LogRecordBody(text: record(continuations: 20_000)
            .renderedFullMessage))
        let capped = try height(of: LogRecordBody(text: record(continuations: LogRenderLimits
            .continuationLines).renderedFullMessage))

        XCTAssertEqual(huge, capped, accuracy: 1,
                       "the box grew with the record: \(huge)pt vs \(capped)pt")
        // The box is the type scale's height plus its own padding and nothing
        // else: 8pt a side.
        XCTAssertLessThan(huge, LogTypeScale.recordBodyHeight + 32,
                          "the record box asked for \(huge)pt")
    }

    // MARK: - Row

    /// A row draws its continuation note, not its continuation: the first few
    /// lines are what the row shows when it is opened, and the record behind
    /// them is what the card is for. Both rows here are collapsed — the state
    /// the list is in until someone clicks.
    func testARowDrawsABoundedNumberOfContinuationLines() throws {
        let huge = try height(of: rowView(row(continuations: 20_000, occurrences: 1)))
        let tiny = try height(of: rowView(row(continuations: 4, occurrences: 1)))

        XCTAssertEqual(huge, tiny, accuracy: 1,
                       "the row grew with the record: \(huge)pt vs \(tiny)pt")
        // 43pt measured: one message line and the note under it.
        XCTAssertLessThan(huge, 120,
                          "the row asked for \(huge)pt at \(width)pt wide")
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

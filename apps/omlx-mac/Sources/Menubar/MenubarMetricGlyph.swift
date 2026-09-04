// Compact menubar readout for a metric status item: a vertical three-letter
// tag (LIV / AVG / ALL) followed by two aligned rows, "P: <prompt tok/s>"
// stacked over "T: <generation tok/s>". The readout is rasterized into an
// NSImage because a status-item button only renders bitmap content reliably,
// and every image assignment forces the menubar to re-composite the item —
// so callers gate assignments with `signature`, which captures the only
// inputs that can change the pixels.

import AppKit

enum MenubarMetricGlyph {
    /// Standard menubar content height in points.
    static let glyphHeight: CGFloat = 18

    /// Worst-case value string that reserves the value column, keeping the
    /// status item at a constant width while readings fluctuate.
    static let valueTemplate = "9999tk/s"

    private static let rowLabels = ("P:", "T:")

    /// nil / non-finite → "–" (unknown, distinct from an idle 0). Values are
    /// shown as whole tok/s, compacting to "12.3ktk/s" from 10k upward so
    /// the reserved column never overflows.
    static func formatTps(_ value: Double?) -> String {
        guard let value, value.isFinite else {
            return "–"
        }
        let clamped = max(0, value)
        if clamped >= 10_000 {
            return String(format: "%.1fktk/s", clamped / 1_000)
        }
        return "\(Int(clamped.rounded()))tk/s"
    }

    /// Cheap pixel-change detector: same signature ⇒ identical glyph, so the
    /// caller can skip the re-raster and the image reassignment entirely.
    static func signature(
        tag: String,
        promptValue: String,
        generationValue: String,
        darkMenubar: Bool
    ) -> String {
        "\(tag)|\(promptValue)|\(generationValue)|\(darkMenubar ? "dark" : "light")"
    }

    // MARK: - Shared tag column

    private struct TagColumn {
        let characters: [NSString]
        let attributes: [NSAttributedString.Key: Any]
        let width: CGFloat

        /// One character per equal vertical slot, top to bottom.
        func draw(height: CGFloat, originX: CGFloat = 0) {
            let slotHeight = height / CGFloat(max(1, characters.count))
            for (index, character) in characters.enumerated() {
                let size = character.size(withAttributes: attributes)
                let slotBottom = height - CGFloat(index + 1) * slotHeight
                character.draw(
                    at: NSPoint(
                        x: originX + (width - size.width) / 2,
                        y: slotBottom + (slotHeight - size.height) / 2
                    ),
                    withAttributes: attributes
                )
            }
        }
    }

    private static func tagColumn(_ tag: String, ink: NSColor) -> TagColumn {
        let attributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 6.5, weight: .bold),
            .foregroundColor: ink.withAlphaComponent(0.85),
        ]
        let characters = tag.map { String($0) as NSString }
        let width = characters
            .map { ceil($0.size(withAttributes: attributes).width) }
            .max() ?? 6
        return TagColumn(characters: characters, attributes: attributes, width: width)
    }

    // MARK: - Two-line readout (LIV / AVG / ALL)

    static func image(
        tag: String,
        promptValue: String,
        generationValue: String,
        darkMenubar: Bool
    ) -> NSImage {
        let ink: NSColor = darkMenubar ? .white : .black
        let column = tagColumn(tag, ink: ink)
        let rowFont = NSFont.systemFont(ofSize: 8.5, weight: .medium)
        let labelAttributes: [NSAttributedString.Key: Any] = [
            .font: rowFont,
            .foregroundColor: ink.withAlphaComponent(0.7),
        ]
        let valueAttributes: [NSAttributedString.Key: Any] = [
            .font: rowFont,
            .foregroundColor: ink.withAlphaComponent(0.95),
        ]

        let labels = (rowLabels.0 as NSString, rowLabels.1 as NSString)
        let labelWidth = ceil(max(
            labels.0.size(withAttributes: labelAttributes).width,
            labels.1.size(withAttributes: labelAttributes).width
        ))
        let values = (promptValue as NSString, generationValue as NSString)
        let valueWidth = ceil(
            [valueTemplate as NSString, values.0, values.1]
                .map { $0.size(withAttributes: valueAttributes).width }
                .max() ?? 0
        )

        let tagGap: CGFloat = 3
        let columnGap: CGFloat = 4
        let height = glyphHeight
        let width = ceil(column.width + tagGap + labelWidth + columnGap + valueWidth) + 2

        let image = NSImage(
            size: NSSize(width: width, height: height),
            flipped: false
        ) { _ in
            column.draw(height: height)

            // Two rows: labels pinned left, values right-aligned inside the
            // reserved column so digits line up and the width never moves.
            let rowLeft = column.width + tagGap
            let valueRight = rowLeft + labelWidth + columnGap + valueWidth
            let rowHeight = values.0.size(withAttributes: valueAttributes).height
            let halfHeight = height / 2
            let topBaseline = halfHeight + (halfHeight - rowHeight) / 2
            let bottomBaseline = (halfHeight - rowHeight) / 2

            labels.0.draw(
                at: NSPoint(x: rowLeft, y: topBaseline),
                withAttributes: labelAttributes
            )
            labels.1.draw(
                at: NSPoint(x: rowLeft, y: bottomBaseline),
                withAttributes: labelAttributes
            )
            values.0.draw(
                at: NSPoint(
                    x: valueRight - values.0.size(withAttributes: valueAttributes).width,
                    y: topBaseline
                ),
                withAttributes: valueAttributes
            )
            values.1.draw(
                at: NSPoint(
                    x: valueRight - values.1.size(withAttributes: valueAttributes).width,
                    y: bottomBaseline
                ),
                withAttributes: valueAttributes
            )
            return true
        }
        // Hand-inked for the actual menubar appearance — template tinting
        // would discard the two-tone label/value contrast.
        image.isTemplate = false
        return image
    }

    // MARK: - Vertical usage bar (CPU / GPU / MEM)

    /// Bar fill resolution: one step per Retina pixel row of the 18 pt
    /// glyph. The signature quantizes to this grid so sub-pixel usage
    /// jitter never forces a re-raster the eye couldn't see anyway.
    static let barLevelSteps = 36

    /// nil / non-finite → -1 (empty track).
    static func quantizedBarLevel(_ fraction: Double?) -> Int {
        guard let fraction, fraction.isFinite else {
            return -1
        }
        return Int((min(1, max(0, fraction)) * Double(barLevelSteps)).rounded())
    }

    /// One enabled host metric = one segment. Enabled segments share a
    /// single status item so the menubar doesn't scatter them across
    /// separate squares with system-managed gaps between.
    typealias BarSegment = (tag: String, fraction: Double?)

    static func barsSignature(segments: [BarSegment], darkMenubar: Bool) -> String {
        let body = segments
            .map { "\($0.tag):\(quantizedBarLevel($0.fraction))" }
            .joined(separator: "|")
        return "\(body)|\(darkMenubar ? "dark" : "light")"
    }

    /// Stacked tag + vertical bar per segment, drawn side by side into one
    /// image. Each bar fills bottom-up against a full-height track,
    /// 100% = full glyph height.
    static func barsImage(segments: [BarSegment], darkMenubar: Bool) -> NSImage {
        let ink: NSColor = darkMenubar ? .white : .black
        let columns = segments.map { tagColumn($0.tag, ink: ink) }
        let tagGap: CGFloat = 3
        let barWidth: CGFloat = 8
        let groupGap: CGFloat = 6
        let radius: CGFloat = 2
        let height = glyphHeight
        let segmentWidths = columns.map { ceil($0.width + tagGap + barWidth) }
        let width = segmentWidths.reduce(0, +)
            + groupGap * CGFloat(max(0, segments.count - 1)) + 2

        let image = NSImage(
            size: NSSize(width: width, height: height),
            flipped: false
        ) { _ in
            var originX: CGFloat = 0
            for (index, segment) in segments.enumerated() {
                let column = columns[index]
                column.draw(height: height, originX: originX)

                let barX = originX + column.width + tagGap
                ink.withAlphaComponent(0.18).setFill()
                NSBezierPath(
                    roundedRect: NSRect(x: barX, y: 0, width: barWidth, height: height),
                    xRadius: radius,
                    yRadius: radius
                ).fill()

                let level = quantizedBarLevel(segment.fraction)
                if level >= 0 {
                    let fillHeight = max(
                        1.5, height * CGFloat(level) / CGFloat(barLevelSteps)
                    )
                    ink.withAlphaComponent(0.9).setFill()
                    NSBezierPath(
                        roundedRect: NSRect(
                            x: barX, y: 0, width: barWidth, height: fillHeight
                        ),
                        xRadius: radius,
                        yRadius: radius
                    ).fill()
                }
                originX += segmentWidths[index] + groupGap
            }
            return true
        }
        image.isTemplate = false
        return image
    }
}

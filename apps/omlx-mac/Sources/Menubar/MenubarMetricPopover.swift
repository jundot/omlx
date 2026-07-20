// Dropdown content for a menubar metric item: what the reading is, the
// current PP/TG numbers, a rolling activity graph per series, and shortcuts
// to the Appearance settings pane and the web dashboard. Hosted in an
// NSPopover whose content controller only exists while the popover is open,
// so a closed dropdown costs zero SwiftUI updates.

import SwiftUI

struct MetricPopoverView: View {
    let kind: MenubarMetricsStore.Kind
    let store: MenubarMetricsStore
    let openSettings: () -> Void
    let openDashboard: () -> Void

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        let rates = store.rates[kind]
        let series = store.history[kind] ?? MenubarMetricsStore.Series()

        VStack(alignment: .leading, spacing: 6) {
            sectionHeader(kind.displayName)

            valueRow(
                label: "PP",
                value: Self.popoverTps(rates?.promptTps),
                tint: theme.blueDot
            )
            valueRow(
                label: "TG",
                value: Self.popoverTps(rates?.generationTps),
                tint: theme.greenDot
            )

            if let note = statusNote {
                Text(note)
                    .font(.omlxText(11))
                    .foregroundStyle(theme.textTertiary)
            }

            Divider().padding(.vertical, 2)

            sectionHeader(String(
                localized: "menubar.metric.activity",
                defaultValue: "Activity",
                comment: "Header above the throughput graphs in a menubar metric popover"
            ))

            graphCaption("PP tk/s")
            MetricSparkline(values: series.promptTps, color: theme.blueDot)
            graphCaption("TG tk/s")
            MetricSparkline(values: series.generationTps, color: theme.greenDot)

            Divider().padding(.vertical, 2)

            HStack(spacing: 8) {
                Button {
                    openSettings()
                } label: {
                    Label(
                        String(
                            localized: "menubar.metric.settings",
                            defaultValue: "Settings",
                            comment: "Button in a menubar metric popover that opens the Appearance settings pane"
                        ),
                        systemImage: "gearshape"
                    )
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.omlx(.normal, size: .small))

                Button {
                    openDashboard()
                } label: {
                    Label(
                        String(
                            localized: "menubar.metric.dashboard",
                            defaultValue: "Dashboard",
                            comment: "Button in a menubar metric popover that opens the web dashboard"
                        ),
                        systemImage: "globe"
                    )
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.omlx(.primary, size: .small))
                .disabled(!store.serverIsRunning)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .frame(width: 260)
    }

    private var statusNote: String? {
        if !store.serverIsRunning {
            return String(
                localized: "menubar.metric.server_off",
                defaultValue: "Server is off",
                comment: "Note in a menubar metric popover while the server is not running"
            )
        }
        if kind == .live, store.rates[.live] == nil {
            // The server answers but /admin/api/activity does not — the
            // admin API key is missing or rejected.
            return String(
                localized: "menubar.metric.live_unavailable",
                defaultValue: "Set an API key to enable live activity",
                comment: "Note in the LIV popover when live stats need admin authentication"
            )
        }
        return nil
    }

    private func sectionHeader(_ title: String) -> some View {
        Text(title.uppercased())
            .font(.omlxText(10, weight: .bold))
            .kerning(1)
            .foregroundStyle(theme.accent)
            .frame(maxWidth: .infinity, alignment: .center)
    }

    private func valueRow(label: String, value: String, tint: Color) -> some View {
        HStack {
            Circle()
                .fill(tint)
                .frame(width: 6, height: 6)
            Text(label)
                .font(.omlxText(12, weight: .medium))
                .foregroundStyle(theme.textSecondary)
            Spacer()
            Text(value)
                .font(.omlxMono(12, weight: .medium))
                .foregroundStyle(theme.text)
        }
    }

    private func graphCaption(_ text: String) -> some View {
        Text(text)
            .font(.omlxMono(9))
            .foregroundStyle(theme.textTertiary)
    }

    /// Popover readout: one decimal below 100 tk/s, whole numbers above,
    /// "–" for unknown.
    static func popoverTps(_ value: Double?) -> String {
        guard let value, value.isFinite else {
            return "–"
        }
        let clamped = max(0, value)
        if clamped < 100 {
            return String(format: "%.1f tk/s", clamped)
        }
        return "\(Int(clamped.rounded())) tk/s"
    }
}

extension MenubarMetricsStore.Kind {
    var displayName: String {
        switch self {
        case .live:
            return String(
                localized: "menubar.metric.title.live",
                defaultValue: "Live Activity",
                comment: "Popover title for the live-throughput menubar item"
            )
        case .average:
            return String(
                localized: "menubar.metric.title.average",
                defaultValue: "Average Session",
                comment: "Popover title for the session-average menubar item"
            )
        case .alltime:
            return String(
                localized: "menubar.metric.title.alltime",
                defaultValue: "All Time",
                comment: "Popover title for the all-time-average menubar item"
            )
        }
    }
}

/// Rolling line graph for one throughput series. Drawn with a single Canvas
/// pass (no charting framework): the per-tick redraw is one path build, and
/// the trace is decorative — the numbers above carry the accessible value —
/// so it opts out of the accessibility tree entirely.
struct MetricSparkline: View {
    let values: [Double]
    let color: Color
    var height: CGFloat = 22
    /// Fixed Y range for bounded series (e.g. 0...1 usage fractions).
    /// nil auto-scales to the data so small variations stay visible.
    var domain: ClosedRange<Double>? = nil

    var body: some View {
        Canvas { context, size in
            guard values.count > 1, size.width > 0, size.height > 0 else {
                return
            }
            let low = domain?.lowerBound ?? (values.min() ?? 0)
            let high = domain?.upperBound ?? (values.max() ?? 0)
            // Without a fixed domain, 5% headroom keeps peaks off the top
            // edge; a flat series draws mid-height instead of hugging the
            // floor.
            let span = domain.map { $0.upperBound - $0.lowerBound }
                ?? (high - low) * 1.05
            let isFlat = span <= .ulpOfOne
            let stepX = size.width / CGFloat(values.count - 1)

            var trace = Path()
            for (index, value) in values.enumerated() {
                let normalized = isFlat ? 0.5 : (value - low) / span
                let point = CGPoint(
                    x: CGFloat(index) * stepX,
                    y: size.height * (1 - CGFloat(normalized))
                )
                if index == 0 {
                    trace.move(to: point)
                } else {
                    trace.addLine(to: point)
                }
            }

            var fillArea = trace
            fillArea.addLine(to: CGPoint(x: size.width, y: size.height))
            fillArea.addLine(to: CGPoint(x: 0, y: size.height))
            fillArea.closeSubpath()
            context.fill(
                fillArea,
                with: .linearGradient(
                    Gradient(colors: [color.opacity(0.25), .clear]),
                    startPoint: .zero,
                    endPoint: CGPoint(x: 0, y: size.height)
                )
            )
            context.stroke(
                trace,
                with: .color(color),
                style: StrokeStyle(lineWidth: 1.2, lineJoin: .round)
            )
        }
        .frame(height: height)
        .accessibilityHidden(true)
    }
}

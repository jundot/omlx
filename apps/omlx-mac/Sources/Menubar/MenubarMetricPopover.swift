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

    var body: some View {
        let snapshot = store.snapshot(for: kind)
        let series = store.history[kind] ?? MenubarMetricsStore.Series()

        VStack(alignment: .leading, spacing: 6) {
            ServingStatsScopeView(
                kind: kind,
                snapshot: snapshot,
                series: series,
                serverIsRunning: store.serverIsRunning
            )

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
        .padding(.horizontal, MenubarStatsPanelLayout.horizontalInset)
        .padding(.vertical, MenubarStatsPanelLayout.verticalInset)
        .frame(width: MenubarStatsPanelLayout.width)
    }
}

/// Read-only, value-driven presentation for a detailed Serving Stats scope.
/// It keeps the metrics, activity history, and terminal detail reusable while
/// preserving the full popover composition for LIV, AVG, and ALL.
struct ServingStatsScopeView: View {
    let snapshot: ServingStatsSnapshot
    let series: MenubarMetricsStore.Series
    let serverIsRunning: Bool
    let composition: ServingStatsPresentation.ScopeComposition
    let showsStatusNote: Bool

    @Environment(\.omlxTheme) private var theme

    init(
        kind: MenubarMetricsStore.Kind,
        snapshot: ServingStatsSnapshot,
        series: MenubarMetricsStore.Series,
        serverIsRunning: Bool
    ) {
        self.init(
            snapshot: snapshot,
            series: series,
            serverIsRunning: serverIsRunning,
            composition: ServingStatsPresentation.popoverScope(for: kind),
            showsStatusNote: true
        )
    }

    init(
        snapshot: ServingStatsSnapshot,
        series: MenubarMetricsStore.Series,
        serverIsRunning: Bool,
        composition: ServingStatsPresentation.ScopeComposition,
        showsStatusNote: Bool = false
    ) {
        self.snapshot = snapshot
        self.series = series
        self.serverIsRunning = serverIsRunning
        self.composition = composition
        self.showsStatusNote = showsStatusNote
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            MenubarStatsPanelHeader(title: composition.kind.displayName)

            if composition.showsThroughput {
                ServingStatsThroughputValues(snapshot: snapshot)
            }

            if showsStatusNote, let note = ServingStatsPresentation.statusNote(
                for: composition.kind,
                snapshot: snapshot,
                serverIsRunning: serverIsRunning
            ) {
                Text(note)
                    .font(.omlxText(11))
                    .foregroundStyle(theme.textTertiary)
            }

            if composition.activityGraphCount > 0 {
                ServingStatsActivityGraphs(series: series)
            }

            Divider().padding(.vertical, 2)
            switch composition.terminalDetail {
            case .currentRequests:
                CurrentRequestsView(
                    activity: snapshot.liveActivity,
                    showsHeader: composition.showsTerminalDetailHeader
                )
            case .scalarMetrics:
                ScalarMetricsView(
                    snapshot: snapshot,
                    showsHeader: composition.showsTerminalDetailHeader
                )
            }
        }
    }
}

/// The compact main-menu overview reuses the serving leaves without the
/// standalone popovers' activity graphs or unavailable-state notes.
struct ServingStatsMenuPanel: View {
    let live: ServingStatsSnapshot
    let average: ServingStatsSnapshot
    let alltime: ServingStatsSnapshot
    let liveSeries: MenubarMetricsStore.Series
    let averageSeries: MenubarMetricsStore.Series
    let alltimeSeries: MenubarMetricsStore.Series
    let serverIsRunning: Bool

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        Group {
            switch ServingStatsPresentation.menuComposition(serverIsRunning: serverIsRunning) {
            case .serverOff:
                Text(String(
                    localized: "menubar.stats.server_off",
                    defaultValue: "Server is off",
                    comment: "Disabled placeholder in the Serving Stats submenu when the server isn't running"
                ))
                .font(.omlxText(11))
                .foregroundStyle(theme.textTertiary)
                .padding(.vertical, 14)
            case .sections(let sections):
                VStack(alignment: .leading, spacing: 0) {
                    ForEach(sections) { section in
                        compactScope(section)
                        if section.kind != .alltime {
                            SectionRule()
                        }
                    }
                }
            }
        }
        .frame(width: MenubarStatsPanelLayout.contentWidth, alignment: .leading)
        .padding(.horizontal, MenubarStatsPanelLayout.horizontalInset)
        .padding(.vertical, MenubarStatsPanelLayout.verticalInset)
        .frame(width: MenubarStatsPanelLayout.width)
    }

    private func compactScope(
        _ section: ServingStatsPresentation.ScopeComposition
    ) -> some View {
        ServingStatsScopeView(
            snapshot: snapshot(for: section.kind),
            series: series(for: section.kind),
            serverIsRunning: serverIsRunning,
            composition: section,
            showsStatusNote: false
        )
    }

    private func snapshot(for kind: MenubarMetricsStore.Kind) -> ServingStatsSnapshot {
        switch kind {
        case .live:
            return live
        case .average:
            return average
        case .alltime:
            return alltime
        }
    }

    private func series(for kind: MenubarMetricsStore.Kind) -> MenubarMetricsStore.Series {
        switch kind {
        case .live:
            return liveSeries
        case .average:
            return averageSeries
        case .alltime:
            return alltimeSeries
        }
    }
}

enum ServingStatsPresentation {
    enum TerminalDetail: Equatable {
        case currentRequests
        case scalarMetrics
    }

    struct ScopeComposition: Equatable, Identifiable {
        let kind: MenubarMetricsStore.Kind
        let showsThroughput: Bool
        let activityGraphCount: Int
        let terminalDetail: TerminalDetail
        let showsTerminalDetailHeader: Bool

        init(
            kind: MenubarMetricsStore.Kind,
            showsThroughput: Bool,
            activityGraphCount: Int,
            terminalDetail: TerminalDetail,
            showsTerminalDetailHeader: Bool = false
        ) {
            self.kind = kind
            self.showsThroughput = showsThroughput
            self.activityGraphCount = activityGraphCount
            self.terminalDetail = terminalDetail
            self.showsTerminalDetailHeader = showsTerminalDetailHeader
        }

        var id: MenubarMetricsStore.Kind { kind }
    }

    enum MenuComposition: Equatable {
        case serverOff
        case sections([ScopeComposition])
    }

    static func popoverScope(for kind: MenubarMetricsStore.Kind) -> ScopeComposition {
        ScopeComposition(
            kind: kind,
            showsThroughput: true,
            activityGraphCount: 2,
            terminalDetail: kind == .live ? .currentRequests : .scalarMetrics,
            showsTerminalDetailHeader: kind == .live
        )
    }

    static func menuComposition(serverIsRunning: Bool) -> MenuComposition {
        guard serverIsRunning else {
            return .serverOff
        }
        return .sections([
            ScopeComposition(
                kind: .live,
                showsThroughput: true,
                activityGraphCount: 2,
                terminalDetail: .currentRequests,
                showsTerminalDetailHeader: true
            ),
            ScopeComposition(
                kind: .average,
                showsThroughput: true,
                activityGraphCount: 0,
                terminalDetail: .scalarMetrics
            ),
            ScopeComposition(
                kind: .alltime,
                showsThroughput: true,
                activityGraphCount: 0,
                terminalDetail: .scalarMetrics
            )
        ])
    }

    static func statusNote(
        for kind: MenubarMetricsStore.Kind,
        snapshot: ServingStatsSnapshot,
        serverIsRunning: Bool
    ) -> String? {
        if !serverIsRunning {
            return String(
                localized: "menubar.metric.server_off",
                defaultValue: "Server is off",
                comment: "Note in a menubar metric popover while the server is not running"
            )
        }
        if kind == .live, snapshot.isUnavailable {
            return String(
                localized: "menubar.metric.live_unavailable",
                defaultValue: "Set an API key to enable live activity",
                comment: "Note in the LIV popover when live stats need admin authentication"
            )
        }
        if kind != .live, snapshot.isUnavailable {
            return String(
                localized: "menubar.metric.unavailable",
                defaultValue: "Serving stats are unavailable",
                comment: "Note in an AVG or ALL popover when its scoped stats are unavailable"
            )
        }
        return nil
    }

    static func popoverTps(_ value: Double?) -> String {
        guard let value, value.isFinite else {
            return "–"
        }
        return "\(ActivityFormat.rate(value)) tk/s"
    }
}

struct ServingStatsActivityGraphs: View {
    let series: MenubarMetricsStore.Series

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            MenubarStatsPanelCaption(text: "PP tk/s")
            MetricSparkline(values: series.promptTps, color: theme.blueDot)
            MenubarStatsPanelCaption(text: "TG tk/s")
            MetricSparkline(values: series.generationTps, color: theme.greenDot)
        }
    }
}

struct ServingStatsThroughputValues: View {
    let snapshot: ServingStatsSnapshot
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        HStack(spacing: 8) {
            value(
                label: "PP",
                value: ServingStatsPresentation.popoverTps(snapshot.rates?.promptTps),
                tint: theme.blueDot
            )
            value(
                label: "TG",
                value: ServingStatsPresentation.popoverTps(snapshot.rates?.generationTps),
                tint: theme.greenDot
            )
        }
    }

    private func value(label: String, value: String, tint: Color) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 4) {
                Circle()
                    .fill(tint)
                    .frame(width: 6, height: 6)
                Text(label)
                    .font(.omlxText(12))
                    .foregroundStyle(theme.textSecondary)
            }
            Text(value)
                .font(.omlxMono(12))
                .foregroundStyle(theme.text)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct ScalarMetricsView: View {
    let snapshot: ServingStatsSnapshot
    let showsHeader: Bool

    init(snapshot: ServingStatsSnapshot, showsHeader: Bool = true) {
        self.snapshot = snapshot
        self.showsHeader = showsHeader
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            if showsHeader {
                MenubarStatsPanelHeader(title: String(
                    localized: "menubar.metric.metrics",
                    defaultValue: "Serving Stats",
                    comment: "Header above scalar counters in an AVG or ALL serving metric popover"
                ))
            }

            VStack(alignment: .leading, spacing: 4) {
                MenubarStatsValueRow(
                    label: String(
                        localized: "menubar.stats.total_tokens",
                        defaultValue: "Total Tokens Processed",
                        comment: "Stats row label for total tokens processed"
                    ),
                    value: Self.compact(snapshot.totalPromptTokens)
                )
                MenubarStatsValueRow(
                    label: String(
                        localized: "menubar.stats.cached_tokens",
                        defaultValue: "Cached Tokens",
                        comment: "Stats row label for cached tokens count"
                    ),
                    value: Self.compact(snapshot.totalCachedTokens)
                )
                MenubarStatsValueRow(
                    label: String(
                        localized: "menubar.stats.cache_efficiency",
                        defaultValue: "Cache Efficiency",
                        comment: "Stats row label for the cache efficiency percentage"
                    ),
                    value: Self.percent(snapshot.cacheEfficiency)
                )
                MenubarStatsValueRow(
                    label: String(
                        localized: "menubar.stats.total_requests",
                        defaultValue: "Total Requests",
                        comment: "Stats row label for total request count"
                    ),
                    value: Self.compact(snapshot.totalRequests)
                )
            }
        }
    }

    private static func compact(_ value: Int?) -> String {
        guard let value else { return "–" }
        let clamped = max(0, value)
        if clamped >= 1_000_000 {
            return String(format: clamped >= 10_000_000 ? "%.0fM" : "%.1fM", Double(clamped) / 1_000_000)
        }
        if clamped >= 1_000 {
            return String(format: clamped >= 10_000 ? "%.0fk" : "%.1fk", Double(clamped) / 1_000)
        }
        return "\(clamped)"
    }

    private static func percent(_ value: Double?) -> String {
        guard let value, value.isFinite else { return "–" }
        return String(format: "%.1f%%", max(0, value))
    }
}

struct CurrentRequestsView: View {
    let activity: MenubarStatsPoller.Stats.LiveActivity?
    let showsHeader: Bool
    @Environment(\.omlxTheme) private var theme

    init(
        activity: MenubarStatsPoller.Stats.LiveActivity?,
        showsHeader: Bool = true
    ) {
        self.activity = activity
        self.showsHeader = showsHeader
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            if showsHeader {
                MenubarStatsPanelHeader(title: String(
                    localized: "menubar.stats.current_requests",
                    defaultValue: "Current Requests",
                    comment: "Section header inside Serving Stats for active and queued requests"
                ))
            }

            if let activity {
                if activity.isIdle {
                    emptyState
                } else {
                    ForEach(activity.groups) { group in
                        Text(group.modelID)
                            .font(.omlxMono(12, weight: .medium))
                            .foregroundStyle(theme.textSecondary)
                            .lineLimit(1)
                        ForEach(group.requests) { request in
                            requestRow(request)
                        }
                    }
                    if activity.queuedRequestCount > 0 {
                        statusRow(String(
                            localized: "menubar.stats.queued_requests",
                            defaultValue: "\(activity.queuedRequestCount) queued requests",
                            comment: "Current Requests row showing how many requests have not started; placeholder is the count"
                        ))
                    }
                    if activity.hiddenRequestCount > 0 {
                        statusRow(String(
                            localized: "menubar.stats.more_requests",
                            defaultValue: "\(activity.hiddenRequestCount) more requests",
                            comment: "Current Requests row showing active requests omitted from the capped list; placeholder is the count"
                        ))
                    }
                }
            } else {
                Text(String(
                    localized: "menubar.stats.loading",
                    defaultValue: "Loading stats…",
                    comment: "Disabled placeholder shown while stats are loading"
                ))
                .font(.omlxText(11))
                .foregroundStyle(theme.textTertiary)
            }
        }
    }

    private var emptyState: some View {
        Text(String(
            localized: "menubar.stats.no_current_requests",
            defaultValue: "No active requests",
            comment: "Disabled empty state in Serving Stats when no requests are active or queued"
        ))
        .font(.omlxText(11))
        .foregroundStyle(theme.textTertiary)
    }

    private func requestRow(_ request: MenubarStatsPoller.Stats.LiveActivity.Request) -> some View {
        HStack(spacing: 6) {
            if let marker = CurrentRequestMarker.marker(for: request.kind) {
                Circle()
                    .fill(marker.color(in: theme))
                    .frame(width: 6, height: 6)
                Text(marker.abbreviation)
                    .font(.omlxMono(12, weight: .medium))
                    .foregroundStyle(marker.color(in: theme))
            }
            Text(request.title)
                .font(.omlxMono(12))
                .foregroundStyle(theme.text)
                .lineLimit(1)
            Spacer(minLength: 4)
            Text(request.detail)
                .font(.omlxMono(12))
                .foregroundStyle(theme.textSecondary)
                .lineLimit(1)
        }
    }

    private func statusRow(_ text: String) -> some View {
        Text(text)
            .font(.omlxText(11))
            .foregroundStyle(theme.textTertiary)
    }
}

enum CurrentRequestMarker: Equatable {
    case prefill
    case generating

    static func marker(
        for kind: MenubarStatsPoller.Stats.LiveActivity.Request.Kind
    ) -> CurrentRequestMarker? {
        switch kind {
        case .prefill:
            return .prefill
        case .generating:
            return .generating
        case .nonStreaming:
            return nil
        }
    }

    var abbreviation: String {
        switch self {
        case .prefill:
            return "PP"
        case .generating:
            return "TG"
        }
    }

    func color(in theme: OMLXTheme) -> Color {
        switch self {
        case .prefill:
            return theme.blueDot
        case .generating:
            return theme.greenDot
        }
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

            // Inset the plot range by half the stroke width so a series
            // pinned to the floor (0%) or ceiling doesn't center its stroke
            // on the canvas edge and render half-clipped.
            let lineWidth: CGFloat = 1.2
            let plotTop = lineWidth / 2
            let plotHeight = size.height - lineWidth

            var trace = Path()
            for (index, value) in values.enumerated() {
                let normalized = isFlat ? 0.5 : (value - low) / span
                let point = CGPoint(
                    x: CGFloat(index) * stepX,
                    y: plotTop + plotHeight * (1 - CGFloat(normalized))
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
                style: StrokeStyle(lineWidth: lineWidth, lineJoin: .round)
            )
        }
        .frame(height: height)
        .accessibilityHidden(true)
    }
}

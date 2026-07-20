// The three panels inside the menubar's System Stats submenu: CPU, GPU,
// and Memory. Each is a value-driven SwiftUI view fed a SystemStatsSnapshot
// by MenubarController's sampling timer, hosted in an NSMenuItem custom
// view. Value-driven (no observation) so updates render synchronously even
// while the menu run loop is in tracking mode.

import SwiftUI

private let panelWidth: CGFloat = 270

// MARK: - CPU

struct CPUStatsPanel: View {
    let snapshot: SystemStatsSnapshot
    let refreshInterval: TimeInterval

    @Environment(\.omlxTheme) private var theme

    private var eColor: Color { Color(nsColor: .systemOrange) }
    private var pColor: Color { Color(nsColor: .systemBlue) }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            StatsPanelHeader(title: "CPU")
            UsageBarRow(
                label: String(localized: "menubar.system.e_cores",
                              defaultValue: "E-cores",
                              comment: "CPU panel row label for efficiency-core usage"),
                value: StatsFormat.percent(snapshot.eCoreUsage),
                fraction: snapshot.eCoreUsage ?? 0,
                color: eColor
            )
            UsageBarRow(
                label: String(localized: "menubar.system.p_cores",
                              defaultValue: "P-cores",
                              comment: "CPU panel row label for performance-core usage"),
                value: StatsFormat.percent(snapshot.pCoreUsage),
                fraction: snapshot.pCoreUsage ?? 0,
                color: pColor
            )
            StatsPanelCaption(
                text: String(localized: "menubar.system.cpu_caption",
                             defaultValue: "E (amber) / P (blue) usage",
                             comment: "CPU panel caption under the usage bars"),
                window: StatsFormat.window(sampleCount: snapshot.eHistory.count,
                                           interval: refreshInterval)
            )
            ZStack {
                MetricSparkline(values: snapshot.pHistory, color: pColor,
                                height: 26, domain: 0...1)
                MetricSparkline(values: snapshot.eHistory, color: eColor,
                                height: 26, domain: 0...1)
            }
            Divider()
            StatsValueRow(
                label: String(localized: "menubar.system.thermal",
                              defaultValue: "Thermal",
                              comment: "CPU panel row label for the system thermal state"),
                value: SystemMetricsPoller.label(
                    for: SystemMetricsPoller.severity(for: snapshot.thermalState)
                )
            )
            StatsValueRow(
                label: String(localized: "menubar.system.load_avg",
                              defaultValue: "Load avg",
                              comment: "CPU panel row label for the 1/5/15-minute load averages"),
                value: SystemStatsSampler.formatLoadAverages(snapshot.loadAverages)
            )
            StatsValueRow(
                label: String(localized: "menubar.system.uptime",
                              defaultValue: "Uptime",
                              comment: "CPU panel row label for system uptime"),
                value: SystemStatsSampler.formatUptime(snapshot.uptimeSeconds)
            )
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .frame(width: panelWidth)
    }
}

// MARK: - GPU

struct GPUStatsPanel: View {
    let snapshot: SystemStatsSnapshot
    let refreshInterval: TimeInterval

    @Environment(\.omlxTheme) private var theme

    private var gpuColor: Color { Color(nsColor: .systemGreen) }
    private var gpuMemColor: Color { Color(nsColor: .systemCyan) }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            StatsPanelHeader(title: "GPU")
            UsageBarRow(
                label: "GPU",
                value: StatsFormat.percent(snapshot.gpuUsage),
                fraction: snapshot.gpuUsage ?? 0,
                color: gpuColor
            )
            UsageBarRow(
                label: String(localized: "menubar.system.gpu_memory",
                              defaultValue: "GPU memory",
                              comment: "GPU panel row label for accelerator in-use memory"),
                value: snapshot.gpuMemoryInUseBytes.map {
                    String(localized: "menubar.system.in_use",
                           defaultValue: "\(SystemStatsSampler.formatBytes($0)) in use",
                           comment: "GPU memory value; placeholder is a byte size like 2.60 GB")
                } ?? "–",
                fraction: gpuMemoryFraction,
                color: gpuMemColor
            )
            StatsPanelCaption(
                text: String(localized: "menubar.system.gpu_caption",
                             defaultValue: "GPU (green) / GPU mem (cyan)",
                             comment: "GPU panel caption under the usage bars"),
                window: StatsFormat.window(sampleCount: snapshot.gpuHistory.count,
                                           interval: refreshInterval)
            )
            MetricSparkline(values: snapshot.gpuHistory, color: gpuColor,
                            height: 26, domain: 0...1)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .frame(width: panelWidth)
    }

    private var gpuMemoryFraction: Double {
        guard let inUse = snapshot.gpuMemoryInUseBytes,
              snapshot.memory.totalBytes > 0
        else {
            return 0
        }
        return min(1, Double(inUse) / Double(snapshot.memory.totalBytes))
    }
}

// MARK: - Memory

struct MemoryStatsPanel: View {
    let snapshot: SystemStatsSnapshot

    @Environment(\.omlxTheme) private var theme

    private var wiredColor: Color { Color(nsColor: .systemBlue) }
    private var activeColor: Color { Color(nsColor: .systemRed) }
    private var compressedColor: Color { Color(nsColor: .systemPurple) }

    var body: some View {
        let memory = snapshot.memory
        VStack(alignment: .leading, spacing: 6) {
            StatsPanelHeader(title: String(
                localized: "menubar.system.memory_title",
                defaultValue: "Memory",
                comment: "Title of the Memory panel in the System Stats submenu"
            ))
            HStack {
                Text("\(SystemMetricsPoller.formatBytesAsGB(memory.usedBytes)) / \(StatsFormat.wholeGB(memory.totalBytes)) GB")
                    .font(.omlxMono(12, weight: .semibold))
                    .foregroundStyle(theme.text)
                Spacer()
                Text(StatsFormat.percent(memory.usedFraction))
                    .font(.omlxMono(12))
                    .foregroundStyle(theme.textSecondary)
            }
            SegmentedUsageBar(
                total: memory.totalBytes,
                segments: [
                    (memory.wiredBytes, wiredColor),
                    (memory.activeBytes, activeColor),
                    (memory.compressedBytes, compressedColor),
                ]
            )
            legendRow(color: wiredColor,
                      label: String(localized: "menubar.system.wired",
                                    defaultValue: "Wired",
                                    comment: "Memory panel legend row for wired memory"),
                      bytes: memory.wiredBytes)
            legendRow(color: activeColor,
                      label: String(localized: "menubar.system.active",
                                    defaultValue: "Active",
                                    comment: "Memory panel legend row for active memory"),
                      bytes: memory.activeBytes)
            legendRow(color: compressedColor,
                      label: String(localized: "menubar.system.compressed",
                                    defaultValue: "Compressed",
                                    comment: "Memory panel legend row for compressed memory"),
                      bytes: memory.compressedBytes)
            legendRow(color: theme.textTertiary.opacity(0.4),
                      label: String(localized: "menubar.system.free",
                                    defaultValue: "Free",
                                    comment: "Memory panel legend row for free memory"),
                      bytes: memory.freeBytes)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .frame(width: panelWidth)
    }

    private func legendRow(color: Color, label: String, bytes: UInt64) -> some View {
        HStack(spacing: 6) {
            RoundedRectangle(cornerRadius: 2)
                .fill(color)
                .frame(width: 8, height: 8)
            Text(label)
                .font(.omlxText(11.5))
                .foregroundStyle(theme.textSecondary)
            Spacer()
            Text(SystemStatsSampler.formatBytes(bytes))
                .font(.omlxMono(11.5))
                .foregroundStyle(theme.text)
        }
    }
}

// MARK: - Combined popover stack

/// The enabled panels stacked into one popover for the combined CPU/GPU/MEM
/// status item, separated like the System Stats submenu.
struct SystemStatsPanelStack: View {
    let kinds: [SystemStatKind]
    let snapshot: SystemStatsSnapshot
    let refreshInterval: TimeInterval

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ForEach(Array(kinds.enumerated()), id: \.element) { index, kind in
                if index > 0 {
                    SectionRule()
                }
                panel(for: kind)
            }
        }
        // Pin the stack to the panel width. Without this, any width-flexible
        // child (a divider) makes the hosting controller's fitting width
        // echo whatever the popover proposed — and a popover proposes its
        // default ~320 pt frame on first open, which then sticks, leaving a
        // dead right margin whenever two or more panels are stacked.
        .frame(width: panelWidth)
    }

    @ViewBuilder
    private func panel(for kind: SystemStatKind) -> some View {
        switch kind {
        case .cpu:
            CPUStatsPanel(snapshot: snapshot, refreshInterval: refreshInterval)
        case .gpu:
            GPUStatsPanel(snapshot: snapshot, refreshInterval: refreshInterval)
        case .memory:
            MemoryStatsPanel(snapshot: snapshot)
        }
    }
}

// MARK: - Shared pieces

/// Panel-width section rule (14 pt inset each side, like a padded Divider).
/// Both frames are fixed widths on purpose: a width-flexible child at the
/// stack level would let the hosting controller's fitting width track the
/// popover's proposed frame instead of the panel width.
struct SectionRule: View {
    var containerWidth: CGFloat = panelWidth
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        Rectangle()
            .fill(theme.rowSep)
            .frame(width: containerWidth - 28, height: 1)
            .frame(width: containerWidth)
            .padding(.vertical, 4)
            .accessibilityHidden(true)
    }
}

private struct StatsPanelHeader: View {
    let title: String
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        Text(title.uppercased())
            .font(.omlxText(10, weight: .bold))
            .kerning(1)
            .foregroundStyle(theme.accent)
            .frame(maxWidth: .infinity, alignment: .center)
    }
}

private struct StatsPanelCaption: View {
    let text: String
    let window: String
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        Text(window.isEmpty ? text : "\(text) · \(window)")
            .font(.omlxText(9.5))
            .foregroundStyle(theme.textTertiary)
    }
}

private struct UsageBarRow: View {
    let label: String
    let value: String
    let fraction: Double
    let color: Color
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text(label)
                    .font(.omlxText(12, weight: .medium))
                    .foregroundStyle(theme.text)
                Spacer()
                Text(value)
                    .font(.omlxMono(12))
                    .foregroundStyle(theme.text)
            }
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(theme.textTertiary.opacity(0.18))
                    Capsule()
                        .fill(color)
                        .frame(width: max(
                            4, geo.size.width * min(1, max(0, fraction))
                        ))
                }
            }
            .frame(height: 5)
        }
        .accessibilityElement(children: .combine)
    }
}

private struct StatsValueRow: View {
    let label: String
    let value: String
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        HStack {
            Text(label)
                .font(.omlxText(12))
                .foregroundStyle(theme.textSecondary)
            Spacer()
            Text(value)
                .font(.omlxMono(12))
                .foregroundStyle(theme.text)
        }
    }
}

private struct SegmentedUsageBar: View {
    let total: UInt64
    let segments: [(bytes: UInt64, color: Color)]
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        GeometryReader { geo in
            HStack(spacing: 1) {
                ForEach(Array(segments.enumerated()), id: \.offset) { _, segment in
                    let fraction = total > 0
                        ? Double(segment.bytes) / Double(total)
                        : 0
                    if fraction > 0.001 {
                        Rectangle()
                            .fill(segment.color)
                            .frame(width: max(2, geo.size.width * fraction))
                    }
                }
                Spacer(minLength: 0)
            }
            .background(theme.textTertiary.opacity(0.18))
            .clipShape(RoundedRectangle(cornerRadius: 4))
        }
        .frame(height: 9)
        .accessibilityHidden(true)
    }
}

// MARK: - Formatting helpers (pure, unit-tested)

enum StatsFormat {
    /// 0.153 → "15%"; nil → "–"
    static func percent(_ fraction: Double?) -> String {
        guard let fraction, fraction.isFinite else { return "–" }
        return "\(Int((min(1, max(0, fraction)) * 100).rounded()))%"
    }

    /// 512_000_000_000 → "512" (whole decimal GB for machine totals)
    static func wholeGB(_ bytes: UInt64) -> String {
        "\(Int((Double(bytes) / 1_000_000_000).rounded()))"
    }

    /// History window caption: 60 samples at 1 s → "60s"; at 0.5 s → "30s".
    /// Empty history → empty string (caption shows no window).
    static func window(sampleCount: Int, interval: TimeInterval) -> String {
        guard sampleCount > 1, interval > 0 else { return "" }
        return "\(Int((Double(sampleCount) * interval).rounded()))s"
    }
}

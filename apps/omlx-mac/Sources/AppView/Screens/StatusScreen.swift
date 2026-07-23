// PR 7 — Status screen. Lays out:
//   • ServerHeroCard (the same card the Server screen mounts; defined in
//     ServerScreen.swift)
//   • Serving Stats — prefill/cache tiles + average speed from /admin/api/stats
//   • System — slice of /admin/api/global-settings + uptime from /api/stats
//   • Updates — release check status + auto-check/auto-notify prefs
//   • Active Now — active_models slice from /api/stats
//
// Polling is on-screen-only: a 5s timer ticks while the view is visible.

import SwiftUI

struct StatusScreen: View {
    @Environment(AppServices.self) private var services
    @State private var vm = StatusScreenVM()

    @State private var showingClearStatsConfirm = false
    @State private var showingClearSsdCacheConfirm = false
    @State private var showingClearHotCacheConfirm = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Same hero card the Server screen uses (omlx-screens.jsx:75
            // and :833 reuse `ServerHero`). Pass no VM — Status doesn't
            // expose port/host editors, so Restart just bounces the cached
            // endpoint.
            ServerHeroCard()

            SectionHeader(String(localized: "status.session_stats",
                                  defaultValue: "Serving Stats",
                                  comment: "Section header for the serving stats tiles")) {
                HStack(spacing: 10) {
                    Segmented(selection: $vm.scope, options: [
                        ("session", String(localized: "status.scope.session",
                                            defaultValue: "Session",
                                            comment: "Segmented control option: session-scope stats")),
                        ("alltime", String(localized: "status.scope.alltime",
                                            defaultValue: "All Time",
                                            comment: "Segmented control option: all-time stats")),
                    ])
                    Button {
                        showingClearStatsConfirm = true
                    } label: {
                        Image(systemName: "trash")
                            .font(.system(size: 11, weight: .semibold))
                    }
                    .buttonStyle(.omlx(.plain, size: .small))
                    .help(vm.scope == "alltime"
                          ? String(localized: "status.clear.alltime.help",
                                   defaultValue: "Clear all-time stats",
                                   comment: "Tooltip on the clear button when all-time scope is selected")
                          : String(localized: "status.clear.session.help",
                                   defaultValue: "Clear session stats",
                                   comment: "Tooltip on the clear button when session scope is selected"))
                    .disabled(vm.stats == nil)
                }
            }
            StatTilesRow(stats: vm.stats)

            SectionHeader(String(localized: "status.speed.section_label",
                                  defaultValue: "Average Speed",
                                  comment: "Header label above average serving speed metrics"))
            AverageSpeedTilesRow(stats: vm.stats)

            SectionHeader(String(localized: "status.section.active_now",
                                  defaultValue: "Active Now",
                                  comment: "Section header for the currently active models list"))
            ActiveNowList(models: vm.stats?.activeModels.models ?? [])

            SectionHeader(String(localized: "status.section.system",
                                  defaultValue: "System",
                                  comment: "Section header for system status rows"),
                          subtitle: vm.systemSubtitle)
            ListGroup {
                Row(label: String(localized: "status.row.gpu_memory",
                                  defaultValue: "GPU Wired Memory",
                                  comment: "Row label for GPU wired memory")) {
                    GpuMemoryTrailing(stats: vm.stats)
                }
                Row(label: String(localized: "status.row.system_ram",
                                  defaultValue: "System RAM",
                                  comment: "Row label for system RAM")) {
                    SystemRamTrailing(metrics: vm.metrics)
                }
                Row(
                    label: String(localized: "status.row.gpu_utilization",
                                  defaultValue: "GPU Utilization",
                                  comment: "Row label for GPU utilization"),
                    sublabel: String(localized: "status.row.gpu_utilization.sub",
                                     defaultValue: "Approximate · active request load",
                                     comment: "Sublabel for GPU utilization row")
                ) {
                    Text(vm.gpuUtilizationText)
                        .font(.omlxMono(12))
                }
                Row(label: String(localized: "status.row.thermal_state",
                                  defaultValue: "Thermal State",
                                  comment: "Row label for thermal state")) {
                    ThermalTrailing(state: vm.metrics.thermalState)
                }
                Row(label: String(localized: "status.row.server_uptime",
                                  defaultValue: "Server Uptime",
                                  comment: "Row label for server uptime")) {
                    Text(vm.uptimeText)
                        .font(.omlxMono(12))
                }
                Row(label: String(localized: "status.row.version",
                                  defaultValue: "oMLX Version",
                                  comment: "Row label for the running oMLX version"),
                    isLast: true) {
                    Text(vm.versionText)
                        .font(.omlxMono(12))
                        .foregroundStyle(.secondary)
                }
            }

            RuntimeCacheSection(
                cache: vm.stats?.runtimeCache,
                onClearSsdTap: { showingClearSsdCacheConfirm = true },
                onClearHotTap: { showingClearHotCacheConfirm = true }
            )

            SectionHeader(String(localized: "status.updates.title",
                                  defaultValue: "Updates",
                                  comment: "Section header for the updates section"))
            UpdatesSection(updates: services.updates)

            if let error = vm.lastError {
                Text(error)
                    .font(.omlxText(11))
                    .foregroundStyle(.red)
                    .padding(.horizontal, 18).padding(.top, 8)
            }
        }
        .task(id: vm.scope) {
            await vm.start(client: services.client)
        }
        .onDisappear { vm.stop() }
        .confirmationDialog(
            vm.scope == "alltime"
                ? String(localized: "status.confirm.clear_alltime",
                         defaultValue: "Clear all-time stats? This cannot be undone.",
                         comment: "Confirmation dialog title for clearing all-time stats")
                : String(localized: "status.confirm.clear_session",
                         defaultValue: "Clear session stats?",
                         comment: "Confirmation dialog title for clearing session stats"),
            isPresented: $showingClearStatsConfirm,
            titleVisibility: .visible
        ) {
            Button(String(localized: "status.confirm.clear_button",
                          defaultValue: "Clear",
                          comment: "Destructive clear button inside stats clear confirmation dialogs"),
                   role: .destructive) {
                Task { await vm.clearStats() }
            }
            Button(String(localized: "common.cancel",
                          defaultValue: "Cancel",
                          comment: "Common cancel button label"),
                   role: .cancel) {}
        }
        .confirmationDialog(
            String(localized: "status.confirm.clear_ssd",
                   defaultValue: "Clear all SSD cache files? Loaded models will rebuild their KV cache on next prefill.",
                   comment: "Confirmation dialog title for clearing the SSD KV cache"),
            isPresented: $showingClearSsdCacheConfirm,
            titleVisibility: .visible
        ) {
            Button(String(localized: "status.confirm.clear_button",
                          defaultValue: "Clear",
                          comment: "Destructive clear button inside stats clear confirmation dialogs"),
                   role: .destructive) {
                Task { await vm.clearSsdCache() }
            }
            Button(String(localized: "common.cancel",
                          defaultValue: "Cancel",
                          comment: "Common cancel button label"),
                   role: .cancel) {}
        }
        .confirmationDialog(
            String(localized: "status.confirm.clear_hot",
                   defaultValue: "Clear the in-memory KV cache for all loaded models? Subsequent requests will re-fault from SSD or recompute.",
                   comment: "Confirmation dialog title for clearing the hot (memory) KV cache"),
            isPresented: $showingClearHotCacheConfirm,
            titleVisibility: .visible
        ) {
            Button(String(localized: "status.confirm.clear_button",
                          defaultValue: "Clear",
                          comment: "Destructive clear button inside stats clear confirmation dialogs"),
                   role: .destructive) {
                Task { await vm.clearHotCache() }
            }
            Button(String(localized: "common.cancel",
                          defaultValue: "Cancel",
                          comment: "Common cancel button label"),
                   role: .cancel) {}
        }
    }
}

// MARK: - Runtime Cache section

/// Two-tier KV cache observability — memory (hot) tier + on-disk (SSD)
/// tier. Mirrors HTML _status.html's "Runtime Cache Observability"
/// panel. The memory tier rows hide when `hot_cache_max_bytes == 0`,
/// which signals the memory tier is disabled for the current run.
private struct RuntimeCacheSection: View {
    let cache: StatsDTO.RuntimeCacheDTO?
    let onClearSsdTap: () -> Void
    let onClearHotTap: () -> Void

    @Environment(\.omlxTheme) private var theme

    private var hotTierEnabled: Bool {
        (cache?.hotCacheMaxBytes ?? 0) > 0
    }

    var body: some View {
        SectionHeader(String(localized: "status.section.runtime_cache",
                              defaultValue: "Runtime Cache",
                              comment: "Section header for the runtime cache observability panel"),
                      subtitle: subtitleText)
        ListGroup {
            if hotTierEnabled {
                Row(label: String(localized: "status.runtime_cache.memory",
                                  defaultValue: "Memory Cache",
                                  comment: "Row label for the in-memory (hot) KV cache size gauge")) {
                    Text(memoryGaugeText)
                        .font(.omlxMono(12))
                }
                Row(label: String(localized: "status.runtime_cache.memory_entries",
                                  defaultValue: "Memory Entries",
                                  comment: "Row label for the number of in-memory KV cache entries")) {
                    Text(memoryEntriesText)
                        .font(.omlxMono(12))
                }
            }
            Row(label: String(localized: "status.runtime_cache.files",
                              defaultValue: "Cache Files",
                              comment: "Row label for cache file count")) {
                Text(fileCountText)
                    .font(.omlxMono(12))
            }
            Row(label: String(localized: "status.runtime_cache.size",
                              defaultValue: "Total Size",
                              comment: "Row label for total cache size on disk")) {
                Text(sizeText)
                    .font(.omlxMono(12))
            }
            if let dir = cache?.ssdCacheDir, !dir.isEmpty {
                Row(label: String(localized: "status.runtime_cache.location",
                                  defaultValue: "Location",
                                  comment: "Row label for the SSD cache directory path")) {
                    Text(dir)
                        .font(.omlxMono(11))
                        .foregroundStyle(theme.textSecondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .frame(maxWidth: 280, alignment: .trailing)
                        .help(dir)
                }
            }
            FreeRow(isLast: true) {
                HStack(spacing: 8) {
                    Spacer()
                    if hotTierEnabled {
                        Button(String(localized: "status.runtime_cache.clear_hot",
                                      defaultValue: "Clear Memory Cache",
                                      comment: "Destructive button to clear the in-memory KV cache"),
                               action: onClearHotTap)
                            .buttonStyle(.omlx(.destructive, size: .small))
                            .disabled((cache?.hotCacheEntries ?? 0) == 0)
                    }
                    Button(String(localized: "status.runtime_cache.clear_ssd",
                                  defaultValue: "Clear SSD Cache",
                                  comment: "Destructive button to clear all SSD cache files"),
                           action: onClearSsdTap)
                        .buttonStyle(.omlx(.destructive, size: .small))
                        .disabled((cache?.totalNumFiles ?? 0) == 0)
                }
            }
        }
    }

    private var memoryGaugeText: String {
        let used = cache?.hotCacheSizeBytes ?? 0
        let cap = cache?.hotCacheMaxBytes ?? 0
        return "\(formatByteCount(used)) / \(formatByteCount(cap))"
    }

    private var memoryEntriesText: String {
        let n = cache?.hotCacheEntries ?? 0
        return String(localized: "status.runtime_cache.memory_entries.count",
                      defaultValue: "Entries: \(n)",
                      comment: "Memory cache entry count; placeholder is the number of entries")
    }

    private var fileCountText: String {
        guard let n = cache?.totalNumFiles else { return "—" }
        return String(localized: "status.runtime_cache.file_count.count",
                      defaultValue: "Files: \(n)",
                      comment: "Cache file count; placeholder is the number of files")
    }

    private var sizeText: String {
        guard let bytes = cache?.totalSizeBytes else { return "—" }
        return formatByteCount(bytes)
    }

    private var subtitleText: String? {
        guard let cache else { return nil }
        let blocks = cache.effectiveBlockSizes ?? []
        guard !blocks.isEmpty else { return nil }
        let list = blocks.map(String.init).joined(separator: " · ")
        return String(localized: "status.runtime_cache.block_size",
                      defaultValue: "Block size · \(list)",
                      comment: "Subtitle showing the effective KV cache block sizes; placeholder is a separator-joined list of numbers")
    }
}

/// Bytes → human-readable string. Mirrors `formatByteCount` in
/// `omlx/admin/static/js/dashboard.js` — 1024-scaled, one decimal place.
private func formatByteCount(_ bytes: Int64) -> String {
    let b = Double(max(0, bytes))
    if b >= 1024 * 1024 * 1024 { return String(format: "%.1f GB", b / (1024 * 1024 * 1024)) }
    if b >= 1024 * 1024        { return String(format: "%.1f MB", b / (1024 * 1024)) }
    if b >= 1024               { return String(format: "%.1f KB", b / 1024) }
    return "\(Int(b.rounded())) B"
}

// MARK: - Stat tiles

private struct StatTilesRow: View {
    let stats: StatsDTO?

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            StatTile(
                label: String(localized: "status.tile.total",
                              defaultValue: "Total Prefill Tokens",
                              comment: "Stat tile label for total prefill tokens processed"),
                value: stats.map { fmtNum($0.totalPromptTokens) } ?? "—"
            )
            StatTile(
                label: String(localized: "status.tile.cached",
                              defaultValue: "Cached Tokens",
                              comment: "Stat tile label for cached tokens"),
                value: stats.map { fmtNum($0.totalCachedTokens) } ?? "—"
            )
            StatTile(
                label: String(localized: "status.tile.cache_efficiency",
                              defaultValue: "Cache Efficiency",
                              comment: "Stat tile label for cache efficiency percentage"),
                value: stats.map { String(format: "%.1f%%", $0.cacheEfficiency) } ?? "—"
            )
        }
        .padding(.horizontal, 14)
        .padding(.bottom, 4)
    }
}

private struct StatTile: View {
    enum AccentRole { case neutral, success, warning, danger }

    let label: String
    let value: String
    var sub: String?
    var accentRole: AccentRole = .neutral

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(label)
                .font(.omlxText(11, weight: .medium))
                .foregroundStyle(theme.textSecondary)
                .textCase(.uppercase)
                .kerning(0.6)
                .multilineTextAlignment(.center)
                .frame(maxWidth: .infinity)
            Text(value)
                .font(.omlxText(22, weight: .semibold))
                .kerning(-0.5)
                .foregroundStyle(accentColor)
                .frame(maxWidth: .infinity)
            if let sub {
                Text(sub)
                    .font(.omlxText(11))
                    .foregroundStyle(theme.textTertiary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: .infinity)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, minHeight: 96, alignment: .center)
        .background(theme.groupBg)
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
    }

    private var accentColor: Color {
        switch accentRole {
        case .neutral: return theme.text
        case .success: return theme.greenDot
        case .warning: return theme.amberDot
        case .danger:  return theme.redDot
        }
    }
}

private struct AverageSpeedTilesRow: View {
    let stats: StatsDTO?

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            StatTile(
                label: String(localized: "status.speed.prompt_processing.tile",
                              defaultValue: "Prompt Processing (excl. cached)",
                              comment: "Tile label for average prompt-processing speed excluding cached tokens"),
                value: stats.map { String(format: "%.1f tok/s", $0.avgPrefillTps) } ?? "—"
            )
            StatTile(
                label: String(localized: "status.speed.token_generation",
                              defaultValue: "Token Generation",
                              comment: "Label for average token-generation speed"),
                value: stats.map { String(format: "%.1f tok/s", $0.avgGenerationTps) } ?? "—"
            )
        }
        .padding(.horizontal, 14)
        .padding(.bottom, 4)
    }
}

// MARK: - System rows

private struct GpuMemoryTrailing: View {
    let stats: StatsDTO?
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        let used = stats?.activeModels.modelMemoryUsed
        let max = stats?.activeModels.modelMemoryMax
        HStack(spacing: 10) {
            ProgressBar(progress: progress, tint: theme.blueDot)
                .frame(width: 140)
            Text(labelText(used: used, max: max))
                .font(.omlxMono(11))
                .foregroundStyle(theme.textSecondary)
                .frame(minWidth: 110, alignment: .trailing)
        }
    }

    private var progress: Double {
        guard
            let used = stats?.activeModels.modelMemoryUsed,
            let max = stats?.activeModels.modelMemoryMax,
            max > 0
        else { return 0 }
        return Double(used) / Double(max)
    }

    private func labelText(used: Int64?, max: Int64?) -> String {
        guard let used, let max else { return "—" }
        let u = SystemMetricsPoller.formatBytesAsGB(UInt64(Swift.max(0, used)))
        let m = SystemMetricsPoller.formatBytesAsGB(UInt64(Swift.max(0, max)))
        return "\(u) / \(m) GB"
    }
}

private struct SystemRamTrailing: View {
    let metrics: SystemMetricsPoller
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        HStack(spacing: 10) {
            ProgressBar(progress: progress, tint: theme.text.opacity(0.7))
                .frame(width: 140)
            Text(labelText)
                .font(.omlxMono(11))
                .foregroundStyle(theme.textSecondary)
                .frame(minWidth: 110, alignment: .trailing)
        }
    }

    private var progress: Double {
        guard let used = metrics.ramUsedBytes,
              let total = metrics.ramTotalBytes,
              total > 0
        else { return 0 }
        return Double(used) / Double(total)
    }

    private var labelText: String {
        guard let used = metrics.ramUsedBytes,
              let total = metrics.ramTotalBytes
        else { return "—" }
        let u = SystemMetricsPoller.formatBytesAsGB(used)
        let t = SystemMetricsPoller.formatBytesAsGB(total)
        return "\(u) / \(t) GB"
    }
}

private struct ThermalTrailing: View {
    let state: ProcessInfo.ThermalState
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        let severity = SystemMetricsPoller.severity(for: state)
        HStack(spacing: 8) {
            Circle()
                .fill(color(for: severity))
                .frame(width: 8, height: 8)
            Text(SystemMetricsPoller.label(for: severity))
                .font(.omlxText(12))
                .foregroundStyle(theme.text)
        }
    }

    private func color(for severity: ThermalSeverity) -> Color {
        switch severity {
        case .nominal:               return theme.greenDot
        case .fair, .serious:        return theme.amberDot
        case .critical:              return theme.redDot
        }
    }
}

// MARK: - Active Now

private struct ActiveNowList: View {
    let models: [StatsDTO.ActiveModelDTO]
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        ListGroup {
            if models.isEmpty {
                FreeRow(isLast: true) {
                    Text(String(localized: "status.active_now.empty",
                                defaultValue: "Server idle — no models loaded",
                                comment: "Empty-state text shown when the Active Now list has no entries"))
                        .font(.omlxText(12))
                        .foregroundStyle(theme.textTertiary)
                        .frame(maxWidth: .infinity, alignment: .center)
                        .padding(.vertical, 12)
                }
            } else {
                ForEach(Array(models.enumerated()), id: \.element.id) { index, model in
                    Row(label: model.id, isLast: index == models.count - 1) {
                        HStack(spacing: 12) {
                            modelStateBadge(for: model)
                            Text(model.estimatedSizeFormatted ?? "—")
                                .font(.omlxMono(11))
                                .foregroundStyle(theme.textSecondary)
                                .frame(minWidth: 60, alignment: .trailing)
                        }
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func modelStateBadge(for model: StatsDTO.ActiveModelDTO) -> some View {
        if model.isLoading == true {
            StatusPill(status: .starting)
        } else if (model.activeRequests ?? 0) > 0 {
            StatusPill(status: .custom(color: theme.greenDot,
                                        label: String(localized: "status.active_now.badge.generating",
                                                      defaultValue: "Generating",
                                                      comment: "Active Now badge: model currently generating tokens"),
                                        fillBg: true))
        } else if (model.waitingRequests ?? 0) > 0 {
            StatusPill(status: .custom(color: theme.amberDot,
                                        label: String(localized: "status.active_now.badge.waiting",
                                                      defaultValue: "Waiting",
                                                      comment: "Active Now badge: model has queued requests"),
                                        fillBg: true))
        } else {
            StatusPill(status: .custom(color: theme.textTertiary,
                                        label: String(localized: "status.active_now.badge.loaded",
                                                      defaultValue: "Loaded",
                                                      comment: "Active Now badge: model is loaded but idle"),
                                        fillBg: true))
        }
    }
}

// MARK: - Updates section

private struct UpdatesSection: View {
    // Reads UpdateController directly so SwiftUI redraws only for the
    // observable update properties used by this section.
    let updates: UpdateController
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        ListGroup {
            FreeRow {
                HStack(spacing: 12) {
                    Squircle(systemSymbol: "arrow.down.circle.fill",
                             size: 32,
                             gradient: SquircleGradient.update)
                    VStack(alignment: .leading, spacing: 2) {
                        primaryLine
                        secondaryLine
                    }
                    Spacer(minLength: 8)
                    actionButton
                }
            }
            Row(
                label: String(localized: "status.updates.channel",
                              defaultValue: "Update Channel",
                              comment: "Row label for selecting the app update channel"),
                sublabel: String(localized: "status.updates.channel.sub",
                                 defaultValue: "Stable, release candidate, or dev builds",
                                 comment: "Sublabel for the update channel picker")
            ) {
                Popup(
                    selection: Binding(
                        get: { updates.channel },
                        set: { updates.channel = $0 }
                    ),
                    width: 190,
                    options: UpdateChannel.allCases.map { ($0, $0.displayName) }
                )
            }
            Row(
                label: String(localized: "status.updates.auto_check",
                              defaultValue: "Automatically Check",
                              comment: "Row label for the auto-check updates toggle"),
                sublabel: String(localized: "status.updates.auto_check.sub",
                                 defaultValue: "Look for updates daily in the background",
                                 comment: "Sublabel for the auto-check toggle")
            ) {
                Toggle("", isOn: Binding(
                    get: { updates.autoCheck },
                    set: { updates.autoCheck = $0 }
                ))
                .labelsHidden()
                .toggleStyle(.switch)
            }
            Row(
                label: String(localized: "status.updates.auto_download",
                              defaultValue: "Notify About Updates",
                              comment: "Row label for the automatic update notification toggle"),
                sublabel: String(localized: "status.updates.auto_download.sub",
                                 defaultValue: "Show release notes before downloading in the background",
                                 comment: "Sublabel for the automatic update notification toggle"),
                isLast: true
            ) {
                Toggle("", isOn: Binding(
                    get: { updates.autoNotify },
                    set: { updates.autoNotify = $0 }
                ))
                .labelsHidden()
                .toggleStyle(.switch)
            }
        }
    }

    @ViewBuilder
    private var primaryLine: some View {
        switch updates.state {
        case .checking:
            Text(String(localized: "status.updates.checking_primary",
                        defaultValue: "Checking for updates…",
                        comment: "Primary update status line while a check is in progress"))
                .font(.omlxText(13, weight: .medium))
                .foregroundStyle(theme.text)
        case .available(let upd), .ready(let upd):
            Text(String(localized: "status.updates.available_primary",
                        defaultValue: "oMLX \(upd.version) is available",
                        comment: "Primary update status line when a new version is available; placeholder is the version string"))
                .font(.omlxText(13, weight: .medium))
                .foregroundStyle(theme.text)
        case .downloading:
            Text(String(localized: "status.updates.downloading_primary",
                        defaultValue: "Downloading update…",
                        comment: "Primary update status line while the new build is downloading"))
                .font(.omlxText(13, weight: .medium))
                .foregroundStyle(theme.text)
        case .idle:
            Text(updates.lastError == nil
                 ? String(localized: "status.updates.up_to_date_primary",
                          defaultValue: "oMLX is up to date",
                          comment: "Primary update status line when no update is available")
                 : String(localized: "status.updates.error_primary",
                          defaultValue: "Update check failed",
                          comment: "Primary update status line when checking for updates failed"))
                .font(.omlxText(13, weight: .medium))
                .foregroundStyle(theme.text)
        }
    }

    @ViewBuilder
    private var secondaryLine: some View {
        switch updates.state {
        case .checking:
            Text(String(localized: "status.updates.checking_secondary",
                        defaultValue: "Checking GitHub releases…",
                        comment: "Secondary update status line while a check is in progress"))
                .font(.omlxText(11))
                .foregroundStyle(theme.textSecondary)
        case .available(let upd):
            Text(String(localized: "status.updates.available_secondary",
                        defaultValue: "Ready to download · \(upd.sizeText ?? "—")",
                        comment: "Secondary update status line when a new version is available; placeholder is the download size or em dash"))
                .font(.omlxText(11))
                .foregroundStyle(theme.textSecondary)
        case .downloading(let pct):
            Text(String(localized: "status.updates.downloading_secondary",
                        defaultValue: "Downloading · \(pct)%",
                        comment: "Secondary update status line during download; placeholder is the percent complete"))
                .font(.omlxText(11))
                .foregroundStyle(theme.textSecondary)
        case .ready(let upd):
            Text(String(localized: "status.updates.ready_secondary",
                        defaultValue: "\(upd.version) is ready to install",
                        comment: "Secondary update status line once the staged bundle is ready; placeholder is the version"))
                .font(.omlxText(11))
                .foregroundStyle(theme.textSecondary)
        case .idle(let lastChecked):
            Text(updates.lastError ?? lastCheckedText(lastChecked))
                .font(.omlxText(11))
                .foregroundStyle(theme.textSecondary)
        }
    }

    @ViewBuilder
    private var actionButton: some View {
        switch updates.state {
        case .available, .ready:
            Button(String(localized: "status.updates.install",
                          defaultValue: "Review Update",
                          comment: "Updates action button to review release notes before installing")) {
                updates.requestUpdateConfirmation()
            }
                .buttonStyle(.omlx(.primary, size: .small))
        case .downloading(let pct):
            Button(String(localized: "status.updates.downloading_button",
                          defaultValue: "Downloading… \(pct)%",
                          comment: "Updates action button label while the download is in progress; placeholder is the percent complete")) { }
                .buttonStyle(.omlx(.normal, size: .small))
                .disabled(true)
        case .checking:
            Button(String(localized: "status.updates.checking_button",
                          defaultValue: "Checking…",
                          comment: "Updates action button label while a check is in progress")) { }
                .buttonStyle(.omlx(.normal, size: .small))
                .disabled(true)
        case .idle:
            Button(String(localized: "status.updates.check",
                          defaultValue: "Check Now",
                          comment: "Updates action button label to trigger a manual update check")) {
                updates.checkForUpdates()
            }
                .buttonStyle(.omlx(.normal, size: .small))
        }
    }

    private func lastCheckedText(_ date: Date?) -> String {
        guard let date else {
            return String(localized: "status.updates.never_checked",
                          defaultValue: "Never checked",
                          comment: "Secondary update line when there is no prior check timestamp")
        }
        let formatter = DateFormatter()
        formatter.dateStyle = .none
        formatter.timeStyle = .short
        let calendar = Calendar.current
        if calendar.isDateInToday(date) {
            return String(localized: "status.updates.last_checked_today",
                          defaultValue: "Last checked today at \(formatter.string(from: date))",
                          comment: "Last-checked text when the last check was today; placeholder is the local time")
        }
        formatter.dateStyle = .medium
        return String(localized: "status.updates.last_checked_date",
                      defaultValue: "Last checked \(formatter.string(from: date))",
                      comment: "Last-checked text when the last check was before today; placeholder is the formatted date and time")
    }
}

// MARK: - Helpers

private func fmtNum(_ n: Int) -> String {
    let v = Double(n)
    if v >= 1e9 { return String(format: "%.2fB", v / 1e9) }
    if v >= 1e6 { return String(format: "%.2fM", v / 1e6) }
    if v >= 1e3 { return String(format: "%.1fK", v / 1e3) }
    return String(n)
}

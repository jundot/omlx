// PR 7 — Status screen. Lays out:
//   • ServerHero (shared with Server screen)
//   • Session Stats — 4 StatTiles from /admin/api/stats (scope segmented)
//   • System — slice of /admin/api/global-settings + uptime from /api/stats
//   • Updates — three-state row + Channel/AutoCheck/AutoDownload (UpdateController stub)
//   • Active Now — active_models slice from /api/stats
//
// Polling is on-screen-only: a 5s timer ticks while the view is visible.

import SwiftUI

struct StatusScreen: View {
    @EnvironmentObject private var services: AppServices
    @StateObject private var vm = StatusScreenVM()

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ServerHeroSlim()

            SectionHeader("Session Stats") {
                Segmented(selection: $vm.scope, options: [
                    ("session", "Session"),
                    ("alltime", "All Time"),
                ])
                .frame(width: 160)
            }
            StatTilesRow(stats: vm.stats)

            SectionHeader("System", subtitle: vm.systemSubtitle)
            ListGroup {
                Row(label: "oMLX Version") {
                    Text(vm.versionText)
                        .font(.omlxMono(12))
                        .foregroundStyle(.secondary)
                }
                Row(label: "Server Uptime") {
                    Text(vm.uptimeText)
                        .font(.omlxMono(12))
                }
                Row(label: "Server", isLast: true) {
                    Text(vm.endpointText)
                        .font(.omlxMono(12))
                        .foregroundStyle(.secondary)
                }
            }

            SectionHeader("Updates")
            UpdatesSection(updates: services.updates)

            SectionHeader("Active Now")
            ActiveNowList(models: vm.stats?.activeModels.models ?? [])

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
    }
}

// MARK: - Slim hero (Status screen variant — no buttons, just status info)

private struct ServerHeroSlim: View {
    @EnvironmentObject private var services: AppServices
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        HStack(spacing: 16) {
            Squircle(gradient: SquircleGradient.server, size: 44) {
                Text("oM")
                    .font(.omlxText(18, weight: .heavy))
                    .kerning(-0.4)
                    .foregroundStyle(.white)
            }
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 10) {
                    Text("oMLX Server")
                        .font(.omlxText(16, weight: .semibold))
                        .foregroundStyle(theme.text)
                    StatusPill(status: pillStatus)
                }
                Text(subtitle)
                    .font(.omlxText(11.5))
                    .foregroundStyle(theme.textSecondary)
            }
            Spacer()
        }
        .padding(14)
        .background(theme.groupBg)
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(theme.groupBorder, lineWidth: 0.5)
        )
        .padding(.horizontal, 14)
        .padding(.bottom, 14)
    }

    private var pillStatus: StatusPill.Status {
        switch services.serverState {
        case .running:      return .running
        case .starting:     return .starting
        case .stopping:     return .stopping
        case .stopped:      return .stopped
        case .unresponsive: return .custom(color: theme.amberDot, label: "Unresponsive", fillBg: true)
        case .failed:       return .error
        }
    }

    private var subtitle: String {
        let host = services.config.host, port = services.config.port
        switch services.serverState {
        case .running, .unresponsive: return "Listening on \(host):\(port)"
        case .starting:               return "Starting…"
        case .stopping:               return "Stopping…"
        case .stopped:                return "Not running"
        case .failed(let m):          return m
        }
    }
}

// MARK: - Stat tiles

private struct StatTilesRow: View {
    let stats: StatsDTO?

    var body: some View {
        HStack(spacing: 10) {
            StatTile(
                label: "Total Tokens",
                value: stats.map { fmtNum($0.totalTokensServed) } ?? "—",
                sub: "prompt + completion"
            )
            StatTile(
                label: "Cached",
                value: stats.map { fmtNum($0.totalCachedTokens) } ?? "—",
                sub: stats.map { String(format: "%.1f%% efficiency", $0.cacheEfficiency) },
                accentRole: .success
            )
            StatTile(
                label: "Generation",
                value: stats.map { String(format: "%.1f tok/s", $0.avgGenerationTps) } ?? "—",
                sub: "across active models"
            )
            StatTile(
                label: "Requests",
                value: stats.map { fmtNum($0.totalRequests) } ?? "—",
                sub: stats.map { "\(($0.activeModels.totalActiveRequests ?? 0)) in flight" }
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
            Text(value)
                .font(.omlxText(22, weight: .semibold))
                .kerning(-0.5)
                .foregroundStyle(accentColor)
            if let sub {
                Text(sub)
                    .font(.omlxText(11))
                    .foregroundStyle(theme.textTertiary)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(theme.groupBg)
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(theme.groupBorder, lineWidth: 0.5)
        )
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

// MARK: - Active Now

private struct ActiveNowList: View {
    let models: [StatsDTO.ActiveModelDTO]
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        ListGroup {
            if models.isEmpty {
                FreeRow(isLast: true) {
                    Text("Server idle — no models loaded")
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
            StatusPill(status: .custom(color: theme.greenDot, label: "Generating", fillBg: true))
        } else if (model.waitingRequests ?? 0) > 0 {
            StatusPill(status: .custom(color: theme.amberDot, label: "Waiting", fillBg: true))
        } else {
            StatusPill(status: .custom(color: theme.textTertiary, label: "Loaded", fillBg: true))
        }
    }
}

// MARK: - Updates section

private struct UpdatesSection: View {
    // Observed directly so SwiftUI redraws on UpdateController's own
    // @Published changes — nested ObservableObjects don't republish.
    @ObservedObject var updates: UpdateController
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
                label: "Update Channel",
                sublabel: "Stable receives tested releases. Beta gets new features sooner."
            ) {
                Popup(
                    selection: Binding(
                        get: { updates.channel },
                        set: { updates.channel = $0 }
                    ),
                    width: 130,
                    options: UpdateChannel.allCases.map { ($0, $0.displayName) }
                )
            }
            Row(
                label: "Automatically Check",
                sublabel: "Look for updates daily in the background"
            ) {
                Toggle("", isOn: Binding(
                    get: { updates.autoCheck },
                    set: { updates.autoCheck = $0 }
                ))
                .labelsHidden()
                .toggleStyle(.switch)
            }
            Row(
                label: "Auto-download Updates",
                sublabel: "Download in the background, install on next launch",
                isLast: true
            ) {
                Toggle("", isOn: Binding(
                    get: { updates.autoDownload },
                    set: { updates.autoDownload = $0 }
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
            Text("Checking for updates…")
                .font(.omlxText(13, weight: .medium))
                .foregroundStyle(theme.text)
        case .available(let upd):
            Text("oMLX \(upd.version) is available")
                .font(.omlxText(13, weight: .medium))
                .foregroundStyle(theme.text)
        case .idle:
            Text("oMLX is up to date")
                .font(.omlxText(13, weight: .medium))
                .foregroundStyle(theme.text)
        }
    }

    @ViewBuilder
    private var secondaryLine: some View {
        switch updates.state {
        case .checking:
            Text("Checking GitHub releases…")
                .font(.omlxText(11))
                .foregroundStyle(theme.textSecondary)
        case .available(let upd):
            Text("You have the current version · \(upd.sizeText ?? "—")")
                .font(.omlxText(11))
                .foregroundStyle(theme.textSecondary)
        case .idle(let lastChecked):
            Text(lastCheckedText(lastChecked))
                .font(.omlxText(11))
                .foregroundStyle(theme.textSecondary)
        }
    }

    @ViewBuilder
    private var actionButton: some View {
        switch updates.state {
        case .available:
            Button("Install & Restart") { updates.installAndRestart() }
                .buttonStyle(.omlx(.primary, size: .small))
        case .checking:
            Button("Checking…") { }
                .buttonStyle(.omlx(.normal, size: .small))
                .disabled(true)
        case .idle:
            Button("Check Now") { updates.checkForUpdates() }
                .buttonStyle(.omlx(.normal, size: .small))
        }
    }

    private func lastCheckedText(_ date: Date?) -> String {
        guard let date else { return "Never checked" }
        let formatter = DateFormatter()
        formatter.dateStyle = .none
        formatter.timeStyle = .short
        let calendar = Calendar.current
        if calendar.isDateInToday(date) {
            return "Last checked today at \(formatter.string(from: date))"
        }
        formatter.dateStyle = .medium
        return "Last checked \(formatter.string(from: date))"
    }
}

// MARK: - View model

@MainActor
final class StatusScreenVM: ObservableObject {
    @Published var scope: String = "session"
    @Published var stats: StatsDTO?
    @Published var lastError: String?

    private weak var client: OMLXClient?
    private var pollTask: Task<Void, Never>?

    var systemSubtitle: String {
        var arch = "Apple Silicon"
        #if arch(x86_64)
        arch = "Intel"
        #endif
        let os = ProcessInfo.processInfo.operatingSystemVersion
        return "\(arch) · macOS \(os.majorVersion).\(os.minorVersion)"
    }

    var versionText: String {
        let v = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "0.0.0"
        let b = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "—"
        return "\(v) · build \(b)"
    }

    var uptimeText: String {
        guard let s = stats?.uptimeSeconds else { return "—" }
        return formatUptime(s)
    }

    var endpointText: String {
        guard let s = stats else { return "—" }
        let host = s.host ?? "127.0.0.1"
        let port = s.port ?? 8080
        return "\(host):\(port)"
    }

    func start(client: OMLXClient) async {
        self.client = client
        pollTask?.cancel()
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                await self.tick()
                try? await Task.sleep(for: .seconds(5))
            }
        }
    }

    func stop() {
        pollTask?.cancel()
        pollTask = nil
    }

    private func tick() async {
        guard let client else { return }
        do {
            self.stats = try await client.getStats(scope: scope)
            self.lastError = nil
        } catch {
            self.lastError = describe(error)
        }
    }

    private func describe(_ error: Error) -> String {
        if let omlx = error as? OMLXClientError { return String(describing: omlx) }
        return error.localizedDescription
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

private func formatUptime(_ seconds: Double) -> String {
    let total = Int(seconds)
    let d = total / 86400
    let h = (total % 86400) / 3600
    let m = (total % 3600) / 60
    let s = total % 60
    if d > 0 { return "\(d)d \(h)h \(m)m" }
    if h > 0 { return "\(h)h \(m)m" }
    if m > 0 { return "\(m)m \(s)s" }
    return "\(s)s"
}

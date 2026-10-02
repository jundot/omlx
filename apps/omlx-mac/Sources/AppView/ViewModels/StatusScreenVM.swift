import SwiftUI

@MainActor
@Observable
final class StatusScreenVM {
    var scope: String = "session"
    var stats: StatsDTO?
    var lastError: String?
    /// Loaded once on appear from `scheduler.max_concurrent_requests`.
    /// 8 is the server's default — used as the divisor before settings load
    /// so the % column doesn't read 0/0 on first paint.
    var maxConcurrent: Int = 8
    var metrics = SystemMetricsPoller()

    @ObservationIgnored
    private weak var client: OMLXClient?
    @ObservationIgnored
    private var pollTask: Task<Void, Never>?

    var systemSubtitle: String {
        var arch = String(localized: "status.arch.apple_silicon",
                          defaultValue: "Apple Silicon",
                          comment: "Architecture label shown in the System section subtitle for Apple Silicon Macs")
        #if arch(x86_64)
        arch = String(localized: "status.arch.intel",
                      defaultValue: "Intel",
                      comment: "Architecture label shown in the System section subtitle for Intel Macs")
        #endif
        let os = ProcessInfo.processInfo.operatingSystemVersion
        return String(localized: "status.system.subtitle",
                      defaultValue: "\(arch) · macOS \(os.majorVersion).\(os.minorVersion)",
                      comment: "Subtitle of the System section showing CPU architecture and macOS version; placeholders are the architecture string and macOS major.minor numbers")
    }

    var versionText: String {
        let v = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "0.0.0"
        let b = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "—"
        return String(localized: "status.version.text",
                      defaultValue: "\(v) · build \(b)",
                      comment: "Version row value combining marketing version and build number; placeholders are the version string and build string")
    }

    var uptimeText: String {
        guard let s = stats?.uptimeSeconds else { return "—" }
        return formatUptime(s)
    }

    var endpointText: String {
        guard let s = stats else { return "—" }
        let host = s.host ?? "127.0.0.1"
        let port = s.port ?? 8000
        return "\(host):\(port)"
    }

    var gpuUtilizationPercent: Double {
        let active = Double(stats?.activeModels.totalActiveRequests ?? 0)
        let cap = Double(max(1, maxConcurrent))
        return min(100.0, active / cap * 100.0)
    }

    var gpuUtilizationText: String {
        // Trailing "%" looks better tight against the number; the value
        // is rounded to int so the row doesn't jitter at 12.499 → 12.500.
        String(format: "%d%%", Int(gpuUtilizationPercent.rounded()))
    }

    func start(client: OMLXClient) async {
        self.client = client
        metrics.start()
        // Load the scheduler cap once; failing silently is fine — the
        // default keeps the % column readable until the server responds.
        if let settings = try? await client.getGlobalSettings(),
           let max = settings.scheduler?.maxConcurrentRequests {
            self.maxConcurrent = max
        }
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
        metrics.stop()
    }

    private func tick() async {
        guard let client else { return }
        do {
            self.stats = try await client.getStats(scope: scope)
            self.lastError = nil
        } catch {
            self.lastError = error.omlxDescription
        }
    }

    /// Clear stats for the currently-displayed scope. Refreshes the tile
    /// values immediately afterwards so the user sees the zeroed state
    /// without waiting for the next poll.
    func clearStats() async {
        guard let client else { return }
        do {
            if scope == "alltime" {
                try await client.clearAlltimeStats()
            } else {
                try await client.clearStats()
            }
            await tick()
        } catch {
            self.lastError = error.omlxDescription
        }
    }

    /// Drop all SSD KV cache files (loaded models + direct disk sweep).
    /// Refreshes stats so the Runtime Cache counters reset to zero.
    func clearSsdCache() async {
        guard let client else { return }
        do {
            _ = try await client.clearSsdCache()
            await tick()
        } catch {
            self.lastError = error.omlxDescription
        }
    }

    /// Drop the in-memory (hot) KV cache for all loaded models. Subsequent
    /// requests re-fault from SSD or recompute.
    func clearHotCache() async {
        guard let client else { return }
        do {
            _ = try await client.clearHotCache()
            await tick()
        } catch {
            self.lastError = error.omlxDescription
        }
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

}

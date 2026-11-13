// RemoteSystemStats — the SERVER Mac's utilization, for installs that use
// another Mac's oMLX server. The server publishes `GET /api/system-metrics`
// (same auth as `/api/status`: `Authorization: Bearer <key>`, sub keys fine);
// this file decodes it and maps it into the same `SystemStatsSnapshot` the
// local `SystemStatsSampler` produces, so the menubar panels, the CPU/GPU/MEM
// status items and the Status screen's RAM block need no second code path.
// Every leaf of the payload may be null; a null maps to nil / 0 exactly as a
// failed local probe would.

import Foundation

struct RemoteSystemMetrics: Decodable, Equatable, Sendable {
    struct CPU: Decodable, Equatable, Sendable {
        var totalPercent: Double?
        var efficiencyPercent: Double?
        var performancePercent: Double?
        var efficiencyCores: Int?
        var performanceCores: Int?
        var loadAverages: [Double]?
    }

    struct GPU: Decodable, Equatable, Sendable {
        var utilizationPercent: Double?
        var memoryInUseBytes: UInt64?
    }

    struct Memory: Decodable, Equatable, Sendable {
        var totalBytes: UInt64?
        var wiredBytes: UInt64?
        var activeBytes: UInt64?
        var inactiveBytes: UInt64?
        var compressedBytes: UInt64?
        var freeBytes: UInt64?
    }

    var timestamp: Double?
    var uptimeSeconds: Double?
    var cpu: CPU?
    var gpu: GPU?
    var memory: Memory?
    /// nominal | fair | serious | critical | null
    var thermalState: String?

    static func decode(_ data: Data) throws -> RemoteSystemMetrics {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(RemoteSystemMetrics.self, from: data)
    }

    /// `GET /api/system-metrics`. Nil on any transport, status or decode
    /// failure (an older server has no such route and answers 404).
    nonisolated static func fetch(from endpoint: OMLXEndpoint, timeout: TimeInterval = 15) async -> RemoteSystemMetrics? {
        guard let url = endpoint.httpURL(path: "/api/system-metrics") else { return nil }
        var request = URLRequest(url: url, timeoutInterval: timeout)
        if let key = endpoint.apiKey, !key.isEmpty {
            request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        }
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse, 200..<300 ~= http.statusCode else { return nil }
            return try decode(data)
        } catch {
            return nil
        }
    }

    // MARK: - Mapping

    /// Memory block in the sampler's shape. `freeBytes` is derived as
    /// total - used (as the local sampler does) so the legend rows sum to the
    /// total; the server's own free/inactive figures are not shown.
    var snapshotMemory: SystemStatsSnapshot.Memory {
        var mem = SystemStatsSnapshot.Memory()
        mem.totalBytes = memory?.totalBytes ?? 0
        mem.wiredBytes = memory?.wiredBytes ?? 0
        mem.activeBytes = memory?.activeBytes ?? 0
        mem.compressedBytes = memory?.compressedBytes ?? 0
        mem.freeBytes = mem.totalBytes > mem.usedBytes ? mem.totalBytes - mem.usedBytes : 0
        return mem
    }

    var processThermalState: ProcessInfo.ThermalState {
        switch thermalState {
        case "fair": return .fair
        case "serious": return .serious
        case "critical": return .critical
        default: return .nominal
        }
    }

    /// Server percents (0...100) to the snapshot's 0...1 fractions.
    nonisolated static func fraction(_ percent: Double?) -> Double? {
        percent.map { min(1, max(0, $0 / 100)) }
    }
}

/// Fetches the server's metrics and keeps the rolling E/P/GPU series
/// client-side, the way `SystemStatsSampler` does.
@MainActor
final class RemoteSystemStatsSource {
    let endpoint: OMLXEndpoint
    private var eHistory: [Double] = []
    private var pHistory: [Double] = []
    private var gpuHistory: [Double] = []

    init(endpoint: OMLXEndpoint) {
        self.endpoint = endpoint
    }

    /// Nil when the server did not answer with a usable payload.
    func fetch() async -> SystemStatsSnapshot? {
        guard let metrics = await RemoteSystemMetrics.fetch(from: endpoint) else { return nil }
        return snapshot(from: metrics)
    }

    /// Maps one payload, appending to the series. Exposed for unit tests.
    func snapshot(from metrics: RemoteSystemMetrics) -> SystemStatsSnapshot {
        var snapshot = SystemStatsSnapshot()

        if let e = RemoteSystemMetrics.fraction(metrics.cpu?.efficiencyPercent) {
            snapshot.eCoreUsage = e
            MenubarMetricsStore.append(&eHistory, e)
        }
        if let p = RemoteSystemMetrics.fraction(metrics.cpu?.performancePercent) {
            snapshot.pCoreUsage = p
            MenubarMetricsStore.append(&pHistory, p)
        }
        snapshot.cpuTotalUsage = RemoteSystemMetrics.fraction(metrics.cpu?.totalPercent)

        if let gpu = RemoteSystemMetrics.fraction(metrics.gpu?.utilizationPercent) {
            snapshot.gpuUsage = gpu
            MenubarMetricsStore.append(&gpuHistory, gpu)
        }
        snapshot.gpuMemoryInUseBytes = metrics.gpu?.memoryInUseBytes

        snapshot.memory = metrics.snapshotMemory
        snapshot.loadAverages = metrics.cpu?.loadAverages ?? []
        snapshot.uptimeSeconds = metrics.uptimeSeconds ?? 0
        snapshot.thermalState = metrics.processThermalState

        snapshot.eHistory = eHistory
        snapshot.pHistory = pHistory
        snapshot.gpuHistory = gpuHistory
        return snapshot
    }
}

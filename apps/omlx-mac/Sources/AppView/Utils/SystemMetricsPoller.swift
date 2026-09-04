// PR 15 — System metrics for the Status screen.
//
// The redesign's System block exposes live RAM usage, thermal state, and
// the GPU/utilization proxy alongside the existing uptime and version rows.
// We poll Mach (`host_statistics64`) every 5 seconds on-screen — cheap,
// kernel-resident, no entitlements required. `ProcessInfo.thermalState`
// changes infrequently but we re-read it on the same tick to avoid a
// second timer.

import Foundation
import Darwin.Mach

/// Bucketed thermal levels. Decoupled from the SwiftUI `Color` so the
/// mapping can be unit-tested without dragging in the theme.
enum ThermalSeverity: String, Equatable, Sendable {
    case nominal, fair, serious, critical
}

@MainActor
@Observable
final class SystemMetricsPoller {
    var ramUsedBytes: UInt64?
    var ramTotalBytes: UInt64?
    var thermalState: ProcessInfo.ThermalState = .nominal

    @ObservationIgnored
    private var pollTask: Task<Void, Never>?

    func start() {
        pollTask?.cancel()
        // Read once immediately so the bar doesn't flash empty on appear.
        tick()
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(5))
                if Task.isCancelled { return }
                await MainActor.run { self?.tick() }
            }
        }
    }

    func stop() {
        pollTask?.cancel()
        pollTask = nil
    }

    private func tick() {
        thermalState = ProcessInfo.processInfo.thermalState
        ramTotalBytes = ProcessInfo.processInfo.physicalMemory
        if let used = Self.readRamUsedBytes() {
            ramUsedBytes = used
        }
    }

    // MARK: - Mach

    /// Reads (active + wired + compressor) page counts and multiplies by
    /// the kernel page size. This is the closest analogue to Activity
    /// Monitor's "Memory Used" without parsing memory_pressure or shelling
    /// to `vm_stat`.
    nonisolated private static func readRamUsedBytes() -> UInt64? {
        var size = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.stride / MemoryLayout<integer_t>.stride
        )
        var stats = vm_statistics64_data_t()
        let result = withUnsafeMutablePointer(to: &stats) { ptr -> kern_return_t in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(size)) { rebound in
                host_statistics64(mach_host_self(), HOST_VM_INFO64, rebound, &size)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        // Ask the kernel for the page size at runtime — the global
        // `vm_kernel_page_size` is a mutable C var Swift can't import
        // under strict concurrency, and `host_page_size` is a stable
        // alternative shipped on every macOS we support.
        var pageSize: vm_size_t = 0
        guard host_page_size(mach_host_self(), &pageSize) == KERN_SUCCESS else {
            return nil
        }
        let pages = UInt64(stats.active_count)
            + UInt64(stats.wire_count)
            + UInt64(stats.compressor_page_count)
        return pages * UInt64(pageSize)
    }

    // MARK: - Helpers (exposed for tests + view layer)

    /// Map the system's `ThermalState` enum (which has no stable raw
    /// representation across SDKs) to our own severity. Anything Apple
    /// adds beyond `.critical` later falls through to `.critical` — better
    /// to flag too hot than to silently render as nominal.
    nonisolated static func severity(for state: ProcessInfo.ThermalState) -> ThermalSeverity {
        switch state {
        case .nominal:  return .nominal
        case .fair:     return .fair
        case .serious:  return .serious
        case .critical: return .critical
        @unknown default: return .critical
        }
    }

    nonisolated static func label(for severity: ThermalSeverity) -> String {
        switch severity {
        case .nominal:
            return String(localized: "metrics.thermal.nominal",
                          defaultValue: "Nominal",
                          comment: "Thermal severity label shown in the Status screen's System block")
        case .fair:
            return String(localized: "metrics.thermal.fair",
                          defaultValue: "Fair",
                          comment: "Thermal severity label shown in the Status screen's System block")
        case .serious:
            return String(localized: "metrics.thermal.serious",
                          defaultValue: "Serious",
                          comment: "Thermal severity label shown in the Status screen's System block")
        case .critical:
            return String(localized: "metrics.thermal.critical",
                          defaultValue: "Critical",
                          comment: "Thermal severity label shown in the Status screen's System block")
        }
    }

    /// Format bytes as a 1-decimal GB string ("34.6"). Uses decimal GB
    /// (10^9) to match how macOS reports system memory in About This Mac
    /// — using GiB here would print 32 GB machines as "29.8".
    nonisolated static func formatBytesAsGB(_ bytes: UInt64) -> String {
        let gb = Double(bytes) / 1_000_000_000.0
        return String(format: "%.1f", gb)
    }
}

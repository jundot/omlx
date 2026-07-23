// Host-level stats behind the menubar's System Stats submenu. Everything
// here is public API only: Mach per-CPU tick counters (E/P cluster split
// via hw.perflevel sysctls), the IOKit accelerator performance dictionary
// for GPU utilization and in-use memory, vm_statistics64 for the memory
// breakdown, getloadavg, and kern.boottime. Power draw, core frequencies,
// and die temperature would need private frameworks and are intentionally
// out of scope; ProcessInfo's thermal state stands in for temperature.
//
// The sampler is polled by MenubarController only while the System Stats
// submenu is open — there is no background sampling cost. CPU usage comes
// from deltas of cumulative tick counters, so the first tick after a long
// gap still yields a valid average over that window.

import AppKit
import Darwin.Mach
import IOKit

/// The host metrics the menubar can surface, in display order.
enum SystemStatKind: String, CaseIterable, Hashable, Sendable {
    case cpu
    case gpu
    case memory

    var tag: String {
        switch self {
        case .cpu: return "CPU"
        case .gpu: return "GPU"
        case .memory: return "MEM"
        }
    }
}

struct SystemStatsSnapshot: Sendable {
    struct Memory: Sendable {
        var totalBytes: UInt64 = 0
        var wiredBytes: UInt64 = 0
        var activeBytes: UInt64 = 0
        var compressedBytes: UInt64 = 0
        /// Derived as total − (wired + active + compressed) so the four
        /// legend rows always sum to the machine total.
        var freeBytes: UInt64 = 0
        var usedBytes: UInt64 { wiredBytes + activeBytes + compressedBytes }
        var usedFraction: Double {
            totalBytes > 0 ? Double(usedBytes) / Double(totalBytes) : 0
        }
    }

    /// Fractions 0...1; nil while a reading is unavailable.
    var eCoreUsage: Double?
    var pCoreUsage: Double?
    /// All-core average, for the single-bar CPU menubar item.
    var cpuTotalUsage: Double?
    var gpuUsage: Double?
    var gpuMemoryInUseBytes: UInt64?
    var memory = Memory()
    var loadAverages: [Double] = []
    var uptimeSeconds: TimeInterval = 0
    var thermalState: ProcessInfo.ThermalState = .nominal

    var eHistory: [Double] = []
    var pHistory: [Double] = []
    var gpuHistory: [Double] = []
}

@MainActor
final class SystemStatsSampler {

    struct CPUTicks: Equatable, Sendable {
        var busy: UInt64
        var total: UInt64
    }

    private var previousTicks: [CPUTicks]?
    private var eHistory: [Double] = []
    private var pHistory: [Double] = []
    private var gpuHistory: [Double] = []
    private let eCoreCount = SystemStatsSampler.readECoreCount()

    func sample() -> SystemStatsSnapshot {
        var snapshot = SystemStatsSnapshot()

        if let ticks = Self.readCPUTicks() {
            if let previous = previousTicks,
               let usage = Self.clusterUsage(
                   previous: previous, current: ticks, eCoreCount: eCoreCount
               ) {
                snapshot.eCoreUsage = usage.e
                snapshot.pCoreUsage = usage.p
                snapshot.cpuTotalUsage = usage.total
                MenubarMetricsStore.append(&eHistory, usage.e)
                MenubarMetricsStore.append(&pHistory, usage.p)
            }
            previousTicks = ticks
        }

        if let gpu = Self.readGPUStatistics() {
            snapshot.gpuUsage = gpu.usage
            snapshot.gpuMemoryInUseBytes = gpu.memoryInUseBytes
            if let usage = gpu.usage {
                MenubarMetricsStore.append(&gpuHistory, usage)
            }
        }

        snapshot.memory = Self.readMemory()
        var loads = [Double](repeating: 0, count: 3)
        if getloadavg(&loads, 3) == 3 {
            snapshot.loadAverages = loads
        }
        snapshot.uptimeSeconds = Self.readUptime()
        snapshot.thermalState = ProcessInfo.processInfo.thermalState

        snapshot.eHistory = eHistory
        snapshot.pHistory = pHistory
        snapshot.gpuHistory = gpuHistory
        return snapshot
    }

    // MARK: - CPU

    /// Logical-CPU count of the efficiency cluster. Apple Silicon enumerates
    /// the E cluster first in Mach's per-CPU arrays; `hw.perflevel1` is the
    /// efficiency level whenever two performance levels exist. 0 (Intel or
    /// unknown) makes every core count as P.
    nonisolated private static func readECoreCount() -> Int {
        var levels: Int32 = 0
        var size = MemoryLayout<Int32>.size
        guard sysctlbyname("hw.nperflevels", &levels, &size, nil, 0) == 0,
              levels >= 2
        else {
            return 0
        }
        var count: Int32 = 0
        size = MemoryLayout<Int32>.size
        guard sysctlbyname("hw.perflevel1.logicalcpu", &count, &size, nil, 0) == 0 else {
            return 0
        }
        return Int(count)
    }

    /// Cumulative busy/total ticks per logical CPU, in kernel enumeration
    /// order (E cluster first on Apple Silicon).
    nonisolated private static func readCPUTicks() -> [CPUTicks]? {
        var cpuCount: natural_t = 0
        var info: processor_info_array_t?
        var infoCount: mach_msg_type_number_t = 0
        let result = host_processor_info(
            mach_host_self(),
            PROCESSOR_CPU_LOAD_INFO,
            &cpuCount,
            &info,
            &infoCount
        )
        guard result == KERN_SUCCESS, let info else { return nil }
        defer {
            vm_deallocate(
                mach_task_self_,
                vm_address_t(bitPattern: info),
                vm_size_t(infoCount) * vm_size_t(MemoryLayout<integer_t>.stride)
            )
        }

        let stride = Int(CPU_STATE_MAX)
        var ticks: [CPUTicks] = []
        ticks.reserveCapacity(Int(cpuCount))
        for cpu in 0..<Int(cpuCount) {
            let base = cpu * stride
            let user = UInt64(UInt32(bitPattern: info[base + Int(CPU_STATE_USER)]))
            let system = UInt64(UInt32(bitPattern: info[base + Int(CPU_STATE_SYSTEM)]))
            let nice = UInt64(UInt32(bitPattern: info[base + Int(CPU_STATE_NICE)]))
            let idle = UInt64(UInt32(bitPattern: info[base + Int(CPU_STATE_IDLE)]))
            let busy = user &+ system &+ nice
            ticks.append(CPUTicks(busy: busy, total: busy &+ idle))
        }
        return ticks
    }

    /// Average busy fraction per cluster between two tick readings. The
    /// counters are 32-bit in the kernel and wrap; a wrapped or shrunk
    /// counter invalidates that CPU's delta, and a reading with no usable
    /// delta at all yields nil. Exposed for unit tests.
    nonisolated static func clusterUsage(
        previous: [CPUTicks],
        current: [CPUTicks],
        eCoreCount: Int
    ) -> (e: Double, p: Double, total: Double)? {
        guard previous.count == current.count, !current.isEmpty else {
            return nil
        }
        var eBusy = 0.0, eTotal = 0.0
        var pBusy = 0.0, pTotal = 0.0
        for index in current.indices {
            let prev = previous[index]
            let cur = current[index]
            guard cur.total > prev.total, cur.busy >= prev.busy else { continue }
            let busy = Double(cur.busy - prev.busy)
            let total = Double(cur.total - prev.total)
            if index < eCoreCount {
                eBusy += busy
                eTotal += total
            } else {
                pBusy += busy
                pTotal += total
            }
        }
        let allBusy = eBusy + pBusy
        let allTotal = eTotal + pTotal
        guard allTotal > 0 else { return nil }
        return (
            e: eTotal > 0 ? min(1, eBusy / eTotal) : 0,
            p: pTotal > 0 ? min(1, pBusy / pTotal) : 0,
            total: min(1, allBusy / allTotal)
        )
    }

    // MARK: - GPU

    /// Reads the accelerator's PerformanceStatistics dictionary (public
    /// IOKit registry, no entitlements). Apple Silicon exposes one AGX
    /// service conforming to IOAccelerator with "Device Utilization %" and
    /// "In use system memory".
    nonisolated private static func readGPUStatistics()
        -> (usage: Double?, memoryInUseBytes: UInt64?)?
    {
        var iterator: io_iterator_t = 0
        guard IOServiceGetMatchingServices(
            kIOMainPortDefault,
            IOServiceMatching("IOAccelerator"),
            &iterator
        ) == KERN_SUCCESS else {
            return nil
        }
        defer { IOObjectRelease(iterator) }

        while true {
            let service = IOIteratorNext(iterator)
            guard service != 0 else { break }
            defer { IOObjectRelease(service) }

            var propertiesRef: Unmanaged<CFMutableDictionary>?
            guard IORegistryEntryCreateCFProperties(
                service, &propertiesRef, kCFAllocatorDefault, 0
            ) == KERN_SUCCESS,
                let properties = propertiesRef?.takeRetainedValue() as? [String: Any],
                let statistics = properties["PerformanceStatistics"] as? [String: Any]
            else {
                continue
            }

            let usage = (statistics["Device Utilization %"] as? NSNumber)
                .map { min(1, max(0, $0.doubleValue / 100)) }
            let memory = (statistics["In use system memory"] as? NSNumber)
                .map { UInt64(truncating: $0) }
            if usage != nil || memory != nil {
                return (usage: usage, memoryInUseBytes: memory)
            }
        }
        return nil
    }

    // MARK: - Memory

    nonisolated private static func readMemory() -> SystemStatsSnapshot.Memory {
        var memory = SystemStatsSnapshot.Memory()
        memory.totalBytes = ProcessInfo.processInfo.physicalMemory

        var size = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.stride / MemoryLayout<integer_t>.stride
        )
        var stats = vm_statistics64_data_t()
        let result = withUnsafeMutablePointer(to: &stats) { ptr -> kern_return_t in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(size)) { rebound in
                host_statistics64(mach_host_self(), HOST_VM_INFO64, rebound, &size)
            }
        }
        var pageSize: vm_size_t = 0
        guard result == KERN_SUCCESS,
              host_page_size(mach_host_self(), &pageSize) == KERN_SUCCESS
        else {
            return memory
        }

        let page = UInt64(pageSize)
        memory.wiredBytes = UInt64(stats.wire_count) * page
        memory.activeBytes = UInt64(stats.active_count) * page
        memory.compressedBytes = UInt64(stats.compressor_page_count) * page
        memory.freeBytes = memory.totalBytes > memory.usedBytes
            ? memory.totalBytes - memory.usedBytes
            : 0
        return memory
    }

    // MARK: - Misc readings

    /// Wall-clock uptime from kern.boottime (ProcessInfo.systemUptime stops
    /// while the machine sleeps, which reads oddly next to `uptime`).
    nonisolated private static func readUptime() -> TimeInterval {
        var boottime = timeval()
        var size = MemoryLayout<timeval>.size
        guard sysctlbyname("kern.boottime", &boottime, &size, nil, 0) == 0,
              boottime.tv_sec > 0
        else {
            return ProcessInfo.processInfo.systemUptime
        }
        let booted = TimeInterval(boottime.tv_sec)
        return max(0, Date().timeIntervalSince1970 - booted)
    }

    // MARK: - Formatting (pure, unit-tested)

    /// "8d 18h" / "5h 12m" / "42m"
    nonisolated static func formatUptime(_ seconds: TimeInterval) -> String {
        let minutes = Int(seconds) / 60
        let hours = minutes / 60
        let days = hours / 24
        if days >= 1 {
            return "\(days)d \(hours % 24)h"
        }
        if hours >= 1 {
            return "\(hours)h \(minutes % 60)m"
        }
        return "\(max(0, minutes))m"
    }

    /// "3.50 · 2.67 · 2.33"
    nonisolated static func formatLoadAverages(_ loads: [Double]) -> String {
        guard !loads.isEmpty else { return "–" }
        return loads.map { String(format: "%.2f", $0) }.joined(separator: " · ")
    }

    /// "237.29 GB" / "730 MB" — legend-row byte formatting, decimal units
    /// to match how macOS reports memory sizes.
    nonisolated static func formatBytes(_ bytes: UInt64) -> String {
        let gb = Double(bytes) / 1_000_000_000
        if gb >= 1 {
            return String(format: "%.2f GB", gb)
        }
        return String(format: "%.0f MB", Double(bytes) / 1_000_000)
    }
}

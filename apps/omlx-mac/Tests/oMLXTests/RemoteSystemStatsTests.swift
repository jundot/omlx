// Remote backend system metrics: `/api/system-metrics` decoding and the
// mapping into the sampler's `SystemStatsSnapshot` and the Status screen's
// RAM/thermal block.

import Foundation
import XCTest
@testable import oMLX

@MainActor
final class RemoteSystemStatsTests: XCTestCase {

    private let full = Data("""
    {
      "timestamp": 1793803000.5,
      "uptime_seconds": 12345.6,
      "cpu": {"total_percent": 23.4, "efficiency_percent": 10.0, "performance_percent": 30.0,
              "efficiency_cores": 4, "performance_cores": 8, "load_averages": [1.2, 1.1, 0.9]},
      "gpu": {"utilization_percent": 45.0, "memory_in_use_bytes": 123456789},
      "memory": {"total_bytes": 64000000000, "wired_bytes": 8000000000, "active_bytes": 12000000000,
                 "inactive_bytes": 5000000000, "compressed_bytes": 4000000000, "free_bytes": 35000000000},
      "thermal_state": "serious"
    }
    """.utf8)

    private let allNull = Data("""
    {
      "timestamp": null, "uptime_seconds": null,
      "cpu": {"total_percent": null, "efficiency_percent": null, "performance_percent": null,
              "efficiency_cores": null, "performance_cores": null, "load_averages": null},
      "gpu": {"utilization_percent": null, "memory_in_use_bytes": null},
      "memory": {"total_bytes": null, "wired_bytes": null, "active_bytes": null,
                 "inactive_bytes": null, "compressed_bytes": null, "free_bytes": null},
      "thermal_state": null
    }
    """.utf8)

    private func source() -> RemoteSystemStatsSource {
        RemoteSystemStatsSource(endpoint: OMLXEndpoint(
            scheme: "http", host: "studio.local", port: 8000, apiKey: "k", isRemote: true))
    }

    func testFullFixtureMapsToSnapshot() throws {
        let metrics = try RemoteSystemMetrics.decode(full)
        let snapshot = source().snapshot(from: metrics)

        XCTAssertEqual(try XCTUnwrap(snapshot.cpuTotalUsage), 0.234, accuracy: 1e-9)
        XCTAssertEqual(try XCTUnwrap(snapshot.eCoreUsage), 0.10, accuracy: 1e-9)
        XCTAssertEqual(try XCTUnwrap(snapshot.pCoreUsage), 0.30, accuracy: 1e-9)
        XCTAssertEqual(try XCTUnwrap(snapshot.gpuUsage), 0.45, accuracy: 1e-9)
        XCTAssertEqual(snapshot.gpuMemoryInUseBytes, 123_456_789)
        XCTAssertEqual(snapshot.loadAverages, [1.2, 1.1, 0.9])
        XCTAssertEqual(snapshot.uptimeSeconds, 12345.6, accuracy: 1e-9)
        XCTAssertEqual(snapshot.thermalState, .serious)
        XCTAssertEqual(snapshot.memory.totalBytes, 64_000_000_000)
        XCTAssertEqual(snapshot.memory.wiredBytes, 8_000_000_000)
        XCTAssertEqual(snapshot.memory.activeBytes, 12_000_000_000)
        XCTAssertEqual(snapshot.memory.compressedBytes, 4_000_000_000)
        XCTAssertEqual(snapshot.memory.usedBytes, 24_000_000_000)
        XCTAssertEqual(snapshot.memory.freeBytes, 40_000_000_000)
        XCTAssertEqual(snapshot.eHistory.count, 1)
        XCTAssertEqual(snapshot.pHistory.count, 1)
        XCTAssertEqual(snapshot.gpuHistory.count, 1)
    }

    func testAllNullFixtureMapsLikeAFailedLocalProbe() throws {
        let metrics = try RemoteSystemMetrics.decode(allNull)
        let snapshot = source().snapshot(from: metrics)

        XCTAssertNil(snapshot.cpuTotalUsage)
        XCTAssertNil(snapshot.eCoreUsage)
        XCTAssertNil(snapshot.pCoreUsage)
        XCTAssertNil(snapshot.gpuUsage)
        XCTAssertNil(snapshot.gpuMemoryInUseBytes)
        XCTAssertEqual(snapshot.loadAverages, [])
        XCTAssertEqual(snapshot.uptimeSeconds, 0)
        XCTAssertEqual(snapshot.thermalState, .nominal)
        XCTAssertEqual(snapshot.memory.totalBytes, 0)
        XCTAssertEqual(snapshot.memory.usedBytes, 0)
        XCTAssertEqual(snapshot.memory.freeBytes, 0)
        XCTAssertTrue(snapshot.eHistory.isEmpty)
        XCTAssertTrue(snapshot.pHistory.isEmpty)
        XCTAssertTrue(snapshot.gpuHistory.isEmpty)
    }

    func testMissingSectionsDecodeAsNil() throws {
        let metrics = try RemoteSystemMetrics.decode(Data("{}".utf8))
        let snapshot = source().snapshot(from: metrics)
        XCTAssertNil(snapshot.cpuTotalUsage)
        XCTAssertEqual(snapshot.thermalState, .nominal)
    }

    func testHistoriesAccumulateAndCapAtStoreCapacity() throws {
        let metrics = try RemoteSystemMetrics.decode(full)
        let source = source()
        var snapshot = SystemStatsSnapshot()
        for _ in 0..<(MenubarMetricsStore.historyCapacity + 5) {
            snapshot = source.snapshot(from: metrics)
        }
        XCTAssertEqual(snapshot.eHistory.count, MenubarMetricsStore.historyCapacity)
        XCTAssertEqual(snapshot.pHistory.count, MenubarMetricsStore.historyCapacity)
        XCTAssertEqual(snapshot.gpuHistory.count, MenubarMetricsStore.historyCapacity)
    }

    func testPercentsAreClampedToFractions() {
        XCTAssertEqual(RemoteSystemMetrics.fraction(150), 1)
        XCTAssertEqual(RemoteSystemMetrics.fraction(-5), 0)
        XCTAssertNil(RemoteSystemMetrics.fraction(nil))
    }

    func testThermalStrings() throws {
        for (text, expected): (String, ProcessInfo.ThermalState) in
            [("nominal", .nominal), ("fair", .fair), ("serious", .serious), ("critical", .critical), ("bogus", .nominal)] {
            let json = Data(#"{"thermal_state": "\#(text)"}"#.utf8)
            XCTAssertEqual(try RemoteSystemMetrics.decode(json).processThermalState, expected, text)
        }
    }

    // MARK: - Status screen

    func testStatusPollerUsesWiredPlusActivePlusCompressed() throws {
        let poller = SystemMetricsPoller()
        poller.applyRemote(try RemoteSystemMetrics.decode(full))
        XCTAssertEqual(poller.ramTotalBytes, 64_000_000_000)
        XCTAssertEqual(poller.ramUsedBytes, 8_000_000_000 + 12_000_000_000 + 4_000_000_000)
        XCTAssertEqual(poller.thermalState, .serious)
    }

    func testStatusPollerBlanksOnNilAndAllNull() throws {
        let poller = SystemMetricsPoller()
        poller.applyRemote(try RemoteSystemMetrics.decode(full))
        poller.applyRemote(try RemoteSystemMetrics.decode(allNull))
        XCTAssertNil(poller.ramTotalBytes)
        XCTAssertNil(poller.ramUsedBytes)
        XCTAssertEqual(poller.thermalState, .nominal)

        poller.applyRemote(try RemoteSystemMetrics.decode(full))
        poller.applyRemote(nil)
        XCTAssertNil(poller.ramTotalBytes)
        XCTAssertNil(poller.ramUsedBytes)
    }
}

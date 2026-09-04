// Unit coverage for the System Stats submenu's pure computation: E/P
// cluster usage from Mach tick deltas, and the formatting helpers the
// panels render.

import Foundation
import XCTest
@testable import oMLX

final class SystemStatsTests: XCTestCase {

    private typealias Ticks = SystemStatsSampler.CPUTicks

    // MARK: - Cluster usage

    func testClusterUsageSplitsECoresFirstAndAveragesPerCluster() throws {
        // 2 E-cores + 2 P-cores; deltas: E = 50%/100%, P = 0%/25%.
        let previous = [
            Ticks(busy: 100, total: 1_000),
            Ticks(busy: 100, total: 1_000),
            Ticks(busy: 100, total: 1_000),
            Ticks(busy: 100, total: 1_000),
        ]
        let current = [
            Ticks(busy: 150, total: 1_100),
            Ticks(busy: 200, total: 1_100),
            Ticks(busy: 100, total: 1_100),
            Ticks(busy: 125, total: 1_100),
        ]

        let usage = try XCTUnwrap(SystemStatsSampler.clusterUsage(
            previous: previous, current: current, eCoreCount: 2
        ))
        XCTAssertEqual(usage.e, 0.75, accuracy: 0.0001)
        XCTAssertEqual(usage.p, 0.125, accuracy: 0.0001)
        XCTAssertEqual(usage.total, 0.4375, accuracy: 0.0001,
                       "all-core average = 175 busy of 400 total ticks")
    }

    func testClusterUsageWithNoECoresCountsEverythingAsP() throws {
        let previous = [Ticks(busy: 0, total: 100)]
        let current = [Ticks(busy: 50, total: 200)]

        let usage = try XCTUnwrap(SystemStatsSampler.clusterUsage(
            previous: previous, current: current, eCoreCount: 0
        ))
        XCTAssertEqual(usage.e, 0)
        XCTAssertEqual(usage.p, 0.5, accuracy: 0.0001)
        XCTAssertEqual(usage.total, 0.5, accuracy: 0.0001)
    }

    // MARK: - Menubar bar glyph quantization

    func testBarLevelQuantizationGatesReRasters() {
        XCTAssertEqual(MenubarMetricGlyph.quantizedBarLevel(nil), -1)
        XCTAssertEqual(MenubarMetricGlyph.quantizedBarLevel(0), 0)
        XCTAssertEqual(MenubarMetricGlyph.quantizedBarLevel(1), 36)
        XCTAssertEqual(MenubarMetricGlyph.quantizedBarLevel(1.7), 36, "clamps above 100%")
        XCTAssertEqual(MenubarMetricGlyph.quantizedBarLevel(0.5), 18)

        // Sub-pixel jitter maps to the same signature — no repaint…
        XCTAssertEqual(
            MenubarMetricGlyph.barsSignature(
                segments: [("MEM", 0.500)], darkMenubar: true
            ),
            MenubarMetricGlyph.barsSignature(
                segments: [("MEM", 0.501)], darkMenubar: true
            )
        )
        // …while a visible change, appearance flip, or a different enabled
        // set does repaint.
        XCTAssertNotEqual(
            MenubarMetricGlyph.barsSignature(
                segments: [("MEM", 0.5)], darkMenubar: true
            ),
            MenubarMetricGlyph.barsSignature(
                segments: [("MEM", 0.55)], darkMenubar: true
            )
        )
        XCTAssertNotEqual(
            MenubarMetricGlyph.barsSignature(
                segments: [("MEM", 0.5)], darkMenubar: true
            ),
            MenubarMetricGlyph.barsSignature(
                segments: [("MEM", 0.5)], darkMenubar: false
            )
        )
        XCTAssertNotEqual(
            MenubarMetricGlyph.barsSignature(
                segments: [("CPU", 0.3), ("MEM", 0.5)], darkMenubar: true
            ),
            MenubarMetricGlyph.barsSignature(
                segments: [("MEM", 0.5)], darkMenubar: true
            )
        )
        // Identical multi-segment readings stay stable.
        XCTAssertEqual(
            MenubarMetricGlyph.barsSignature(
                segments: [("CPU", 0.3), ("GPU", nil), ("MEM", 0.5)],
                darkMenubar: true
            ),
            MenubarMetricGlyph.barsSignature(
                segments: [("CPU", 0.3), ("GPU", nil), ("MEM", 0.5)],
                darkMenubar: true
            )
        )
    }

    func testClusterUsageSkipsWrappedCountersAndRejectsEmptyDeltas() {
        // Wrapped counter (current < previous) must not poison the average…
        let previous = [Ticks(busy: 500, total: 1_000), Ticks(busy: 0, total: 100)]
        let current = [Ticks(busy: 10, total: 20), Ticks(busy: 100, total: 200)]
        let usage = SystemStatsSampler.clusterUsage(
            previous: previous, current: current, eCoreCount: 0
        )
        XCTAssertEqual(usage?.p ?? -1, 1.0, accuracy: 0.0001)

        // …and a reading with no usable delta at all yields nil.
        XCTAssertNil(SystemStatsSampler.clusterUsage(
            previous: previous, current: previous, eCoreCount: 0
        ))
        XCTAssertNil(SystemStatsSampler.clusterUsage(
            previous: [], current: [], eCoreCount: 0
        ))
        XCTAssertNil(SystemStatsSampler.clusterUsage(
            previous: previous, current: Array(current.prefix(1)), eCoreCount: 0
        ))
    }

    // MARK: - Formatting

    func testFormatUptimeBuckets() {
        XCTAssertEqual(SystemStatsSampler.formatUptime(42 * 60), "42m")
        XCTAssertEqual(SystemStatsSampler.formatUptime(5 * 3_600 + 12 * 60), "5h 12m")
        XCTAssertEqual(
            SystemStatsSampler.formatUptime(8 * 86_400 + 18 * 3_600),
            "8d 18h"
        )
        XCTAssertEqual(SystemStatsSampler.formatUptime(0), "0m")
    }

    func testFormatLoadAverages() {
        XCTAssertEqual(
            SystemStatsSampler.formatLoadAverages([3.5, 2.666, 2.334]),
            "3.50 · 2.67 · 2.33"
        )
        XCTAssertEqual(SystemStatsSampler.formatLoadAverages([]), "–")
    }

    func testFormatBytesUsesDecimalUnitsWithMBFallback() {
        XCTAssertEqual(SystemStatsSampler.formatBytes(730_000_000), "730 MB")
        XCTAssertEqual(SystemStatsSampler.formatBytes(8_550_000_000), "8.55 GB")
        XCTAssertEqual(SystemStatsSampler.formatBytes(237_290_000_000), "237.29 GB")
    }

    func testStatsFormatHelpers() {
        XCTAssertEqual(StatsFormat.percent(nil), "–")
        XCTAssertEqual(StatsFormat.percent(0.153), "15%")
        XCTAssertEqual(StatsFormat.percent(1.7), "100%", "fractions clamp to 100%")
        XCTAssertEqual(StatsFormat.wholeGB(512_000_000_000), "512")
        XCTAssertEqual(StatsFormat.window(sampleCount: 60, interval: 1.0), "60s")
        XCTAssertEqual(StatsFormat.window(sampleCount: 60, interval: 0.5), "30s")
        XCTAssertEqual(StatsFormat.window(sampleCount: 0, interval: 1.0), "")
    }

    func testMemorySnapshotDerivesFreeAndUsedFraction() {
        var memory = SystemStatsSnapshot.Memory()
        memory.totalBytes = 512_000_000_000
        memory.wiredBytes = 8_550_000_000
        memory.activeBytes = 237_290_000_000
        memory.compressedBytes = 730_000_000
        memory.freeBytes = memory.totalBytes - memory.usedBytes

        XCTAssertEqual(memory.usedBytes, 246_570_000_000)
        XCTAssertEqual(memory.usedFraction, 0.4816, accuracy: 0.001)
        XCTAssertEqual(memory.freeBytes, 265_430_000_000)
    }
}

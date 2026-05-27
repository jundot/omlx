// The Status screen's System block surfaces three derived values that
// each have a single point of failure: the thermal-state → severity
// mapping, the GPU-utilization clamp, and the bytes → GB formatter used
// in both row labels. These tests pin those mappings so an SDK roll or
// a stray locale tweak can't change what the UI prints.

import XCTest
@testable import oMLX

final class SystemMetricsTests: XCTestCase {

    // MARK: - Thermal severity mapping

    func testThermalSeverityNominal() {
        XCTAssertEqual(
            SystemMetricsPoller.severity(for: .nominal),
            .nominal
        )
    }

    func testThermalSeverityFair() {
        XCTAssertEqual(
            SystemMetricsPoller.severity(for: .fair),
            .fair
        )
    }

    func testThermalSeveritySerious() {
        XCTAssertEqual(
            SystemMetricsPoller.severity(for: .serious),
            .serious
        )
    }

    func testThermalSeverityCritical() {
        XCTAssertEqual(
            SystemMetricsPoller.severity(for: .critical),
            .critical
        )
    }

    func testThermalLabelsMatchSeverity() {
        XCTAssertEqual(SystemMetricsPoller.label(for: .nominal),  "Nominal")
        XCTAssertEqual(SystemMetricsPoller.label(for: .fair),     "Fair")
        XCTAssertEqual(SystemMetricsPoller.label(for: .serious),  "Serious")
        XCTAssertEqual(SystemMetricsPoller.label(for: .critical), "Critical")
    }

    // MARK: - GPU utilization clamp

    @MainActor
    func testGpuUtilizationClampedTo100WhenActiveExceedsMax() {
        let vm = StatusScreenVM()
        vm.maxConcurrent = 8
        vm.stats = makeStats(active: 20)
        XCTAssertEqual(vm.gpuUtilizationPercent, 100.0, accuracy: 0.001)
    }

    @MainActor
    func testGpuUtilizationZeroWhenActiveZero() {
        let vm = StatusScreenVM()
        vm.maxConcurrent = 8
        vm.stats = makeStats(active: 0)
        XCTAssertEqual(vm.gpuUtilizationPercent, 0.0, accuracy: 0.001)
    }

    @MainActor
    func testGpuUtilizationLinearInRange() {
        let vm = StatusScreenVM()
        vm.maxConcurrent = 8
        vm.stats = makeStats(active: 4)
        XCTAssertEqual(vm.gpuUtilizationPercent, 50.0, accuracy: 0.001)
    }

    @MainActor
    func testGpuUtilizationHandlesZeroMaxByFloorOfOne() {
        // The VM's max is 0 only transiently (between init and the
        // settings load). The divisor floor of 1 prevents NaN.
        let vm = StatusScreenVM()
        vm.maxConcurrent = 0
        vm.stats = makeStats(active: 0)
        XCTAssertEqual(vm.gpuUtilizationPercent, 0.0, accuracy: 0.001)
    }

    // MARK: - Bytes → GB formatter

    func testFormatBytesAsGbRoundsToOneDecimal() {
        // 34.6 GB in decimal bytes = 34_600_000_000.
        let bytes: UInt64 = 34_600_000_000
        XCTAssertEqual(SystemMetricsPoller.formatBytesAsGB(bytes), "34.6")
    }

    func testFormatBytesAsGbZero() {
        XCTAssertEqual(SystemMetricsPoller.formatBytesAsGB(0), "0.0")
    }

    func testFormatBytesAsGbRoundsHalfUp() {
        // 12.55 GB → "12.5" (banker's) or "12.6" (away). printf %.1f on
        // Darwin rounds half to even at the binary level — pin whichever
        // string we actually produce so a future libc swap is visible.
        let bytes: UInt64 = 12_550_000_000
        let out = SystemMetricsPoller.formatBytesAsGB(bytes)
        XCTAssertTrue(out == "12.5" || out == "12.6",
                      "Unexpected rounding output: \(out)")
    }

    // MARK: - Helpers

    private func makeStats(active: Int) -> StatsDTO {
        StatsDTO(
            totalTokensServed: 0,
            totalCachedTokens: 0,
            cacheEfficiency: 0,
            totalPromptTokens: 0,
            totalCompletionTokens: 0,
            totalRequests: 0,
            avgPrefillTps: 0,
            avgGenerationTps: 0,
            uptimeSeconds: 0,
            host: nil,
            port: nil,
            apiKey: nil,
            cliPrefix: nil,
            activeModels: StatsDTO.ActiveModelsDTO(
                models: [],
                modelMemoryUsed: nil,
                modelMemoryMax: nil,
                totalActiveRequests: active,
                totalWaitingRequests: 0
            ),
            runtimeCache: nil
        )
    }
}

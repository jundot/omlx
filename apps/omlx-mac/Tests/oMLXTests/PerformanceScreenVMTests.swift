// Effective-ceiling preview math for the Performance screen. The server's
// hard ceiling is min(static, dynamic, metal_cap); these tests pin the
// Swift mirror of that math (and the kernel wired-limit warning) against
// the field case from #1463: Custom 44 GB on a 48 GB Mac silently clamped
// to the 36 GB Metal cap.

import XCTest
@testable import oMLX

@MainActor
final class PerformanceScreenVMTests: XCTestCase {

    private static let gb = 1073741824.0

    private func systemInfo(
        totalGB: Double,
        metalCapGB: Double = 0,
        requestGB: Double = 0,
        freeGB: Double = 0,
        inactiveGB: Double = 0,
        activeGB: Double = 0,
        omlxGB: Double = 0
    ) -> GlobalSettingsDTO.SystemInfo {
        GlobalSettingsDTO.SystemInfo(
            totalMemoryBytes: Int64(totalGB * Self.gb),
            totalMemory: nil,
            omlxPhysFootprintBytes: Int64(omlxGB * Self.gb),
            freeMemoryBytes: Int64(freeGB * Self.gb),
            inactiveMemoryBytes: Int64(inactiveGB * Self.gb),
            activeMemoryBytes: Int64(activeGB * Self.gb),
            iogpuWiredLimitBytes: Int64(metalCapGB * Self.gb),
            omlxWiredLimitRequestBytes: Int64(requestGB * Self.gb)
        )
    }

    private func makeVM(
        tier: String,
        customCeiling: String = "",
        guardOn: Bool = true,
        info: GlobalSettingsDTO.SystemInfo?
    ) -> PerformanceScreenVM {
        let vm = PerformanceScreenVM()
        vm.prefillMemoryGuard = guardOn
        vm.memoryGuardTier = tier
        vm.memoryGuardCustomCeilingText = customCeiling
        vm.systemInfo = info
        return vm
    }

    // #1463 field case: 48 GB Mac, Apple default Metal cap 36 GB, custom
    // ceiling 44 GB. Static is 46 GB (total - 2), so the Metal cap binds.
    func testCustomCeilingClampedByMetalCap() throws {
        let vm = makeVM(
            tier: "custom",
            customCeiling: "44",
            info: systemInfo(totalGB: 48, metalCapGB: 36, requestGB: 46)
        )

        let breakdown = try XCTUnwrap(vm.memoryGuardBreakdown)
        XCTAssertTrue(breakdown.contains("44"))
        XCTAssertTrue(breakdown.contains("36.0"))
        XCTAssertTrue(breakdown.contains("kernel"))

        XCTAssertNotNil(vm.wiredLimitWarningText)
        // ceil(46 GiB / 1 MiB) = 47104
        XCTAssertEqual(vm.wiredLimitCommand, "sudo sysctl iogpu.wired_limit_mb=47104")
    }

    func testCustomCeilingUnderCapsIsNotClamped() throws {
        let vm = makeVM(
            tier: "custom",
            customCeiling: "20",
            info: systemInfo(totalGB: 48, metalCapGB: 36, requestGB: 36)
        )

        let breakdown = try XCTUnwrap(vm.memoryGuardBreakdown)
        XCTAssertTrue(breakdown.contains("20.0"))
        XCTAssertFalse(breakdown.contains("kernel"))

        // Kernel cap equals the request: no warning to show.
        XCTAssertNil(vm.wiredLimitWarningText)
    }

    func testCustomCeilingClampedByStaticReserve() throws {
        // 36 GB Mac, no Metal cap reported: static = total - 2 = 34 GB.
        let vm = makeVM(
            tier: "custom",
            customCeiling: "44",
            info: systemInfo(totalGB: 36)
        )

        let breakdown = try XCTUnwrap(vm.memoryGuardBreakdown)
        XCTAssertTrue(breakdown.contains("34.0"))
        XCTAssertFalse(breakdown.contains("kernel"))
    }

    func testBalancedTierBreakdownSumsReclaimLayers() throws {
        // dynamic = omlx 6 + free 8 + inactive 4 + active 10 * 0.5 = 23;
        // static = 48 - 6 = 42, metal 36 -> dynamic binds at 23.
        let vm = makeVM(
            tier: "balanced",
            info: systemInfo(
                totalGB: 48, metalCapGB: 36, requestGB: 36,
                freeGB: 8, inactiveGB: 4, activeGB: 10, omlxGB: 6
            )
        )

        let breakdown = try XCTUnwrap(vm.memoryGuardBreakdown)
        XCTAssertTrue(breakdown.contains("50"))
        XCTAssertTrue(breakdown.contains("5.0"))
        XCTAssertTrue(breakdown.contains("23.0"))
        XCTAssertFalse(breakdown.contains("kernel"))
    }

    func testPreviewHiddenWhenGuardOffOrCustomEmpty() {
        let info = systemInfo(totalGB: 48, metalCapGB: 36, requestGB: 46)

        let guardOff = makeVM(tier: "custom", customCeiling: "44", guardOn: false, info: info)
        XCTAssertNil(guardOff.memoryGuardBreakdown)
        XCTAssertNil(guardOff.wiredLimitWarningText)

        let emptyCustom = makeVM(tier: "custom", info: info)
        XCTAssertNil(emptyCustom.memoryGuardBreakdown)

        let noSnapshot = makeVM(tier: "custom", customCeiling: "44", info: nil)
        XCTAssertNil(noSnapshot.memoryGuardBreakdown)
        XCTAssertNil(noSnapshot.wiredLimitWarningText)
    }

    func testSystemInfoDecodesSnakeCaseMemoryFields() throws {
        let json = """
        {
            "total_memory_bytes": 51539607552,
            "total_memory": "48.0 GB",
            "omlx_phys_footprint_bytes": 1073741824,
            "free_memory_bytes": 2147483648,
            "inactive_memory_bytes": 3221225472,
            "active_memory_bytes": 4294967296,
            "iogpu_wired_limit_bytes": 38654705664,
            "omlx_wired_limit_request_bytes": 49392123904
        }
        """
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let sys = try decoder.decode(
            GlobalSettingsDTO.SystemInfo.self, from: Data(json.utf8)
        )
        XCTAssertEqual(sys.totalMemoryBytes, 51539607552)
        XCTAssertEqual(sys.omlxPhysFootprintBytes, 1073741824)
        XCTAssertEqual(sys.freeMemoryBytes, 2147483648)
        XCTAssertEqual(sys.inactiveMemoryBytes, 3221225472)
        XCTAssertEqual(sys.activeMemoryBytes, 4294967296)
        XCTAssertEqual(sys.iogpuWiredLimitBytes, 38654705664)
        XCTAssertEqual(sys.omlxWiredLimitRequestBytes, 49392123904)
    }
}

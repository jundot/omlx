// Effective-ceiling preview for the Performance screen. The server sends
// each tier's ceiling (system.memory_guard_preview); these tests pin how
// the screen renders it, the custom-ceiling clamp, and the kernel
// wired-limit warning against the field case from #1463: Custom 44 GB on a
// 48 GB Mac silently clamped to the 36 GB Metal cap.

import XCTest
@testable import oMLX

@MainActor
final class PerformanceScreenVMTests: XCTestCase {

    private static let gb = 1073741824.0

    private func tierPreview(
        reserveGB: Double = 0,
        freeGB: Double = 0,
        inactiveGB: Double = 0,
        otherGB: Double = 0,
        staticGB: Double = 0,
        metalCapGB: Double = 0,
        ceilingGB: Double = 0,
        binding: String = "dynamic"
    ) -> GlobalSettingsDTO.MemoryGuardTierPreview {
        func bytes(_ v: Double) -> Int64 { Int64(v * Self.gb) }
        return GlobalSettingsDTO.MemoryGuardTierPreview(
            reserveBytes: bytes(reserveGB),
            freeBytes: bytes(freeGB),
            inactiveBytes: bytes(inactiveGB),
            otherAppsBytes: bytes(otherGB),
            staticBytes: bytes(staticGB),
            dynamicBytes: nil,
            metalCapBytes: bytes(metalCapGB),
            ceilingBytes: bytes(ceilingGB),
            binding: binding
        )
    }

    private func systemInfo(
        totalGB: Double,
        metalCapGB: Double = 0,
        requestGB: Double = 0,
        preview: [String: GlobalSettingsDTO.MemoryGuardTierPreview]? = nil
    ) -> GlobalSettingsDTO.SystemInfo {
        GlobalSettingsDTO.SystemInfo(
            totalMemoryBytes: Int64(totalGB * Self.gb),
            totalMemory: nil,
            omlxPhysFootprintBytes: 0,
            freeMemoryBytes: 0,
            inactiveMemoryBytes: 0,
            activeMemoryBytes: 0,
            iogpuWiredLimitBytes: Int64(metalCapGB * Self.gb),
            omlxWiredLimitRequestBytes: Int64(requestGB * Self.gb),
            memoryGuardPreview: preview
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
            info: systemInfo(
                totalGB: 48, metalCapGB: 36, requestGB: 46,
                preview: ["custom": tierPreview(staticGB: 46, metalCapGB: 36)]
            )
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
            info: systemInfo(
                totalGB: 48, metalCapGB: 36, requestGB: 36,
                preview: ["custom": tierPreview(staticGB: 46, metalCapGB: 36)]
            )
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
            info: systemInfo(
                totalGB: 36, preview: ["custom": tierPreview(staticGB: 34)]
            )
        )

        let breakdown = try XCTUnwrap(vm.memoryGuardBreakdown)
        XCTAssertTrue(breakdown.contains("34.0"))
        XCTAssertFalse(breakdown.contains("kernel"))
    }

    func testBalancedTierBreakdownShowsServerTerms() throws {
        let vm = makeVM(
            tier: "balanced",
            info: systemInfo(
                totalGB: 48, metalCapGB: 36, requestGB: 36,
                preview: [
                    "balanced": tierPreview(
                        reserveGB: 3.8, freeGB: 8, inactiveGB: 4,
                        staticGB: 44.2, metalCapGB: 36, ceilingGB: 14.2
                    ),
                ]
            )
        )

        let breakdown = try XCTUnwrap(vm.memoryGuardBreakdown)
        XCTAssertTrue(breakdown.contains("8.0"))
        XCTAssertTrue(breakdown.contains("3.8"))
        XCTAssertTrue(breakdown.contains("14.2"))
        XCTAssertFalse(breakdown.contains("kernel"))
    }

    func testAggressiveTierBreakdownNamesKernelLimit() throws {
        let vm = makeVM(
            tier: "aggressive",
            info: systemInfo(
                totalGB: 48, metalCapGB: 36, requestGB: 46,
                preview: [
                    "aggressive": tierPreview(
                        reserveGB: 1.5, freeGB: 30, inactiveGB: 4, otherGB: 2,
                        staticGB: 46.5, metalCapGB: 36, ceilingGB: 36,
                        binding: "metal_cap"
                    ),
                ]
            )
        )

        let breakdown = try XCTUnwrap(vm.memoryGuardBreakdown)
        XCTAssertTrue(breakdown.contains("36.0"))
        XCTAssertTrue(breakdown.contains("kernel"))
    }

    func testPreviewHiddenWhenGuardOffOrCustomEmpty() {
        let info = systemInfo(
            totalGB: 48, metalCapGB: 36, requestGB: 46,
            preview: ["custom": tierPreview(staticGB: 46, metalCapGB: 36)]
        )

        let guardOff = makeVM(tier: "custom", customCeiling: "44", guardOn: false, info: info)
        XCTAssertNil(guardOff.memoryGuardBreakdown)
        XCTAssertNil(guardOff.wiredLimitWarningText)

        let emptyCustom = makeVM(tier: "custom", info: info)
        XCTAssertNil(emptyCustom.memoryGuardBreakdown)

        let noSnapshot = makeVM(tier: "custom", customCeiling: "44", info: nil)
        XCTAssertNil(noSnapshot.memoryGuardBreakdown)
        XCTAssertNil(noSnapshot.wiredLimitWarningText)
    }

    func testIdleTimeoutParsingCanonicalizesZeroAsDisabled() {
        let vm = PerformanceScreenVM()
        XCTAssertNil(vm.parsedIdleTimeout)

        vm.idleTimeoutText = "0"
        XCTAssertNil(vm.parsedIdleTimeout)

        vm.idleTimeoutText = "60"
        XCTAssertEqual(vm.parsedIdleTimeout, 60)
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
            "omlx_wired_limit_request_bytes": 49392123904,
            "memory_guard_preview": {
                "balanced": {
                    "reserve_bytes": 4080218931,
                    "other_apps_bytes": 0,
                    "ceiling_bytes": 15032385536,
                    "binding": "dynamic"
                }
            }
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
        let balanced = try XCTUnwrap(sys.memoryGuardPreview?["balanced"])
        XCTAssertEqual(balanced.reserveBytes, 4080218931)
        XCTAssertEqual(balanced.ceilingBytes, 15032385536)
        XCTAssertEqual(balanced.binding, "dynamic")
    }
}

extension PerformanceScreenVMTests {
    /// The privileged command is a fixed sysctl with a validated MiB integer.
    /// Invalid values are rejected before any AppleScript or admin dialog.
    func testPrivilegedWiredLimitValidation() throws {
        XCTAssertEqual(try PrivilegedActions.validatedWiredLimitMB(1024), 1024)
        for invalid in [0, -1, 1_048_577] {
            XCTAssertThrowsError(try PrivilegedActions.validatedWiredLimitMB(invalid)) { error in
                XCTAssertEqual(error as? PrivilegedActionError, .invalidValue)
            }
        }
    }

    /// SMJobBless answers "already installed, success" and never replaces the
    /// tool, so the app decides from its own ledger whether to bless, reuse or
    /// remove-and-bless. No admin prompt happens in this test.
    func testHelperInstallStateDecidesBlessReuseReplace() {
        XCTAssertEqual(
            PrivilegedHelper.installState(
                helperOnDisk: false, recordedVersion: nil, bundledVersion: 1
            ),
            .bless
        )
        XCTAssertEqual(
            PrivilegedHelper.installState(
                helperOnDisk: true, recordedVersion: 1, bundledVersion: 1
            ),
            .reuse
        )
        // A helper from an older build would otherwise stay installed forever.
        XCTAssertEqual(
            PrivilegedHelper.installState(
                helperOnDisk: true, recordedVersion: 0, bundledVersion: 1
            ),
            .replace
        )
        // A downgrade of the app also replaces: the ledger is authoritative.
        XCTAssertEqual(
            PrivilegedHelper.installState(
                helperOnDisk: true, recordedVersion: 2, bundledVersion: 1
            ),
            .replace
        )
        // Installed before the ledger existed: unknown version, so replace.
        XCTAssertEqual(
            PrivilegedHelper.installState(
                helperOnDisk: true, recordedVersion: nil, bundledVersion: 1
            ),
            .replace
        )
    }

    /// A cancelled prompt or a refused value must not open a second dialog.
    func testOnlyInstallFailuresFallBackToAppleScript() {
        XCTAssertFalse(
            PrivilegedHelper.shouldFallBackToAppleScript(PrivilegedActionError.cancelled)
        )
        XCTAssertFalse(
            PrivilegedHelper.shouldFallBackToAppleScript(PrivilegedActionError.invalidValue)
        )
        XCTAssertTrue(
            PrivilegedHelper.shouldFallBackToAppleScript(
                PrivilegedActionError.failed("SMJobBless refused to install the helper.")
            )
        )
        XCTAssertTrue(
            PrivilegedHelper.shouldFallBackToAppleScript(
                CocoaError(.fileNoSuchFile)
            )
        )
    }

    /// The helper identity is duplicated in the plists and the build script;
    /// the packaging test `test_privileged_helper_packaging.py` pins it there,
    /// this pins the Swift side the XPC connection uses.
    func testHelperIdentityMatchesTheInstalledLayout() {
        XCTAssertEqual(PrivilegedHelper.helperID, "app.omlx.helper")
        XCTAssertEqual(
            PrivilegedHelper.installedPath,
            "/Library/PrivilegedHelperTools/app.omlx.helper"
        )
        XCTAssertGreaterThanOrEqual(PrivilegedHelper.helperBundleVersion, 1)
    }
}

// MARK: - D6: the helper is a capability (five status states)

extension PerformanceScreenVMTests {
    private func freshDefaults() -> UserDefaults {
        let name = "omlx.helper.tests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defaults.removePersistentDomain(forName: name)
        return defaults
    }

    /// State 1 — not installed: Install is visible and enabling it blesses.
    func testHelperNotInstalledOffersInstallAndBlesses() throws {
        let status = PrivilegedHelper.status(
            helperOnDisk: false, recordedVersion: nil,
            wiredLimitCurrentMB: nil, wiredLimitRequestedMB: nil
        )
        XCTAssertFalse(status.installed)
        XCTAssertNil(status.version)
        let presentation = PrivilegedHelper.presentation(for: status)
        XCTAssertEqual(presentation.installAction, .install)
        XCTAssertTrue(presentation.willPrompt)

        var blessed = 0
        let defaults = freshDefaults()
        let state = try PrivilegedHelper.ensureInstalled(
            defaults: defaults, bundledVersion: 1, helperOnDisk: false,
            bless: { blessed += 1 },
            remove: { XCTFail("remove must not run on a fresh install") }
        )
        XCTAssertEqual(state, .bless)
        XCTAssertEqual(blessed, 1)
        XCTAssertEqual(defaults.object(forKey: PrivilegedHelper.installedVersionKey) as? Int, 1)
    }

    /// State 2 — installed, same version: state shown, no prompt (reuse).
    func testHelperSameVersionReusesWithoutPrompt() throws {
        let defaults = freshDefaults()
        defaults.set(1, forKey: PrivilegedHelper.installedVersionKey)
        let status = PrivilegedHelper.status(
            helperOnDisk: true, recordedVersion: 1,
            wiredLimitCurrentMB: nil, wiredLimitRequestedMB: nil
        )
        XCTAssertEqual(status.installState, .reuse)
        let presentation = PrivilegedHelper.presentation(for: status)
        XCTAssertEqual(presentation.installAction, .reinstall)
        XCTAssertFalse(presentation.willPrompt)

        var blessed = 0, removed = 0
        let state = try PrivilegedHelper.ensureInstalled(
            defaults: defaults, bundledVersion: 1, helperOnDisk: true,
            bless: { blessed += 1 }, remove: { removed += 1 }
        )
        XCTAssertEqual(state, .reuse)
        XCTAssertEqual(blessed, 0)
        XCTAssertEqual(removed, 0)
    }

    /// State 3 — installed, other version: reinstall offered (remove, bless).
    func testHelperOtherVersionReinstalls() throws {
        let defaults = freshDefaults()
        defaults.set(0, forKey: PrivilegedHelper.installedVersionKey)
        let status = PrivilegedHelper.status(
            helperOnDisk: true, recordedVersion: 0,
            wiredLimitCurrentMB: nil, wiredLimitRequestedMB: nil
        )
        XCTAssertEqual(status.installState, .replace)
        let presentation = PrivilegedHelper.presentation(for: status)
        XCTAssertEqual(presentation.installAction, .reinstall)
        XCTAssertTrue(presentation.willPrompt)

        var order: [String] = []
        let state = try PrivilegedHelper.ensureInstalled(
            defaults: defaults, bundledVersion: 1, helperOnDisk: true,
            bless: { order.append("bless") }, remove: { order.append("remove") }
        )
        XCTAssertEqual(state, .replace)
        XCTAssertEqual(order, ["remove", "bless"])
        XCTAssertEqual(defaults.object(forKey: PrivilegedHelper.installedVersionKey) as? Int, 1)
    }

    /// State 4 — limit == requested: Apply hidden, but Install still visible.
    func testHelperWiredLimitEqualHidesApplyButKeepsInstall() {
        let absent = PrivilegedHelper.status(
            helperOnDisk: false, recordedVersion: nil,
            wiredLimitCurrentMB: 122_880, wiredLimitRequestedMB: 122_880
        )
        XCTAssertFalse(absent.wiredLimitApplyVisible)
        XCTAssertEqual(PrivilegedHelper.presentation(for: absent).installAction, .install)

        let installed = PrivilegedHelper.status(
            helperOnDisk: true, recordedVersion: 1,
            wiredLimitCurrentMB: 122_880, wiredLimitRequestedMB: 122_880
        )
        XCTAssertFalse(installed.wiredLimitApplyVisible)
        XCTAssertEqual(PrivilegedHelper.presentation(for: installed).installAction, .reinstall)
    }

    /// State 5 — limit < requested: both Apply and Install visible.
    func testHelperWiredLimitBelowShowsBoth() {
        let status = PrivilegedHelper.status(
            helperOnDisk: true, recordedVersion: 1,
            wiredLimitCurrentMB: 29_696, wiredLimitRequestedMB: 122_880
        )
        XCTAssertTrue(status.wiredLimitApplyVisible)
        let presentation = PrivilegedHelper.presentation(for: status)
        XCTAssertTrue(presentation.applyVisible)
        XCTAssertEqual(presentation.installAction, .reinstall)
    }

    /// An absent tool never carries a stale ledger version.
    func testHelperStatusAbsenceClearsStaleLedger() {
        let status = PrivilegedHelper.status(
            helperOnDisk: false, recordedVersion: 9,
            wiredLimitCurrentMB: nil, wiredLimitRequestedMB: nil
        )
        XCTAssertFalse(status.installed)
        XCTAssertNil(status.version)
        XCTAssertEqual(status.installState, .bless)
    }

    /// The real reader converts the server's byte fields to MB and keeps the
    /// installed/version pair local even without a server snapshot.
    func testHelperCurrentStatusConvertsBytesToMB() {
        let defaults = freshDefaults()
        defaults.set(1, forKey: PrivilegedHelper.installedVersionKey)
        let info = systemInfo(totalGB: 128, metalCapGB: 120, requestGB: 120)
        let status = PrivilegedHelper.currentStatus(
            defaults: defaults, systemInfo: info, helperOnDisk: true
        )
        XCTAssertTrue(status.installed)
        XCTAssertEqual(status.version, 1)
        XCTAssertEqual(status.wiredLimitCurrentMB, 122_880)
        XCTAssertEqual(status.wiredLimitRequestedMB, 122_880)
        XCTAssertFalse(status.wiredLimitApplyVisible)

        let welcome = PrivilegedHelper.currentStatus(defaults: defaults, helperOnDisk: true)
        XCTAssertTrue(welcome.installed)
        XCTAssertNil(welcome.wiredLimitCurrentMB)
        XCTAssertNil(welcome.wiredLimitRequestedMB)
    }

    /// The uninstall's one helper dialog reports removed / cancelled / failed;
    /// a cancel or failure is never fatal and clears the ledger only on success.
    func testHelperRemovalOutcomes() {
        let defaults = freshDefaults()
        defaults.set(1, forKey: PrivilegedHelper.installedVersionKey)
        XCTAssertEqual(PrivilegedHelper.removeForUninstall(defaults: defaults, remove: {}), .removed)
        XCTAssertNil(defaults.object(forKey: PrivilegedHelper.installedVersionKey))

        XCTAssertEqual(
            PrivilegedHelper.removeForUninstall(defaults: defaults, remove: { throw PrivilegedActionError.cancelled }),
            .cancelled
        )
        let failed = PrivilegedHelper.removeForUninstall(
            defaults: defaults, remove: { throw PrivilegedActionError.failed("boom") }
        )
        if case .failed(let message) = failed {
            XCTAssertFalse(message.isEmpty)
        } else {
            XCTFail("expected .failed, got \(failed)")
        }
    }

    /// The Performance VM renders the permanent row and the separate wired-limit
    /// row from the injected status — independent of the memory-guard warning.
    func testPerformanceHelperRowsRenderFromStatus() {
        let vm = makeVM(tier: "balanced", guardOn: false, info: nil)
        vm.helperStatus = PrivilegedHelper.status(
            helperOnDisk: true, recordedVersion: 1,
            wiredLimitCurrentMB: 29_696, wiredLimitRequestedMB: 122_880
        )
        XCTAssertEqual(vm.helperStatusLine, "Helper: installed (v1)")
        XCTAssertEqual(vm.helperInstallTitle, "Reinstall…")
        XCTAssertEqual(vm.wiredLimitStatusLine, "29696 MB of 122880 MB requested")
        XCTAssertTrue(vm.wiredLimitApplyVisible)
        // The guard is off and there is no warning, yet Apply is still offered.
        XCTAssertNil(vm.wiredLimitWarningText)

        vm.helperStatus = PrivilegedHelper.status(
            helperOnDisk: false, recordedVersion: nil,
            wiredLimitCurrentMB: 122_880, wiredLimitRequestedMB: 122_880
        )
        XCTAssertEqual(vm.helperStatusLine, "Helper: not installed")
        XCTAssertEqual(vm.helperInstallTitle, "Install helper…")
        XCTAssertFalse(vm.wiredLimitApplyVisible)
    }
}

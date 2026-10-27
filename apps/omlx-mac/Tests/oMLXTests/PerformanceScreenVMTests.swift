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
    private func fairnessClient(port: Int = 18161) -> OMLXClient {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [DecodeFairnessURLProtocol.self]
        return OMLXClient(port: port, session: URLSession(configuration: configuration))
    }

    func testFairnessLoadResetAndCancelRetainDraft() async {
        let vm = PerformanceScreenVM()
        let client = fairnessClient()
        await vm.load(client: client)
        XCTAssertTrue(vm.supportsDecodeFairShare)
        XCTAssertEqual(vm.parsedDecodeFairShare, 0.5)
        vm.decodeFairShareText = "0.75"
        vm.decodeFairness = false
        await vm.resetDefaults(client: client)
        XCTAssertEqual(vm.parsedDecodeFairShare, 0.5)
        XCTAssertTrue(vm.decodeFairness)
        vm.cancelReset()
        XCTAssertEqual(vm.parsedDecodeFairShare, 0.75)
        XCTAssertFalse(vm.decodeFairness)
        await vm.save(client: client)
        XCTAssertNil(vm.lastError)
        XCTAssertFalse(vm.hasPendingChanges)
    }

    func testFairnessOldServerDoesNotSendUnsupportedFields() async {
        let vm = PerformanceScreenVM()
        let client = fairnessClient(port: 18162)
        await vm.load(client: client)
        XCTAssertFalse(vm.supportsDecodeFairShare)
        XCTAssertFalse(vm.supportsDecodeFairness)
        vm.decodeFairShareText = "0.25"
        vm.decodeFairness = false
        XCTAssertFalse(vm.hasPendingChanges)
        await vm.save(client: client)
        XCTAssertNil(vm.lastError)
    }

    func testFairnessFailedSaveRetainsPendingValue() async {
        let vm = PerformanceScreenVM()
        let client = fairnessClient(port: 18163)
        await vm.load(client: client)
        vm.decodeFairShareText = "0.25"
        await vm.save(client: client)
        XCTAssertNotNil(vm.lastError)
        XCTAssertEqual(vm.loadedDecodeFairShare, 0.5)
        XCTAssertEqual(vm.parsedDecodeFairShare, 0.25)
        XCTAssertTrue(vm.hasPendingChanges)
    }

    func testFairnessZeroAndInvalidInput() async {
        let vm = PerformanceScreenVM()
        let client = fairnessClient(port: 18164)
        await vm.load(client: client)
        for text in ["", "NaN", "infinity", "-0.1", "1.1"] {
            vm.decodeFairShareText = text
            await vm.save(client: client)
            XCTAssertNotNil(vm.lastError)
            XCTAssertNil(vm.parsedDecodeFairShare)
        }
        vm.decodeFairShareText = "0"
        await vm.save(client: client)
        XCTAssertNil(vm.lastError)
        XCTAssertEqual(vm.loadedDecodeFairShare, 0)
        XCTAssertFalse(vm.hasPendingChanges)
    }
}

private final class DecodeFairnessURLProtocol: URLProtocol, @unchecked Sendable {
    override class func canInit(with request: URLRequest) -> Bool {
        request.url?.host == "127.0.0.1" && (18161...18164).contains(request.url?.port ?? 0)
    }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        do {
            let port = request.url!.port!
            var status = 200
            let body: Data
            if request.httpMethod == "GET" {
                let path = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
                    .appendingPathComponent("Fixtures/global-settings.json")
                var json = try JSONSerialization.jsonObject(with: Data(contentsOf: path)) as! [String: Any]
                var scheduler = json["scheduler"] as! [String: Any]
                if port != 18162 {
                    scheduler["decode_fairness"] = true
                    scheduler["decode_fair_share"] = 0.5
                }
                json["scheduler"] = scheduler
                body = try JSONSerialization.data(withJSONObject: json)
            } else {
                var data = request.httpBody ?? Data()
                if data.isEmpty, let stream = request.httpBodyStream {
                    stream.open()
                    defer { stream.close() }
                    var buffer = [UInt8](repeating: 0, count: 4096)
                    while stream.hasBytesAvailable {
                        let count = stream.read(&buffer, maxLength: buffer.count)
                        if count <= 0 { break }
                        data.append(contentsOf: buffer.prefix(count))
                    }
                }
                let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
                switch port {
                case 18161:
                    XCTAssertEqual(json["decode_fair_share"] as? Double, 0.75)
                    XCTAssertEqual(json["decode_fairness"] as? Bool, false)
                    XCTAssertEqual(json.count, 2, "Only changed settings should be sent")
                case 18162:
                    XCTAssertNil(json["decode_fair_share"])
                    XCTAssertNil(json["decode_fairness"])
                case 18163:
                    XCTAssertEqual(json["decode_fair_share"] as? Double, 0.25)
                    status = 500
                default:
                    XCTAssertEqual(json["decode_fair_share"] as? Double, 0)
                    XCTAssertEqual(json.count, 1)
                }
                body = Data((status == 200 ? "{\"success\":true,\"message\":\"saved\",\"runtime_applied\":[\"decode_fair_share\"]}" : "{\"detail\":\"test save failure\"}").utf8)
            }
            let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: nil, headerFields: nil)!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: body)
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }
    override func stopLoading() {}
}

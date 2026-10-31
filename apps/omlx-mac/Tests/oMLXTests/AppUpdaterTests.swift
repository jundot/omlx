import XCTest
@testable import oMLX

@MainActor
final class AppUpdaterTests: XCTestCase {
    private enum TestError: Error, Equatable {
        case detachFailed
        case removalFailed
    }

    func testReadyIsNotifiedOnlyAfterMountedResourcesAreReleased() throws {
        var events: [String] = []

        try AppUpdater.finishStagedUpdate(
            detach: { events.append("detach") },
            removeTemporaryFiles: { events.append("remove temporary files") },
            notifyReady: { events.append("ready") }
        )

        XCTAssertEqual(events, [
            "detach",
            "remove temporary files",
            "ready",
        ])
    }

    func testDetachFailurePreventsTemporaryRemovalAndReadyNotification() {
        var events: [String] = []

        XCTAssertThrowsError(
            try AppUpdater.finishStagedUpdate(
                detach: {
                    events.append("detach")
                    throw TestError.detachFailed
                },
                removeTemporaryFiles: { events.append("remove temporary files") },
                notifyReady: { events.append("ready") }
            )
        ) { error in
            XCTAssertEqual(error as? TestError, .detachFailed)
        }

        XCTAssertEqual(events, ["detach"])
    }

    func testTemporaryRemovalFailurePreventsReadyNotification() {
        var events: [String] = []

        XCTAssertThrowsError(
            try AppUpdater.finishStagedUpdate(
                detach: { events.append("detach") },
                removeTemporaryFiles: {
                    events.append("remove temporary files")
                    throw TestError.removalFailed
                },
                notifyReady: { events.append("ready") }
            )
        ) { error in
            XCTAssertEqual(error as? TestError, .removalFailed)
        }

        XCTAssertEqual(events, ["detach", "remove temporary files"])
    }

    // MARK: - Download

    private func stubConfiguration() -> URLSessionConfiguration {
        let config = AppUpdater.downloadSessionConfiguration(allowsMeteredNetwork: true)
        config.protocolClasses = [RangeStubProtocol.self]
        return config
    }

    private func makeUpdater(
        onProgress: @escaping @MainActor (AppUpdater.Progress) -> Void = { _ in }
    ) -> AppUpdater {
        AppUpdater(
            dmgURL: URL(string: "http://stub.test/oMLX.dmg")!,
            version: "9.9.9",
            allowsMeteredNetwork: true,
            onProgress: onProgress,
            onError: { _ in },
            onReady: {}
        )
    }

    private func temporaryDMG() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("omlx-update-test-\(UUID().uuidString).dmg")
    }

    private func samplePayload(_ count: Int, seed: UInt8 = 31) -> Data {
        Data((0..<count).map { UInt8(truncatingIfNeeded: $0 &* Int(seed)) })
    }

    func testDownloadSessionHasNoShortWholeTransferDeadline() {
        let config = AppUpdater.downloadSessionConfiguration(allowsMeteredNetwork: true)

        // The request timeout is an idle timeout and is the stall detector. A
        // fixed one-hour resource timeout cut off slow but healthy transfers on
        // per-connection-throttled routes (#4199).
        XCTAssertEqual(config.timeoutIntervalForRequest, 30)
        XCTAssertEqual(config.timeoutIntervalForResource, 7 * 24 * 60 * 60)
    }

    func testDownloadSessionFollowsMeteredNetworkChoice() {
        let metered = AppUpdater.downloadSessionConfiguration(allowsMeteredNetwork: true)
        XCTAssertTrue(metered.allowsConstrainedNetworkAccess)
        XCTAssertTrue(metered.allowsExpensiveNetworkAccess)

        let unmetered = AppUpdater.downloadSessionConfiguration(allowsMeteredNetwork: false)
        XCTAssertFalse(unmetered.allowsConstrainedNetworkAccess)
        XCTAssertFalse(unmetered.allowsExpensiveNetworkAccess)
    }

    func testContentRangeParsing() {
        let range = AppUpdater.parseContentRange("bytes 5242880-20971519/20971520")
        XCTAssertEqual(range?.first, 5_242_880)
        XCTAssertEqual(range?.total, 20_971_520)

        let unknownTotal = AppUpdater.parseContentRange("bytes 10-19/*")
        XCTAssertEqual(unknownTotal?.first, 10)
        XCTAssertNil(unknownTotal?.total)

        XCTAssertNil(AppUpdater.parseContentRange(nil))
        XCTAssertNil(AppUpdater.parseContentRange("items 0-9/10"))
        XCTAssertNil(AppUpdater.parseContentRange("bytes */20"))
    }

    func testStalledDownloadResumesFromTheBytesOnDisk() async throws {
        let payload = samplePayload(600_000)
        RangeStubProtocol.reset(payload: payload)
        RangeStubProtocol.failFirstRequestAfterBytes = 250_000
        let dest = temporaryDMG()
        defer { try? FileManager.default.removeItem(at: dest) }
        var reported: [Int64] = []
        let updater = makeUpdater { progress in
            if case .downloading(_, let received, _) = progress { reported.append(received) }
        }

        try await updater.downloadDMG(to: dest, configuration: stubConfiguration(), resumeDelay: .zero)

        XCTAssertEqual(try Data(contentsOf: dest), payload)
        XCTAssertEqual(RangeStubProtocol.rangeHeaders, [nil, "bytes=250000-"])
        for _ in 0..<10 { await Task.yield() }
        XCTAssertEqual(reported.max(), 600_000, "progress counts the bytes on disk, not the attempt's")
    }

    func testChangedAssetOnResumeStartsTheFileOver() async throws {
        let old = samplePayload(300_000)
        let new = samplePayload(320_000, seed: 7)
        RangeStubProtocol.reset(payload: old)
        RangeStubProtocol.failFirstRequestAfterBytes = 100_000
        RangeStubProtocol.replacementAfterFirstRequest = new
        let dest = temporaryDMG()
        defer { try? FileManager.default.removeItem(at: dest) }

        try await makeUpdater().downloadDMG(to: dest, configuration: stubConfiguration(), resumeDelay: .zero)

        XCTAssertEqual(try Data(contentsOf: dest), new)
        XCTAssertEqual(RangeStubProtocol.rangeHeaders, [nil, "bytes=100000-"])
    }

    func testChangedAssetServedAsPartialContentStartsTheFileOver() async throws {
        // The CDN answers a Range request with 206 even when If-Range no
        // longer matches, so the ETag has to be compared explicitly.
        let old = samplePayload(300_000)
        let new = samplePayload(320_000, seed: 7)
        RangeStubProtocol.reset(payload: old)
        RangeStubProtocol.failFirstRequestAfterBytes = 100_000
        RangeStubProtocol.replacementAfterFirstRequest = new
        RangeStubProtocol.replacementHonoursRange = true
        let dest = temporaryDMG()
        defer { try? FileManager.default.removeItem(at: dest) }

        try await makeUpdater().downloadDMG(to: dest, configuration: stubConfiguration(), resumeDelay: .zero)

        XCTAssertEqual(try Data(contentsOf: dest), new)
        XCTAssertEqual(RangeStubProtocol.rangeHeaders, [nil, "bytes=100000-", nil])
    }

    func testDownloadGivesUpAfterRepeatedStallsWithoutProgress() async throws {
        RangeStubProtocol.reset(payload: samplePayload(1_000))
        RangeStubProtocol.failEveryRequest = true
        let dest = temporaryDMG()
        defer { try? FileManager.default.removeItem(at: dest) }

        do {
            try await makeUpdater().downloadDMG(to: dest, configuration: stubConfiguration(), resumeDelay: .zero)
            XCTFail("expected the download to fail")
        } catch {
            XCTAssertEqual((error as? URLError)?.code, .timedOut)
        }
        XCTAssertEqual(RangeStubProtocol.rangeHeaders.count, 1 + AppUpdater.maxDownloadResumes)
    }

    func testCancelledDownloadDoesNotStartOrResume() async throws {
        RangeStubProtocol.reset(payload: samplePayload(1_000))
        let dest = temporaryDMG()
        defer { try? FileManager.default.removeItem(at: dest) }
        let updater = makeUpdater()
        updater.cancel()

        do {
            try await updater.downloadDMG(to: dest, configuration: stubConfiguration(), resumeDelay: .zero)
            XCTFail("expected the download to be cancelled")
        } catch {
            guard case AppUpdater.UpdateError.cancelled = error else {
                return XCTFail("expected cancelled, got \(error)")
            }
        }
        XCTAssertEqual(RangeStubProtocol.rangeHeaders.count, 0)
    }

    // MARK: - Staged-app signature verification

    /// Builds a minimal .app bundle (plist + tiny executable) at path.
    private func makeAppBundle(at appURL: URL) throws {
        let macOS = appURL.appendingPathComponent("Contents/MacOS", isDirectory: true)
        try FileManager.default.createDirectory(at: macOS, withIntermediateDirectories: true)
        let infoPlist = """
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0"><dict>
            <key>CFBundleExecutable</key><string>oMLX</string>
            <key>CFBundleIdentifier</key><string>ai.omlx.update-test</string>
            <key>CFBundlePackageType</key><string>APPL</string>
        </dict></plist>
        """
        try infoPlist.write(
            to: appURL.appendingPathComponent("Contents/Info.plist"),
            atomically: true,
            encoding: .utf8
        )
        let shell = "#!/bin/sh\nexit 0\n"
        let executable = macOS.appendingPathComponent("oMLX")
        try shell.write(to: executable, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755], ofItemAtPath: executable.path
        )
    }

    private func codesign(_ arguments: [String]) -> Int32 {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/codesign")
        process.arguments = arguments
        process.standardOutput = Pipe()
        process.standardError = Pipe()
        do {
            try process.run()
        } catch {
            return -1
        }
        process.waitUntilExit()
        return process.terminationStatus
    }

    func testVerifyAppSignatureRejectsAdHocSignedBundle() throws {
        let appURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("omlx-update-test-signed-\(UUID().uuidString).app")
        try makeAppBundle(at: appURL)
        defer { try? FileManager.default.removeItem(at: appURL) }

        XCTAssertEqual(codesign(["-f", "-s", "-", appURL.path]), 0, "test setup: ad-hoc sign failed")

        XCTAssertThrowsError(
            try AppUpdater.verifyAppSignature(at: appURL.path)
        ) { error in
            guard case AppUpdater.UpdateError.signatureInvalid = error else {
                return XCTFail("expected signatureInvalid, got \(error)")
            }
        }
    }

    func testVerifyAppSignatureRejectsUnsignedBundle() throws {
        let appURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("omlx-update-test-unsigned-\(UUID().uuidString).app")
        try makeAppBundle(at: appURL)
        defer { try? FileManager.default.removeItem(at: appURL) }

        XCTAssertThrowsError(
            try AppUpdater.verifyAppSignature(at: appURL.path)
        ) { error in
            guard case AppUpdater.UpdateError.signatureInvalid = error else {
                return XCTFail("expected signatureInvalid, got \(error)")
            }
        }
    }
}

/// Serves a fixed payload for `http://stub.test/…` and honours `Range`, so the
/// updater's resume loop runs without a network. The first request can be cut
/// off after a number of bytes with a timeout error, and the asset can change
/// between requests.
private final class RangeStubProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) private static var payload = Data()
    nonisolated(unsafe) static var failFirstRequestAfterBytes: Int?
    nonisolated(unsafe) static var failEveryRequest = false
    nonisolated(unsafe) static var replacementAfterFirstRequest: Data?
    /// With a replacement: serve the requested range of it (206, new ETag)
    /// instead of ignoring the range (200), like a server that ignores If-Range.
    nonisolated(unsafe) static var replacementHonoursRange = false
    nonisolated(unsafe) private(set) static var rangeHeaders: [String?] = []

    static func reset(payload: Data) {
        self.payload = payload
        failFirstRequestAfterBytes = nil
        failEveryRequest = false
        replacementAfterFirstRequest = nil
        replacementHonoursRange = false
        rangeHeaders = []
    }

    override class func canInit(with request: URLRequest) -> Bool {
        request.url?.host == "stub.test"
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        guard let client, let url = request.url else { return }
        let rangeHeader = request.value(forHTTPHeaderField: "Range")
        Self.rangeHeaders.append(rangeHeader)
        let isFirstRequest = Self.rangeHeaders.count == 1

        var payload = Self.payload
        var etag = "\"v1\""
        var start = 0
        if let rangeHeader, rangeHeader.hasPrefix("bytes="), rangeHeader.hasSuffix("-"),
           let first = Int(rangeHeader.dropFirst(6).dropLast()) {
            start = first
        }
        if !isFirstRequest, let replacement = Self.replacementAfterFirstRequest {
            payload = replacement
            etag = "\"v2\""
            if !Self.replacementHonoursRange { start = 0 }
        }

        let body = Data(payload[start...])
        var headers = ["ETag": etag, "Content-Type": "application/octet-stream", "Content-Length": "\(body.count)"]
        let status: Int
        if start > 0 {
            status = 206
            headers["Content-Range"] = "bytes \(start)-\(payload.count - 1)/\(payload.count)"
        } else {
            status = 200
        }
        let response = HTTPURLResponse(url: url, statusCode: status, httpVersion: "HTTP/1.1", headerFields: headers)!
        client.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)

        if Self.failEveryRequest {
            client.urlProtocol(self, didFailWithError: URLError(.timedOut))
            return
        }
        if isFirstRequest, let cut = Self.failFirstRequestAfterBytes {
            client.urlProtocol(self, didLoad: body.prefix(cut))
            // Let the bytes reach the delegate before the connection "drops",
            // as they would on a real socket.
            DispatchQueue.global().asyncAfter(deadline: .now() + 0.2) {
                guard let client = self.client else { return }
                client.urlProtocol(self, didFailWithError: URLError(.timedOut))
            }
            return
        }
        client.urlProtocol(self, didLoad: body)
        client.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

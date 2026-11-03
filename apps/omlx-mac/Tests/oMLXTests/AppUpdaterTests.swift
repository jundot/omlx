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

    func testStalledDownloadResumesFromTheBytesOnDisk() async throws {
        let payload = Data((0..<600_000).map { UInt8(truncatingIfNeeded: $0 &* 31) })
        RangeStubProtocol.payload = payload
        RangeStubProtocol.ranges = []
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [RangeStubProtocol.self]
        let dest = FileManager.default.temporaryDirectory
            .appendingPathComponent("omlx-update-test-\(UUID().uuidString).dmg")
        defer { try? FileManager.default.removeItem(at: dest) }
        let updater = AppUpdater(
            dmgURL: URL(string: "https://stub.test/oMLX.dmg")!,
            version: "9.9.9",
            allowsMeteredNetwork: true,
            onProgress: { _ in },
            onError: { _ in },
            onReady: {}
        )

        try await updater.downloadDMG(to: dest, configuration: config)

        XCTAssertEqual(RangeStubProtocol.ranges, [nil, "bytes=250000-"])
        XCTAssertEqual(try Data(contentsOf: dest), payload)
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

/// Serves `payload` with `Range` support and ends the first response after 250,000 bytes.
private final class RangeStubProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var payload = Data()
    nonisolated(unsafe) static var ranges: [String?] = []

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func stopLoading() {}

    override func startLoading() {
        let range = request.value(forHTTPHeaderField: "Range")
        Self.ranges.append(range)
        let start = range.flatMap { Int($0.dropFirst(6).dropLast()) } ?? 0
        let body = Self.payload[start...]
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: start > 0 ? 206 : 200,
            httpVersion: "HTTP/1.1",
            headerFields: ["ETag": "\"v1\"", "Content-Length": "\(body.count)"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Self.ranges.count == 1 ? body.prefix(250_000) : body)
        client?.urlProtocolDidFinishLoading(self)
    }
}

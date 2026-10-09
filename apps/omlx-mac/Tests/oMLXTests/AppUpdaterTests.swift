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

    // MARK: - Staged-app signature verification

    private func makeUpdater() -> AppUpdater {
        AppUpdater(
            dmgURL: URL(fileURLWithPath: "/tmp/unused.dmg"),
            version: "test",
            onProgress: { _ in },
            onError: { _ in },
            onReady: {}
        )
    }

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
            try makeUpdater().verifyAppSignature(at: appURL.path)
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
            try makeUpdater().verifyAppSignature(at: appURL.path)
        ) { error in
            guard case AppUpdater.UpdateError.signatureInvalid = error else {
                return XCTFail("expected signatureInvalid, got \(error)")
            }
        }
    }
}

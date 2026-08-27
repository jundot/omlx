import Darwin
import XCTest
@testable import oMLX

final class UpdateInstallerTests: XCTestCase {
    private var temporaryDirectory: URL!

    override func setUpWithError() throws {
        temporaryDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("omlx update $HOME \"quoted\" \(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: temporaryDirectory,
            withIntermediateDirectories: true
        )
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: temporaryDirectory)
    }

    func testWorkerRequestPreservesPathsWithoutShellParsing() throws {
        let live = temporaryDirectory.appendingPathComponent("oMLX.app")
        let staged = temporaryDirectory.appendingPathComponent(
            UpdateInstaller.stagedAppName
        )

        let request = try XCTUnwrap(UpdateInstaller.workerRequest(from: [
            "/Applications/oMLX.app/Contents/MacOS/oMLX",
            UpdateInstaller.workerModeArgument,
            "1234",
            live.path,
            staged.path,
        ]))

        XCTAssertEqual(request.parentPID, 1234)
        XCTAssertEqual(request.liveApp, live.standardizedFileURL)
        XCTAssertEqual(request.stagedApp, staged.standardizedFileURL)
    }

    func testLaunchAgentRunsOnceWithoutKeepAlive() throws {
        let live = temporaryDirectory.appendingPathComponent("oMLX.app")
        let staged = temporaryDirectory.appendingPathComponent(
            UpdateInstaller.stagedAppName
        )
        let executable = live.appendingPathComponent("Contents/MacOS/oMLX")

        let plist = UpdateInstaller.launchAgentPropertyList(
            label: "app.omlx.updater.test",
            executable: executable,
            parentPID: 1234,
            liveApp: live,
            stagedApp: staged
        )

        XCTAssertEqual(plist["RunAtLoad"] as? Bool, true)
        XCTAssertEqual(plist["KeepAlive"] as? Bool, false)
        XCTAssertEqual(plist["ProgramArguments"] as? [String], [
            executable.path,
            UpdateInstaller.workerModeArgument,
            "1234",
            live.path,
            staged.path,
        ])
    }

    func testAtomicSwapExchangesCompleteBundles() throws {
        let live = try makeBundle(name: "oMLX.app", marker: "old")
        let staged = try makeBundle(
            name: UpdateInstaller.stagedAppName,
            marker: "new"
        )

        try UpdateInstaller.atomicSwap(liveApp: live, stagedApp: staged)

        XCTAssertEqual(try marker(in: live), "new")
        XCTAssertEqual(try marker(in: staged), "old")
    }

    func testAtomicSwapFailureLeavesLiveBundleUntouched() throws {
        let live = try makeBundle(name: "oMLX.app", marker: "old")
        let missingStaged = temporaryDirectory.appendingPathComponent(
            UpdateInstaller.stagedAppName
        )

        XCTAssertThrowsError(
            try UpdateInstaller.atomicSwap(
                liveApp: live,
                stagedApp: missingStaged
            )
        )
        XCTAssertEqual(try marker(in: live), "old")
        XCTAssertFalse(FileManager.default.fileExists(atPath: missingStaged.path))
    }

    func testFailedUpdatedAppRelaunchRollsBackToPreviousBundle() throws {
        let live = try makeBundle(name: "oMLX.app", marker: "old")
        let staged = try makeBundle(
            name: UpdateInstaller.stagedAppName,
            marker: "new"
        )

        try UpdateInstaller.replaceAndRelaunch(
            liveApp: live,
            stagedApp: staged,
            relaunchAction: { app in
                if try self.marker(in: app) == "new" {
                    throw CocoaError(.executableNotLoadable)
                }
            }
        )
        XCTAssertEqual(try marker(in: live), "old")
        XCTAssertEqual(try marker(in: staged), "new")
    }

    func testWaitForProcessExitObservesIndependentProcess() throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/sleep")
        process.arguments = ["0.2"]
        try process.run()

        XCTAssertTrue(
            UpdateInstaller.waitForProcessExit(
                process.processIdentifier,
                timeout: 2
            )
        )
    }

    func testUpdaterMountKeepsImageVerificationEnabled() {
        let dmg = URL(fileURLWithPath: "/tmp/oMLX release.dmg")
        let arguments = AppUpdater.mountArguments(dmg)

        XCTAssertTrue(arguments.contains("-readonly"))
        XCTAssertFalse(arguments.contains("-noverify"))
        XCTAssertEqual(arguments.last, dmg.path)
    }

    func testUpdaterAssessesDiskImageAsPrimarySignature() {
        let dmg = URL(fileURLWithPath: "/tmp/oMLX.dmg")
        XCTAssertEqual(
            AppUpdater.diskImageAssessmentArguments(dmg),
            [
                "--assess", "--type", "open", "--verbose=2",
                "--context", "context:primary-signature", dmg.path,
            ]
        )
    }

    func testDeveloperIDRequirementPinsBundleTeamAndCertificateClass() throws {
        let requirement = try AppUpdater.developerIDRequirement(
            for: .init(bundleIdentifier: "app.omlx", teamIdentifier: "AB12CD34EF")
        )

        XCTAssertTrue(requirement.contains("identifier \"app.omlx\""))
        XCTAssertTrue(requirement.contains("subject.OU] = \"AB12CD34EF\""))
        XCTAssertTrue(requirement.contains("1.2.840.113635.100.6.1.13"))
        XCTAssertTrue(requirement.contains("1.2.840.113635.100.6.2.6"))
    }

    func testDeveloperIDRequirementRejectsInjectedIdentityValues() {
        XCTAssertThrowsError(
            try AppUpdater.developerIDRequirement(
                for: .init(
                    bundleIdentifier: "app.omlx\" or true",
                    teamIdentifier: "AB12CD34EF"
                )
            )
        )
        XCTAssertThrowsError(
            try AppUpdater.developerIDRequirement(
                for: .init(bundleIdentifier: "app.omlx", teamIdentifier: "not-a-team")
            )
        )
    }

    private func makeBundle(name: String, marker: String) throws -> URL {
        let bundle = temporaryDirectory.appendingPathComponent(name)
        try FileManager.default.createDirectory(
            at: bundle,
            withIntermediateDirectories: true
        )
        try Data(marker.utf8).write(to: bundle.appendingPathComponent("marker"))
        return bundle
    }

    private func marker(in bundle: URL) throws -> String {
        try String(
            contentsOf: bundle.appendingPathComponent("marker"),
            encoding: .utf8
        )
    }
}

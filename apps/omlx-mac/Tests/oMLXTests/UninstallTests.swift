// D1 — the uninstall model layer, driven over injected paths. Nothing real is
// deleted: every fixture lives in a per-test temporary directory, and the
// runner's side effects go through a recording seam. These pin the frozen
// defaults (models + identity kept), the per-directory model opt-in, the byte
// summary, and "a cancelled helper dialog leaves the rest done".

import XCTest
@testable import oMLX

final class UninstallTests: XCTestCase {
    private var roots: [URL] = []

    override func tearDown() {
        let fm = FileManager.default
        for root in roots { try? fm.removeItem(at: root) }
        roots.removeAll()
        super.tearDown()
    }

    // MARK: Fixture

    private func makeFixture() throws -> UninstallLocator {
        let fm = FileManager.default
        let root = fm.temporaryDirectory
            .appendingPathComponent("omlx-uninstall-tests-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        roots.append(root)

        func dir(_ name: String) throws -> URL {
            let url = root.appendingPathComponent(name, isDirectory: true)
            try fm.createDirectory(at: url, withIntermediateDirectories: true)
            return url
        }
        func file(_ url: URL, _ count: Int, _ byte: UInt8 = 0x41) throws {
            try Data(repeating: byte, count: count).write(to: url)
        }

        let app = try dir("oMLX.app")
        try file(app.appendingPathComponent("oMLX"), 100)

        let tool = root.appendingPathComponent("app.omlx.helper")
        let plist = root.appendingPathComponent("app.omlx.helper.plist")
        try file(tool, 50, 0x42)
        try file(plist, 30, 0x43)

        let support = try dir("AppSupport")
        try file(support.appendingPathComponent("runtime.bin"), 200, 0x44)

        let base = try dir(".omlx")
        try file(base.appendingPathComponent("settings.json"), 10, 0x45)
        try file(base.appendingPathComponent("model_settings.json"), 10, 0x46)
        let logs = try dir(".omlx/logs")
        try file(logs.appendingPathComponent("server.log"), 10, 0x47)
        let cluster = try dir(".omlx/cluster")
        try file(cluster.appendingPathComponent("identity.json"), 16, 0x48)
        try file(cluster.appendingPathComponent("devices.json"), 20, 0x49)
        try file(cluster.appendingPathComponent("deployments.json"), 20, 0x4A)

        let models1 = try dir("Models1")
        let models2 = try dir("Models2")
        try file(models1.appendingPathComponent("model.gguf"), 500, 0x4B)
        try file(models2.appendingPathComponent("model.gguf"), 300, 0x4C)

        let diag = try dir("Logs-oMLX")
        try file(diag.appendingPathComponent("diag.log"), 10, 0x4D)
        let caches = try dir("Caches-oMLX")

        return UninstallLocator(
            appBundleURL: app,
            helperToolPath: tool.path,
            helperDaemonPlistPath: plist.path,
            appSupportURL: support,
            basePath: base.path,
            modelDirs: [models1.path, models2.path],
            diagnosticsURL: diag,
            cachesURL: caches,
            savedStateURLs: [],
            lsregisterPath: "/usr/bin/true"
        )
    }

    // MARK: Recording seam

    private final class Recorder {
        var trashed: [URL] = []
        var deleted: [URL] = []
        var unregistered: [URL] = []
        var gc = 0, dock = 0, quit = 0
        var helperOutcome: HelperRemovalOutcome = .removed
        var deleteHelperFiles = false
        var locator: UninstallLocator?

        func actions(_ fm: FileManager) -> UninstallActions {
            UninstallActions(
                trash: { [self] url in trashed.append(url); try fm.removeItem(at: url) },
                delete: { [self] url in deleted.append(url); try fm.removeItem(at: url) },
                removeHelper: { [self] in
                    if deleteHelperFiles, let l = locator {
                        try? fm.removeItem(atPath: l.helperToolPath)
                        try? fm.removeItem(atPath: l.helperDaemonPlistPath)
                    }
                    return helperOutcome
                },
                unregisterBundle: { [self] url in unregistered.append(url) },
                garbageCollectLaunchServices: { [self] in gc += 1 },
                repaintDock: { [self] in dock += 1 },
                quitApp: { [self] in quit += 1 }
            )
        }
    }

    // MARK: Plan defaults

    func testDefaultPlanRemovesItsScopesAndKeepsModelsAndIdentity() throws {
        let locator = try makeFixture()
        let plan = UninstallPlanner.build(locator: locator, options: .default)
        let scopes = Set(plan.items.map(\.scope))
        XCTAssertEqual(scopes, ["app", "helper", "runtime", "settings", "clusterState", "diagnostics"])
        // Identity and models are never in the default plan.
        XCTAssertFalse(plan.items.contains { $0.url == locator.identityURL })
        XCTAssertFalse(scopes.contains("models"))
        XCTAssertFalse(scopes.contains("identity"))
        XCTAssertEqual(plan.keptModelDirs, locator.modelDirs)
        XCTAssertTrue(plan.keepsIdentity)
        XCTAssertTrue(plan.removesApp)
        XCTAssertGreaterThan(plan.totalBytes, 0)
    }

    func testClusterStateExcludesIdentityEvenWhenRemoved() throws {
        let locator = try makeFixture()
        var options = UninstallOptions.default
        options.removeIdentity = true
        let plan = UninstallPlanner.build(locator: locator, options: options)
        // Identity appears exactly once, under its own scope.
        let identityItems = plan.items.filter { $0.url == locator.identityURL }
        XCTAssertEqual(identityItems.count, 1)
        XCTAssertEqual(identityItems.first?.scope, "identity")
        XCTAssertFalse(plan.keepsIdentity)
        // clusterState still carries the other cluster files.
        let clusterState = plan.items.filter { $0.scope == "clusterState" }
        XCTAssertEqual(clusterState.count, 2)
    }

    func testModelDeletionIsOptInPerDirectory() throws {
        let locator = try makeFixture()
        var options = UninstallOptions.default
        options.modelDirsToRemove = [locator.modelDirs[0]]
        let plan = UninstallPlanner.build(locator: locator, options: options)
        let modelItems = plan.items.filter { $0.scope == "models" }
        XCTAssertEqual(modelItems.count, 1)
        XCTAssertEqual(modelItems.first?.url.path, locator.modelDirs[0])
        XCTAssertEqual(plan.keptModelDirs, [locator.modelDirs[1]])
    }

    func testKeepSwitchesRemoveTheirScope() throws {
        let locator = try makeFixture()
        var options = UninstallOptions.default
        options.removeApp = false
        options.removeHelper = false
        options.removeRuntime = false
        options.removeSettings = false
        options.removeClusterState = false
        options.removeDiagnostics = false
        let plan = UninstallPlanner.build(locator: locator, options: options)
        XCTAssertTrue(plan.items.isEmpty)
        XCTAssertFalse(plan.removesApp)
        XCTAssertEqual(plan.keptModelDirs, locator.modelDirs)
        XCTAssertTrue(plan.keepsIdentity)
    }

    func testByteSummaryCountsRegularFiles() throws {
        let locator = try makeFixture()
        // The app bundle holds one 100-byte file.
        XCTAssertEqual(UninstallPlanner.size(of: locator.appBundleURL), 100)
        let plan = UninstallPlanner.build(locator: locator, options: .default)
        let appItem = plan.items.first { $0.scope == "app" }
        XCTAssertEqual(appItem?.bytes, 100)
        // helper = tool(50) + plist(30).
        XCTAssertEqual(plan.items.first { $0.scope == "helper" }?.bytes, 80)
    }

    // MARK: Runner

    func testRunnerDefaultRemovesScopesAndSparesModelsAndIdentity() throws {
        let fm = FileManager.default
        let locator = try makeFixture()
        let recorder = Recorder()
        recorder.locator = locator
        recorder.deleteHelperFiles = true
        let plan = UninstallPlanner.build(locator: locator, options: .default)
        let report = UninstallRunner.run(plan: plan, actions: recorder.actions(fm))

        XCTAssertEqual(
            Set(report.removedScopes),
            ["app", "helper", "runtime", "settings", "clusterState", "diagnostics"]
        )
        XCTAssertEqual(report.helperOutcome, .removed)
        XCTAssertFalse(report.helperLeftInPlace)
        XCTAssertTrue(report.errors.isEmpty)
        XCTAssertTrue(report.didQuit)
        XCTAssertEqual(recorder.quit, 1)
        // Models and identity survive on disk.
        XCTAssertTrue(fm.fileExists(atPath: locator.identityURL.path))
        for dir in locator.modelDirs { XCTAssertTrue(fm.fileExists(atPath: dir)) }
        // The app bundle and helper files are gone.
        XCTAssertFalse(fm.fileExists(atPath: locator.appBundleURL.path))
        XCTAssertFalse(fm.fileExists(atPath: locator.helperToolPath))
        // LaunchServices hygiene ran.
        XCTAssertEqual(recorder.unregistered, [locator.appBundleURL])
        XCTAssertEqual(recorder.gc, 1)
        XCTAssertEqual(recorder.dock, 1)
    }

    func testCancelledHelperLeavesTheRestDone() throws {
        let fm = FileManager.default
        let locator = try makeFixture()
        let recorder = Recorder()
        recorder.locator = locator
        recorder.helperOutcome = .cancelled
        let plan = UninstallPlanner.build(locator: locator, options: .default)
        let report = UninstallRunner.run(plan: plan, actions: recorder.actions(fm))

        XCTAssertTrue(report.helperLeftInPlace)
        XCTAssertEqual(report.helperOutcome, .cancelled)
        // Everything except the helper was still removed.
        XCTAssertEqual(Set(report.removedScopes), ["app", "runtime", "settings", "clusterState", "diagnostics"])
        XCTAssertFalse(report.errors.isEmpty)
        XCTAssertTrue(fm.fileExists(atPath: locator.identityURL.path))
        for dir in locator.modelDirs { XCTAssertTrue(fm.fileExists(atPath: dir)) }
        // The helper tool was left in place.
        XCTAssertTrue(fm.fileExists(atPath: locator.helperToolPath))
        XCTAssertTrue(report.didQuit)
    }

    func testKeepingTheAppSkipsTheQuit() throws {
        let fm = FileManager.default
        let locator = try makeFixture()
        var options = UninstallOptions.default
        options.removeApp = false
        let recorder = Recorder()
        let plan = UninstallPlanner.build(locator: locator, options: options)
        let report = UninstallRunner.run(plan: plan, actions: recorder.actions(fm))
        XCTAssertFalse(report.appRemoved)
        XCTAssertFalse(report.didQuit)
        XCTAssertEqual(recorder.quit, 0)
        XCTAssertTrue(fm.fileExists(atPath: locator.appBundleURL.path))
        // No app removed => no bundle unregister, but gc/dock still run.
        XCTAssertTrue(recorder.unregistered.isEmpty)
        XCTAssertEqual(recorder.gc, 1)
    }

    func testModelOptInRemovesOnlyTheChosenDirectory() throws {
        let fm = FileManager.default
        let locator = try makeFixture()
        var options = UninstallOptions.default
        options.modelDirsToRemove = [locator.modelDirs[0]]
        let recorder = Recorder()
        let plan = UninstallPlanner.build(locator: locator, options: options)
        let report = UninstallRunner.run(plan: plan, actions: recorder.actions(fm), quitAfter: false)
        XCTAssertFalse(fm.fileExists(atPath: locator.modelDirs[0]))
        XCTAssertTrue(fm.fileExists(atPath: locator.modelDirs[1]))
        XCTAssertEqual(report.keptModelDirs, [locator.modelDirs[1]])
        XCTAssertFalse(report.didQuit)
    }

    // MARK: Sheet view model

    @MainActor
    func testSheetVMRunsStopsServerAndQuitsLast() async throws {
        let fm = FileManager.default
        let locator = try makeFixture()
        let recorder = Recorder()
        recorder.locator = locator
        recorder.deleteHelperFiles = true
        var stopped = 0
        var quit = 0
        let vm = UninstallSheetVM(
            locator: locator,
            actions: recorder.actions(fm),
            stopServer: { stopped += 1 },
            quitApp: { quit += 1 }
        )
        XCTAssertTrue(vm.plan.keepsIdentity)
        XCTAssertEqual(vm.plan.keptModelDirs, locator.modelDirs)
        XCTAssertNil(vm.report)

        await vm.run()
        XCTAssertEqual(stopped, 1)
        XCTAssertNotNil(vm.report)
        XCTAssertEqual(quit, 0, "run must not quit; quitting is the explicit final step")
        XCTAssertTrue(fm.fileExists(atPath: locator.identityURL.path))

        vm.finish()
        XCTAssertEqual(quit, 1)
    }

    @MainActor
    func testSheetVMKeptLineNamesModelDirsVerbatim() throws {
        let locator = try makeFixture()
        let vm = UninstallSheetVM(
            locator: locator,
            actions: Recorder().actions(FileManager.default)
        )
        for dir in locator.modelDirs {
            XCTAssertTrue(vm.keptLine.contains(dir), "kept line must name \(dir) verbatim")
        }
        XCTAssertTrue(vm.keptLine.contains("identity"))
    }

    func testByteFormatting() {
        XCTAssertEqual(UninstallFormat.bytes(0), "0 B")
        XCTAssertEqual(UninstallFormat.bytes(100), "100 B")
        XCTAssertEqual(UninstallFormat.bytes(2048), "2.0 KB")
        XCTAssertEqual(UninstallFormat.bytes(5 * 1_048_576), "5.0 MB")
        XCTAssertEqual(UninstallFormat.bytes(3 * 1_073_741_824), "3.0 GB")
    }
}
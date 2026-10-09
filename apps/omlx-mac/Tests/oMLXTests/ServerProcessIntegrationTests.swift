// AutoRestartBudgetTests below are hermetic unit tests and always run.
// The ServerProcess integration smoke test exercises a real spawn +
// graceful shutdown end-to-end. Skipped by default so regular
// `xcodebuild test` runs stay hermetic and fast — opt in via
// `OMLX_INTEGRATION=1`.
//
// What this catches that the mock-based unit tests can't:
//   • PythonRuntime resolution against an actual interpreter
//   • Process spawn + termination-handler wiring
//   • SIGTERM honoured by the child within stopGraceSeconds
//
// We deliberately do NOT assert on the .running state transition or on
// the wire-level /health response — those depend on the health-check
// timing and the test-host's URLSession sandbox/policy, neither of which
// is stable across machines. The signal we keep is "did the parent
// successfully spawn, hold, and reap the child" — that is the part the
// mocks can't cover.
//
// To run locally (uses the dev_server.py stub so we don't need the
// bundled venvstacks framework). Xcode forwards TEST_RUNNER_* env vars
// to the xctest runner with the prefix stripped:
//
//   PYBIN="$(/usr/bin/which python3)"
//   REPO="$(git rev-parse --show-toplevel)"
//   TEST_RUNNER_OMLX_INTEGRATION=1 \
//   TEST_RUNNER_OMLX_PYTHON_OVERRIDE="$PYBIN" \
//   TEST_RUNNER_OMLX_DEV_SERVER_SCRIPT="$REPO/apps/omlx-mac/Scripts/dev_server.py" \
//     xcodebuild -project apps/omlx-mac/oMLX.xcodeproj \
//                -scheme oMLX \
//                -only-testing:oMLXTests/ServerProcessIntegrationTests \
//                test

import Darwin
import XCTest
@testable import oMLX

final class AutoRestartBudgetTests: XCTestCase {
    func testStableHealthResetsConsumedAttempts() {
        let start = Date(timeIntervalSince1970: 1_000)
        var budget = AutoRestartBudget(maxAttempts: 3, stableThreshold: 60)

        XCTAssertEqual(budget.consumeRestart(at: start), 1)
        budget.recordHealthy(at: start.addingTimeInterval(5))
        budget.recordHealthy(at: start.addingTimeInterval(64))
        XCTAssertEqual(budget.attempts, 1)

        budget.recordHealthy(at: start.addingTimeInterval(65))
        XCTAssertEqual(budget.attempts, 0)
        XCTAssertEqual(budget.consumeRestart(at: start.addingTimeInterval(66)), 1)
    }

    func testCrashAfterStableIntervalResetsWithoutAnotherHealthTick() {
        let start = Date(timeIntervalSince1970: 2_000)
        var budget = AutoRestartBudget(maxAttempts: 3, stableThreshold: 60)

        XCTAssertEqual(budget.consumeRestart(at: start), 1)
        budget.recordHealthy(at: start.addingTimeInterval(5))

        XCTAssertEqual(budget.consumeRestart(at: start.addingTimeInterval(65)), 1)
    }

    func testConsecutiveCrashesStillStopAtMaximum() {
        let start = Date(timeIntervalSince1970: 3_000)
        var budget = AutoRestartBudget(maxAttempts: 3, stableThreshold: 60)

        XCTAssertEqual(budget.consumeRestart(at: start), 1)
        XCTAssertEqual(budget.consumeRestart(at: start.addingTimeInterval(1)), 2)
        XCTAssertEqual(budget.consumeRestart(at: start.addingTimeInterval(2)), 3)
        XCTAssertNil(budget.consumeRestart(at: start.addingTimeInterval(3)))
    }
}

@MainActor
final class ServerProcessIntegrationTests: XCTestCase {

    override func setUpWithError() throws {
        guard ProcessInfo.processInfo.environment["OMLX_INTEGRATION"] == "1" else {
            throw XCTSkip("Set OMLX_INTEGRATION=1 to run integration smoke tests.")
        }
    }

    func testSpawnAndCleanShutdown() async throws {
        // The dev override pair must both be set — otherwise we'd need the
        // bundled venvstacks framework to satisfy `python -m omlx.cli`. Skip
        // with an actionable message rather than throw an opaque spawn
        // failure halfway through.
        let env = ProcessInfo.processInfo.environment
        guard let pythonOverride = env["OMLX_PYTHON_OVERRIDE"], !pythonOverride.isEmpty,
              FileManager.default.isExecutableFile(atPath: pythonOverride),
              let devScript = env["OMLX_DEV_SERVER_SCRIPT"], !devScript.isEmpty,
              FileManager.default.fileExists(atPath: devScript)
        else {
            throw XCTSkip(
                "Integration smoke test needs OMLX_PYTHON_OVERRIDE + " +
                "OMLX_DEV_SERVER_SCRIPT set. See file header for the command."
            )
        }

        let runtime = try PythonRuntime.resolve()
        XCTAssertFalse(runtime.isBundled,
                       "Smoke test should use the override interpreter, not the bundled one.")

        let port = Self.findFreePort()
        XCTAssertGreaterThan(port, 0, "Couldn't find a free port for the test.")

        let tempBase = FileManager.default.temporaryDirectory
            .appendingPathComponent("oMLX-integ-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempBase, withIntermediateDirectories: true)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: tempBase)
        }

        let proc = ServerProcess(
            runtime: runtime,
            bindAddress: "127.0.0.1",
            port: port,
            basePath: tempBase
        )

        // Spawn — the only assertion we make here is "no exception, no
        // immediate port conflict, no spawn-syscall failure."
        switch try proc.start() {
        case .started, .alreadyRunning:
            break
        case .portConflict(let conflict):
            XCTFail("Port \(port) reported in-use before spawn (isOMLX=\(conflict.isOMLX)).")
            return
        }
        XCTAssertNotNil(proc.pid, "Process should have a pid after start().")

        // Give the child enough time to actually bind so the port-released
        // check at the end is meaningful (otherwise we could fluke-pass by
        // checking before bind).
        try? await Task.sleep(for: .seconds(2))

        // Graceful stop — SIGTERM should bring the child down within
        // stopGraceSeconds. We pass a shorter timeout so a hung child
        // surfaces fast.
        await proc.stop(timeout: 5)
        if case .stopped = proc.state {} else {
            XCTFail("Server didn't transition to .stopped after stop(); state=\(proc.state)")
        }

        // The reaped child must release the port — otherwise SIGTERM didn't
        // actually take or the parent leaked the file descriptor.
        XCTAssertFalse(Self.isPortInUse(port: port),
                       "Port \(port) still bound after stop — orphaned child?")
    }

    func testEnvironmentPortOverridesSavedPort() async throws {
        let base = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let oldPort = Self.findFreePort()
        let nextPort = Self.findFreePort()
        try AppConfig.saveServerEndpoint(basePath: base.path, port: oldPort)
        let previous = ProcessInfo.processInfo.environment["OMLX_PORT"]
        setenv("OMLX_PORT", String(nextPort), 1)
        defer {
            if let previous { setenv("OMLX_PORT", previous, 1) }
            else { unsetenv("OMLX_PORT") }
        }
        let proc = ServerProcess(runtime: try PythonRuntime.resolve(), port: oldPort, basePath: base)
        addTeardownBlock { @MainActor in await proc.stop(timeout: 2) }
        try proc.start()
        try await waitForPort(nextPort)
        XCTAssertEqual(proc.port, nextPort)
        await proc.stop(timeout: 2)
        XCTAssertFalse(Self.isPortInUse(port: nextPort))
    }

    func testSavedPortSurvivesRestartAndUpdatesClient() async throws {
        let runtime = try PythonRuntime.resolve()
        let base = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let oldPort = Self.findFreePort()
        let proc = ServerProcess(runtime: runtime, port: oldPort, basePath: base)
        let config = AppConfig(bindAddress: "127.0.0.1", port: oldPort, apiKey: "test1234",
                               basePath: base.path, modelDir: base.appendingPathComponent("models").path,
                               hfEndpoint: "")
        try config.save()
        let services = AppServices(config: config, server: proc)
        addTeardownBlock { @MainActor in await proc.stop(timeout: 2) }
        try proc.start()
        try await waitForPort(oldPort)
        let newPort = Self.findFreePort()
        XCTAssertNotEqual(newPort, oldPort)
        var saved = config
        saved.port = newPort
        if ProcessInfo.processInfo.environment["OMLX_DEV_SERVER_SCRIPT"] == nil {
            _ = try await services.client.updateGlobalSettings(GlobalSettingsPatch(port: newPort))
        } else {
            try saved.save()
        }
        try await services.restartServer()
        try await waitForPort(newPort)
        XCTAssertFalse(Self.isPortInUse(port: oldPort))
        XCTAssertEqual(proc.port, newPort)
        XCTAssertEqual(services.client.port, newPort)
        XCTAssertEqual(services.config.port, newPort)
        XCTAssertEqual(try AppConfig.readSettingsForTests(basePath: base.path).port, newPort)
        let thirdPort = Self.findFreePort()
        saved.port = thirdPort
        if ProcessInfo.processInfo.environment["OMLX_DEV_SERVER_SCRIPT"] == nil {
            _ = try await services.client.updateGlobalSettings(GlobalSettingsPatch(port: thirdPort))
        } else {
            try saved.save()
        }
        // The admin restart route sends the same delayed SIGTERM.
        kill(try XCTUnwrap(proc.pid), SIGTERM)
        try await waitForPort(thirdPort)
        XCTAssertFalse(Self.isPortInUse(port: newPort))
        XCTAssertEqual(services.client.port, thirdPort)
        await proc.stop(timeout: 2)
        try proc.start()
        try await waitForPort(thirdPort)
        if ProcessInfo.processInfo.environment["OMLX_DEV_SERVER_SCRIPT"] == nil {
            let vm = ServerScreenVM()
            vm.applyConfig(services.config)
            await vm.load(client: services.client)
            XCTAssertNil(vm.lastError)
            let fourthPort = Self.findFreePort()
            vm.portText = String(fourthPort)
            vm.applyServerSettings(services: services)
            try await waitForPort(fourthPort)
            XCTAssertNil(vm.lastError)
            XCTAssertEqual(services.client.port, fourthPort)
            XCTAssertEqual(try AppConfig.readSettingsForTests(basePath: base.path).port, fourthPort)
            XCTAssertFalse(Self.isPortInUse(port: thirdPort))
            let fifthPort = Self.findFreePort()
            _ = try await services.client.updateGlobalSettings(GlobalSettingsPatch(port: fifthPort))
            try await services.applyStorageChanges(
                modelDirs: [base.appendingPathComponent("other-models").path], port: fifthPort
            )
            try await waitForPort(fifthPort)
            XCTAssertFalse(Self.isPortInUse(port: fourthPort))
            XCTAssertEqual(services.client.port, fifthPort)
            XCTAssertEqual(try AppConfig.readSettingsForTests(basePath: base.path).port, fifthPort)
            await proc.stop(timeout: 2)
            XCTAssertFalse(Self.isPortInUse(port: fifthPort))
        } else {
            await proc.stop(timeout: 2)
            XCTAssertFalse(Self.isPortInUse(port: thirdPort))
        }
    }

    func testOfflinePortApplyDoesNotNeedHTTPOrStartServer() async throws {
        try await checkOfflinePortApply(occupied: false)
    }

    func testPortConflictCanBeFixedWithOfflineApply() async throws {
        try await checkOfflinePortApply(occupied: true)
    }

    private func checkOfflinePortApply(occupied: Bool) async throws {
        let runtime = try PythonRuntime.resolve()
        let base = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let oldPort = Self.findFreePort()
        let proc = ServerProcess(runtime: runtime, port: oldPort, basePath: base)
        let config = AppConfig(bindAddress: "127.0.0.1", port: oldPort, apiKey: "test1234",
                               basePath: base.path, modelDir: base.appendingPathComponent("models").path,
                               hfEndpoint: "")
        try config.save()
        let services = AppServices(config: config, server: proc)
        var occupyingProcess: ServerProcess?
        if occupied {
            let other = ServerProcess(runtime: runtime, port: oldPort, basePath: base)
            occupyingProcess = other
            addTeardownBlock { @MainActor in await other.stop(timeout: 2) }
            try other.start()
            try await waitForPort(oldPort)
            guard case .portConflict = try proc.start() else {
                return XCTFail("Expected an occupied-port startup failure")
            }
        }
        let vm = ServerScreenVM()
        vm.applyConfig(config)
        await vm.load(client: services.client)
        for invalid in ["0", "-1", "65536", "invalid"] {
            vm.portText = invalid
            vm.applyServerSettings(services: services)
            XCTAssertNotNil(vm.lastError)
            XCTAssertEqual(try AppConfig.readSettingsForTests(basePath: base.path).port, oldPort)
        }
        let newPort = Self.findFreePort()
        vm.portText = String(newPort)
        vm.applyServerSettings(services: services)
        let deadline = Date().addingTimeInterval(3)
        while services.config.port != newPort && Date() < deadline {
            try await Task.sleep(for: .milliseconds(20))
        }
        XCTAssertNil(vm.lastError)
        XCTAssertEqual(proc.port, newPort)
        XCTAssertEqual(services.client.port, newPort)
        XCTAssertEqual(try AppConfig.readSettingsForTests(basePath: base.path).port, newPort)
        XCTAssertNil(proc.pid)
        XCTAssertFalse(vm.hasPendingServerChanges(services: services))
        await occupyingProcess?.stop(timeout: 2)
        addTeardownBlock { @MainActor in await proc.stop(timeout: 2) }
        try services.startServer()
        try await waitForPort(newPort)
        await proc.stop(timeout: 2)
        XCTAssertFalse(Self.isPortInUse(port: newPort))
    }

    private func waitForPort(_ port: Int) async throws {
        let deadline = Date().addingTimeInterval(15)
        while Date() < deadline {
            if Self.isPortInUse(port: port) { return }
            try await Task.sleep(for: .milliseconds(25))
        }
        XCTFail("Child did not bind port \(port)")
    }

    // MARK: - Helpers

    /// Bind to port 0, let the OS pick a free port, close the socket, and
    /// hand the port back. Small race window between close and the
    /// ServerProcess spawn — acceptable for a local smoke test.
    private static func findFreePort() -> Int {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { return 0 }
        defer { close(fd) }

        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = 0
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")

        let size = socklen_t(MemoryLayout<sockaddr_in>.size)
        let bound = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(fd, $0, size)
            }
        }
        guard bound == 0 else { return 0 }

        var picked = sockaddr_in()
        var pickedSize = socklen_t(MemoryLayout<sockaddr_in>.size)
        let got = withUnsafeMutablePointer(to: &picked) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { ptr in
                Darwin.getsockname(fd, ptr, &pickedSize)
            }
        }
        guard got == 0 else { return 0 }
        return Int(UInt16(bigEndian: picked.sin_port))
    }

    /// Tiny connect-probe to verify the port is released after stop. Returns
    /// true if a connection succeeds.
    private static func isPortInUse(port: Int) -> Bool {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { return false }
        defer { close(fd) }
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = UInt16(port).bigEndian
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")
        let size = socklen_t(MemoryLayout<sockaddr_in>.size)
        let result = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.connect(fd, $0, size)
            }
        }
        return result == 0
    }
}

@MainActor
final class StartupNoticeTests: XCTestCase {
    func testNoticeUpdatesEndpointAndIsConsumedOnce() throws {
        let runtime = PythonRuntime(
            executable: URL(fileURLWithPath: "/usr/bin/python3"),
            homebrewPaths: [], pythonPath: [], pythonHome: nil, isBundled: false
        )
        let server = ServerProcess(runtime: runtime, bindAddress: "192.168.1.10", port: 18150)
        let services = AppServices(server: server)
        let url = server.prepareStartupNotice()
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertNil(server.consumeStartupNotice())
        try "Server access is now limited to this Mac".write(to: url, atomically: true, encoding: .utf8)
        XCTAssertEqual(server.consumeStartupNotice(), "Server access is now limited to this Mac")
        XCTAssertEqual(server.bindAddress, "127.0.0.1")
        XCTAssertEqual(services.config.bindAddress, "127.0.0.1")
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
        XCTAssertNil(server.consumeStartupNotice())
    }
}

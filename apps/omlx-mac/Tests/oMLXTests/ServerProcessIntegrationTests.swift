// Integration smoke test for ServerProcess. Exercises a real spawn +
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
            host: "127.0.0.1",
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

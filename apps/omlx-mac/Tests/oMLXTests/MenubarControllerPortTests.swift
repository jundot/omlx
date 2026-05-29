// Regression coverage for the menubar-shows-stale-port bug.
//
// Before the fix, MenubarController captured `config: AppConfig` by value
// at init and rendered `config.port` in the running-status header / port
// alert / Chat URL. The user-facing flow:
//   1. ServerScreen's Apply commits a new port via
//      `AppServices.applyServerEndpoint(port:)`.
//   2. AppServices calls `server.reconfigure(port:)` and restarts the
//      ServerProcess on the new port.
//   3. The server transitions to `.running(newPid)`; the menubar's
//      stateDidChange observer fires `refreshMenuState()`.
//   4. `refreshMenuState()` rebuilds the header — and read the OLD port
//      from the stale `config` snapshot. The user saw `:8080` after
//      changing to `:8964`.
//
// Fix: `MenubarController.displayPort(server:fallback:)` sources from
// the live server (which `reconfigure(port:)` updates), falling back to
// the captured config snapshot only when there is no server (bootstrap
// failed). These tests exercise the helper directly — instantiating the
// full controller in a unit test would require a live `NSStatusBar`.

import Foundation
import XCTest
@testable import oMLX

@MainActor
final class MenubarControllerPortTests: XCTestCase {

    /// Test-only PythonRuntime. ServerProcess holds it but doesn't
    /// dereference until `start()` — these tests never start, they just
    /// read `.port` / `.host` after `reconfigure`.
    private func makeRuntime() -> PythonRuntime {
        PythonRuntime(
            executable: URL(fileURLWithPath: "/usr/bin/true"),
            homebrewPaths: [],
            pythonPath: [],
            pythonHome: nil,
            isBundled: false
        )
    }

    // MARK: - displayPort

    func testDisplayPortFallsBackToConfigWhenNoServer() {
        XCTAssertEqual(
            MenubarController.displayPort(server: nil, fallback: 8080),
            8080,
            "With no server (bootstrap failed), the displayed port must come from the AppConfig snapshot."
        )
    }

    func testDisplayPortPrefersLiveServer() {
        let server = ServerProcess(runtime: makeRuntime(), port: 8888)
        XCTAssertEqual(
            MenubarController.displayPort(server: server, fallback: 8080),
            8888,
            "When a server is present, its `port` is authoritative — `fallback` is only for the no-server case."
        )
    }

    func testDisplayPortFollowsReconfigure() throws {
        // The original bug: menubar's `config.port` snapshot never sees
        // this change, so the running-header text keeps showing 8080.
        let server = ServerProcess(runtime: makeRuntime(), port: 8080)
        try server.reconfigure(port: 8964)
        XCTAssertEqual(
            MenubarController.displayPort(server: server, fallback: 8080),
            8964,
            "After Server screen's Apply commits a new port (which calls server.reconfigure(port:)), the menubar must source from the live server."
        )
    }

    // MARK: - displayHost

    func testDisplayHostFallsBackToConfigWhenNoServer() {
        XCTAssertEqual(
            MenubarController.displayHost(server: nil, fallback: "127.0.0.1"),
            "127.0.0.1"
        )
    }

    func testDisplayHostPrefersLiveServer() {
        let server = ServerProcess(runtime: makeRuntime(), host: "0.0.0.0", port: 8080)
        XCTAssertEqual(
            MenubarController.displayHost(server: server, fallback: "127.0.0.1"),
            "0.0.0.0"
        )
    }

    func testDisplayHostFollowsReconfigure() throws {
        let server = ServerProcess(runtime: makeRuntime(), host: "127.0.0.1", port: 8080)
        try server.reconfigure(host: "0.0.0.0")
        XCTAssertEqual(
            MenubarController.displayHost(server: server, fallback: "127.0.0.1"),
            "0.0.0.0",
            "Listen Address changes propagate to the server via saveHost → applyServerEndpoint → server.reconfigure(host:); the menubar must reflect that."
        )
    }
}

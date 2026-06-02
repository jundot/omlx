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

    // MARK: - webAdminURL
    //
    // The "Open Web Dashboard" menubar item routes through the server's
    // /admin/auto-login endpoint so the dashboard opens without the manual
    // login form. The action method itself needs a live NSStatusBar, so we
    // test the pure URL builder it delegates to.

    func testWebAdminURLUsesAutoLoginWithRedirect() throws {
        let url = try XCTUnwrap(
            MenubarController.webAdminURL(host: "127.0.0.1", port: 8000, apiKey: "secret")
        )
        let comps = try XCTUnwrap(URLComponents(url: url, resolvingAgainstBaseURL: false))
        XCTAssertEqual(comps.scheme, "http")
        XCTAssertEqual(comps.host, "127.0.0.1")
        XCTAssertEqual(comps.port, 8000)
        XCTAssertEqual(comps.path, "/admin/auto-login")
        let items = comps.queryItems ?? []
        XCTAssertEqual(items.first { $0.name == "redirect" }?.value, "/admin/dashboard")
        XCTAssertEqual(items.first { $0.name == "key" }?.value, "secret")
    }

    func testWebAdminURLPercentEncodesKey() throws {
        // A key with URL-reserved characters must survive intact — raw
        // string interpolation would corrupt it; URLComponents encodes it.
        let url = try XCTUnwrap(
            MenubarController.webAdminURL(host: "127.0.0.1", port: 8000, apiKey: "a+b/c&d")
        )
        // The decoded query item value round-trips to the original key.
        let comps = try XCTUnwrap(URLComponents(url: url, resolvingAgainstBaseURL: false))
        XCTAssertEqual(comps.queryItems?.first { $0.name == "key" }?.value, "a+b/c&d")
        // And the raw URL string carries the encoded form, not the literal.
        XCTAssertTrue(url.absoluteString.contains("key=a%2Bb/c%26d"),
                      "key should be percent-encoded in the URL string, got \(url.absoluteString)")
    }

    func testWebAdminURLOmitsKeyWhenMissing() throws {
        for key in [nil, ""] as [String?] {
            let url = try XCTUnwrap(
                MenubarController.webAdminURL(host: "127.0.0.1", port: 8000, apiKey: key)
            )
            let comps = try XCTUnwrap(URLComponents(url: url, resolvingAgainstBaseURL: false))
            XCTAssertNil(comps.queryItems?.first { $0.name == "key" },
                         "empty/nil key must not emit a key= param (server redirects to login instead)")
            XCTAssertEqual(comps.queryItems?.first { $0.name == "redirect" }?.value,
                           "/admin/dashboard")
        }
    }
}

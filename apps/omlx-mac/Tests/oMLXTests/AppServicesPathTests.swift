// Exercises the path-rewriting helpers AppServices uses during a basePath
// migration. These are the same helpers that silently failed in production
// (PR — settings.json rewrite during basePath move), so this is the regression
// fence we're putting up to keep that from regressing.
//
// The functions under test are pure-Foundation; no SwiftUI / AppKit / server
// process needed, so each test stays fast and hermetic. relocateOrphanPaths
// touches the filesystem — we feed it a settings.json in a per-test temp dir.

import XCTest
@testable import oMLX

final class AppServicesPathTests: XCTestCase {

    // MARK: relocate(path:oldBase:newBase:)

    func testRelocateInsidePrefix() {
        XCTAssertEqual(
            AppServices.relocate(path: "/old/sub/dir",
                                 oldBase: "/old",
                                 newBase: "/new"),
            "/new/sub/dir"
        )
    }

    func testRelocateExactMatch() {
        XCTAssertEqual(
            AppServices.relocate(path: "/old", oldBase: "/old", newBase: "/new"),
            "/new"
        )
    }

    func testRelocateOutsideTreeIsUnchanged() {
        XCTAssertEqual(
            AppServices.relocate(path: "/Volumes/SSD/models",
                                 oldBase: "/Users/Fido/.omlx",
                                 newBase: "/Users/Fido/.omlx-other"),
            "/Volumes/SSD/models"
        )
    }

    func testRelocateNearMissPrefixIsUnchanged() {
        // `/old-x` must NOT be rewritten when oldBase is `/old` — guards
        // against a naive `hasPrefix` without a trailing-slash boundary.
        XCTAssertEqual(
            AppServices.relocate(path: "/old-x/sub",
                                 oldBase: "/old",
                                 newBase: "/new"),
            "/old-x/sub"
        )
    }

    func testRelocateEmptyStringIsUnchanged() {
        XCTAssertEqual(
            AppServices.relocate(path: "", oldBase: "/old", newBase: "/new"),
            ""
        )
    }

    func testRelocateTildeIsExpanded() {
        // The function normalizes its input via standardizingPath +
        // expandingTildeInPath. A `~`-prefixed path under the home dir
        // should still match if oldBase is the expanded home equivalent.
        let home = NSHomeDirectory()
        XCTAssertEqual(
            AppServices.relocate(path: "~/.omlx/models",
                                 oldBase: "\(home)/.omlx",
                                 newBase: "\(home)/.omlx-other"),
            "\(home)/.omlx-other/models"
        )
    }

    // MARK: relocateOrphanPaths(in:oldBase:newBase:)

    private func makeTempSettingsFile(contents: [String: Any]) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("oMLXTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("settings.json")
        let data = try JSONSerialization.data(withJSONObject: contents, options: [.prettyPrinted])
        try data.write(to: url)
        return url
    }

    private func readJSON(_ url: URL) throws -> [String: Any] {
        let data = try Data(contentsOf: url)
        return try JSONSerialization.jsonObject(with: data) as! [String: Any]
    }

    func testRelocateOrphanPathsHappyPath() throws {
        let url = try makeTempSettingsFile(contents: [
            "model": [
                "model_dirs": ["/Users/Fido/.omlx-other/models"],
                "model_dir": "/Users/Fido/.omlx-other/models",
                "max_model_memory": "auto"
            ],
            "cache": [
                "ssd_cache_dir": "/Users/Fido/.omlx-other/cache",
                "enabled": true
            ],
            "logging": [
                "log_dir": "/Users/Fido/.omlx-other/logs",
                "retention_days": 7
            ],
            "server": ["host": "127.0.0.1", "port": 8080]
        ])

        try AppServices.relocateOrphanPaths(
            in: url,
            oldBase: "/Users/Fido/.omlx-other",
            newBase: "/Users/Fido/.omlx"
        )

        let after = try readJSON(url)
        let model = after["model"] as! [String: Any]
        XCTAssertEqual(model["model_dirs"] as! [String], ["/Users/Fido/.omlx/models"])
        XCTAssertEqual(model["model_dir"] as! String, "/Users/Fido/.omlx/models")
        XCTAssertEqual(model["max_model_memory"] as! String, "auto",
                       "unrelated keys must survive the rewrite")

        let cache = after["cache"] as! [String: Any]
        XCTAssertEqual(cache["ssd_cache_dir"] as! String, "/Users/Fido/.omlx/cache")
        XCTAssertEqual(cache["enabled"] as! Bool, true)

        let logging = after["logging"] as! [String: Any]
        XCTAssertEqual(logging["log_dir"] as! String, "/Users/Fido/.omlx/logs")

        let server = after["server"] as! [String: Any]
        XCTAssertEqual(server["port"] as! Int, 8080,
                       "sibling sections we don't touch must round-trip identically")
    }

    func testRelocateOrphanPathsTolerantsNullLogDir() throws {
        // Regression: log_dir: null from Python landed as NSNull in the dict
        // and earlier code paths could crash or refuse to serialize when
        // serializing back. The rewrite must leave it intact.
        let url = try makeTempSettingsFile(contents: [
            "model": ["model_dirs": ["/old/models"]],
            "logging": ["log_dir": NSNull(), "retention_days": 7]
        ])

        try AppServices.relocateOrphanPaths(in: url, oldBase: "/old", newBase: "/new")

        let after = try readJSON(url)
        let logging = after["logging"] as! [String: Any]
        XCTAssertTrue(logging["log_dir"] is NSNull)
        XCTAssertEqual(logging["retention_days"] as! Int, 7)
    }

    func testRelocateOrphanPathsLeavesOutsidePathsAlone() throws {
        // model_dir lives on a separate volume — the user explicitly pointed
        // it outside the basePath tree. The migration must NOT yank it back.
        let url = try makeTempSettingsFile(contents: [
            "model": [
                "model_dirs": ["/Volumes/SSD/models"],
                "model_dir": "/Volumes/SSD/models"
            ],
            "cache": ["ssd_cache_dir": "/old/cache"]
        ])

        try AppServices.relocateOrphanPaths(in: url, oldBase: "/old", newBase: "/new")

        let after = try readJSON(url)
        let model = after["model"] as! [String: Any]
        XCTAssertEqual(model["model_dirs"] as! [String], ["/Volumes/SSD/models"])
        XCTAssertEqual(model["model_dir"] as! String, "/Volumes/SSD/models")

        let cache = after["cache"] as! [String: Any]
        XCTAssertEqual(cache["ssd_cache_dir"] as! String, "/new/cache",
                       "matching paths still get rewritten")
    }

    func testRelocateOrphanPathsFileMissingIsNoOp() throws {
        // The file may legitimately not exist on first-run installs; the
        // function must not crash or throw.
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("does-not-exist-\(UUID().uuidString).json")
        XCTAssertNoThrow(
            try AppServices.relocateOrphanPaths(in: missing,
                                                oldBase: "/old", newBase: "/new")
        )
    }

    func testRelocateOrphanPathsSkipsEmptyStringPaths() throws {
        // Empty-string path fields are valid placeholders that mean "use
        // default" — the rewrite must not silently turn "" into newBase.
        let url = try makeTempSettingsFile(contents: [
            "model": ["model_dir": ""],
            "cache": ["ssd_cache_dir": ""]
        ])

        try AppServices.relocateOrphanPaths(in: url, oldBase: "/old", newBase: "/new")

        let after = try readJSON(url)
        XCTAssertEqual((after["model"] as! [String: Any])["model_dir"] as! String, "")
        XCTAssertEqual((after["cache"] as! [String: Any])["ssd_cache_dir"] as! String, "")
    }
}

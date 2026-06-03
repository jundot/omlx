// AppConfig invariants we rely on at runtime:
//   • modelDir is always a literal path (never empty).
//   • save() preserves unknown keys (e.g. cache, integrations, ui).
//   • defaultModelDir is `<basePath>/models`, no shell expansion games.
//
// Tests write to a per-test temp directory so they don't trample the user's
// real ~/.omlx or Library/Application Support state.

import XCTest
@testable import oMLX

final class AppConfigTests: XCTestCase {

    private var tempBase: String!

    override func setUpWithError() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("AppConfigTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        tempBase = dir.path
    }

    override func tearDownWithError() throws {
        if let tempBase {
            try? FileManager.default.removeItem(atPath: tempBase)
        }
    }

    // MARK: defaultModelDir

    func testDefaultModelDirIsBasePathSlashModels() {
        XCTAssertEqual(
            AppConfig.defaultModelDir(forBasePath: "/some/base"),
            "/some/base/models"
        )
    }

    func testDefaultModelDirHandlesTrailingSlash() {
        XCTAssertEqual(
            AppConfig.defaultModelDir(forBasePath: "/some/base/"),
            "/some/base/models"
        )
    }

    // MARK: save / round-trip

    func testSaveProducesExpectedTopLevelKeys() throws {
        let cfg = AppConfig(
            bindAddress: "127.0.0.1",
            port: 9000,
            apiKey: "secret",
            basePath: tempBase,
            modelDir: "\(tempBase!)/models",
            hfEndpoint: "https://hf-mirror.example"
        )

        try cfg.save()

        let url = AppConfig.settingsURL(basePath: tempBase)
        let data = try Data(contentsOf: url)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]

        XCTAssertEqual((json["server"] as! [String: Any])["host"] as! String, "127.0.0.1")
        XCTAssertNil((json["server"] as! [String: Any])["bind_address"])
        XCTAssertEqual((json["server"] as! [String: Any])["port"] as! Int, 9000)
        XCTAssertEqual((json["server"] as! [String: Any])["auto_start_on_launch"] as! Bool, true)
        XCTAssertEqual((json["auth"] as! [String: Any])["api_key"] as! String, "secret")
        let model = json["model"] as! [String: Any]
        XCTAssertEqual(model["model_dirs"] as! [String], ["\(tempBase!)/models"])
        XCTAssertEqual(model["model_dir"] as! String, "\(tempBase!)/models")
        XCTAssertEqual((json["huggingface"] as! [String: Any])["endpoint"] as! String,
                       "https://hf-mirror.example")
        XCTAssertEqual(json["version"] as! String, "1.0")
    }

    func testSavePreservesUnknownKeys() throws {
        // Pre-populate settings.json with keys AppConfig doesn't own. These
        // come from the running Python server (claude_code, integrations, ui,
        // etc.) and must round-trip untouched through Swift saves.
        let url = AppConfig.settingsURL(basePath: tempBase)
        let original: [String: Any] = [
            "version": "1.0",
            "claude_code": ["enabled": true, "model": "claude-opus-4-5"],
            "integrations": ["github": ["token": "abc"]],
            "ui": ["theme": "dark"],
            "server": ["host": "0.0.0.0", "bind_address": "0.0.0.0", "port": 1234],
            "model": [
                "model_dirs": ["/some/where/else"],
                "model_dir": "/will-be-overwritten",
                "max_model_memory": "auto"             // also unknown
            ],
            "cache": ["enabled": true, "ssd_cache_dir": "/x/cache"]
        ]
        try JSONSerialization.data(withJSONObject: original, options: [.prettyPrinted])
            .write(to: url)

        let cfg = AppConfig(
            bindAddress: "127.0.0.1",
            port: 8080,
            autoStartOnLaunch: false,
            apiKey: nil,
            basePath: tempBase,
            modelDir: "/new/models",
            hfEndpoint: ""
        )
        try cfg.save()

        let after = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]

        // Foreign top-level keys survive.
        XCTAssertEqual((after["claude_code"] as! [String: Any])["model"] as! String, "claude-opus-4-5")
        XCTAssertEqual((after["integrations"] as! [String: Any])["github"] as! [String: String], ["token": "abc"])
        XCTAssertEqual((after["ui"] as! [String: Any])["theme"] as! String, "dark")

        // Unknown sub-keys under owned sections survive too — only the fields
        // AppConfig owns get rewritten.
        let server = after["server"] as! [String: Any]
        XCTAssertEqual(server["host"] as! String, "127.0.0.1")
        XCTAssertEqual(server["auto_start_on_launch"] as! Bool, false)
        XCTAssertNil(server["bind_address"])

        let model = after["model"] as! [String: Any]
        XCTAssertEqual(model["model_dirs"] as! [String], ["/new/models"],
                       "model_dirs is AppConfig-owned and must stay in sync with model_dir")
        XCTAssertEqual(model["max_model_memory"] as! String, "auto")
        XCTAssertEqual(model["model_dir"] as! String, "/new/models",
                       "model_dir is AppConfig-owned and gets the new value")

        let cache = after["cache"] as! [String: Any]
        XCTAssertEqual(cache["ssd_cache_dir"] as! String, "/x/cache",
                       "cache.ssd_cache_dir is not in AppConfig's slice")
    }

    func testWildcardBindAddressUsesHostKeyButConnectsViaLoopback() throws {
        let cfg = AppConfig(
            bindAddress: "0.0.0.0",
            port: 9000,
            apiKey: nil,
            basePath: tempBase,
            modelDir: "\(tempBase!)/models",
            hfEndpoint: ""
        )

        XCTAssertEqual(cfg.host, "127.0.0.1")

        try cfg.save()

        let url = AppConfig.settingsURL(basePath: tempBase)
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
        let server = json["server"] as! [String: Any]
        XCTAssertEqual(server["host"] as! String, "0.0.0.0")
        XCTAssertNil(server["bind_address"])
    }

    func testLoadAcceptsBindAddressFallback() throws {
        let url = AppConfig.settingsURL(basePath: tempBase)
        let original: [String: Any] = [
            "server": ["bind_address": "0.0.0.0", "port": 9000],
            "model": ["model_dir": "\(tempBase!)/models"]
        ]
        try JSONSerialization.data(withJSONObject: original, options: [.prettyPrinted])
            .write(to: url)

        let slice = try AppConfig.readSettingsForTests(basePath: tempBase)

        XCTAssertEqual(slice.bindAddress, "0.0.0.0")
        XCTAssertEqual(slice.port, 9000)
    }

    func testLoadReadsAutoStartOnLaunch() throws {
        let url = AppConfig.settingsURL(basePath: tempBase)
        let original: [String: Any] = [
            "server": [
                "host": "127.0.0.1",
                "port": 9000,
                "auto_start_on_launch": false
            ],
            "model": ["model_dir": "\(tempBase!)/models"]
        ]
        try JSONSerialization.data(withJSONObject: original, options: [.prettyPrinted])
            .write(to: url)

        let slice = try AppConfig.readSettingsForTests(basePath: tempBase)

        XCTAssertEqual(slice.autoStartOnLaunch, false)
    }

    func testLoadReadsModelDirsAndPrimaryModelDir() throws {
        let url = AppConfig.settingsURL(basePath: tempBase)
        let original: [String: Any] = [
            "server": ["host": "127.0.0.1", "port": 9000],
            "model": [
                "model_dirs": ["/models/a", "/models/b"],
                "model_dir": "/models/a"
            ]
        ]
        try JSONSerialization.data(withJSONObject: original, options: [.prettyPrinted])
            .write(to: url)

        let slice = try AppConfig.readSettingsForTests(basePath: tempBase)

        XCTAssertEqual(slice.modelDirs ?? [], ["/models/a", "/models/b"])
        XCTAssertEqual(slice.modelDir, "/models/a")
    }

    // MARK: modelDir invariant

    func testDefaultConfigHasNonEmptyModelDir() {
        // Even on a fresh install with no settings.json, AppConfig.default
        // must hand back a usable modelDir. Otherwise the UI shows a blank
        // field and the server falls through to its own default — diverging.
        XCTAssertFalse(AppConfig.default.modelDir.isEmpty)
        XCTAssertFalse(AppConfig.default.effectiveModelDirs.isEmpty)
        XCTAssertTrue(AppConfig.default.modelDir.hasSuffix("/models"))
    }

    // MARK: bootstrap file

    func testBootstrapRoundTrips() throws {
        // The bootstrap file is the fallback for Finder/Dock launches that
        // don't inherit the user's shell rc. Write a path, read it, clear it.
        let writtenPath = "\(tempBase!)/custom-base"
        try AppConfig.writeBootstrapBasePath(writtenPath)

        XCTAssertEqual(AppConfig.readBootstrapBasePath(), writtenPath)

        try AppConfig.writeBootstrapBasePath(nil)
        XCTAssertNil(AppConfig.readBootstrapBasePath(),
                     "passing nil should remove the file")
    }
}

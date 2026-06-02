import XCTest
@testable import oMLX

final class ShellEnvWriterTests: XCTestCase {
    private var tempHome: URL!
    private var oldPath: String?

    override func setUpWithError() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ShellEnvWriterTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        tempHome = dir
        oldPath = getenv("PATH").map { String(cString: $0) }
        ShellEnvWriter.homeOverrideForTests = dir
        ShellEnvWriter.shellOverrideForTests = "/bin/zsh"
        setenv("PATH", "/usr/bin", 1)
    }

    override func tearDownWithError() throws {
        ShellEnvWriter.homeOverrideForTests = nil
        ShellEnvWriter.shellOverrideForTests = nil
        if let oldPath {
            setenv("PATH", oldPath, 1)
        }
        if let tempHome {
            try? FileManager.default.removeItem(at: tempHome)
        }
    }

    func testEnsureCLIShimWritesExecutableWrapperAndPathBlock() throws {
        let appURL = try makeFakeAppURL()

        try ShellEnvWriter.ensureCLIShim(appBundleURL: appURL)

        let shim = tempHome
            .appendingPathComponent(".omlx", isDirectory: true)
            .appendingPathComponent("bin", isDirectory: true)
            .appendingPathComponent("omlx")
        XCTAssertTrue(FileManager.default.isExecutableFile(atPath: shim.path))
        let shimText = try String(contentsOf: shim, encoding: .utf8)
        XCTAssertTrue(shimText.contains("Contents/MacOS/omlx-cli"))
        XCTAssertTrue(shimText.contains("exec "))

        let zshrc = tempHome.appendingPathComponent(".zshrc")
        let rcText = try String(contentsOf: zshrc, encoding: .utf8)
        XCTAssertTrue(rcText.contains("# oMLX: CLI shim path begin"))
        XCTAssertTrue(rcText.contains("$HOME/.omlx/bin"))
    }

    func testEnsureCLIShimIsIdempotent() throws {
        let appURL = try makeFakeAppURL()

        try ShellEnvWriter.ensureCLIShim(appBundleURL: appURL)
        try ShellEnvWriter.ensureCLIShim(appBundleURL: appURL)

        let zshrc = tempHome.appendingPathComponent(".zshrc")
        let rcText = try String(contentsOf: zshrc, encoding: .utf8)
        let count = rcText.components(separatedBy: "# oMLX: CLI shim path begin").count - 1
        XCTAssertEqual(count, 1)
    }

    private func makeFakeAppURL() throws -> URL {
        let appURL = tempHome
            .appendingPathComponent("Apps", isDirectory: true)
            .appendingPathComponent("oMLX.app", isDirectory: true)
        let cli = appURL
            .appendingPathComponent("Contents", isDirectory: true)
            .appendingPathComponent("MacOS", isDirectory: true)
            .appendingPathComponent("omlx-cli")
        try FileManager.default.createDirectory(
            at: cli.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try "#!/bin/sh\n".write(to: cli, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755],
            ofItemAtPath: cli.path
        )
        return appURL
    }
}

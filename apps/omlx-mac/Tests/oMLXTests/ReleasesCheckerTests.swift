import XCTest
@testable import oMLX

final class ReleasesCheckerTests: XCTestCase {

    func testCompareVersionsOrdersPrereleaseSuffixes() {
        XCTAssertEqual(
            ReleasesChecker.compareVersions("0.4.0rc2", "0.4.0rc1"),
            .orderedDescending
        )
        XCTAssertEqual(
            ReleasesChecker.compareVersions("0.4.0", "0.4.0rc2"),
            .orderedDescending
        )
        XCTAssertEqual(
            ReleasesChecker.compareVersions("0.4.0rc1", "0.4.0.dev1"),
            .orderedDescending
        )
    }

    func testStableChannelExcludesPrereleases() {
        let selected = ReleasesChecker.selectLatest(
            [
                release("v0.4.0rc2"),
                release("v0.3.12"),
            ],
            channel: .stable
        )

        XCTAssertEqual(selected?.tagName, "v0.3.12")
    }

    func testReleaseCandidateChannelIncludesRCButExcludesDev() {
        let selected = ReleasesChecker.selectLatest(
            [
                release("v0.4.1.dev1"),
                release("v0.4.0rc2"),
                release("v0.4.0rc1"),
            ],
            channel: .releaseCandidate
        )

        XCTAssertEqual(selected?.tagName, "v0.4.0rc2")
    }

    func testDevChannelIncludesDev() {
        let selected = ReleasesChecker.selectLatest(
            [
                release("v0.4.1.dev1"),
                release("v0.4.0rc2"),
                release("v0.4.0"),
            ],
            channel: .dev
        )

        XCTAssertEqual(selected?.tagName, "v0.4.1.dev1")
    }

    private func release(
        _ tag: String,
        prerelease: Bool = false,
        draft: Bool = false
    ) -> GitHubRelease {
        GitHubRelease(
            tagName: tag,
            name: tag,
            body: nil,
            htmlURL: URL(string: "https://github.com/jundot/omlx/releases/tag/\(tag)")!,
            prerelease: prerelease,
            draft: draft,
            assets: []
        )
    }
}

@MainActor
final class UpdateControllerPrefsTests: XCTestCase {

    func testLegacyAutoDownloadPrefMigratesToAutoNotify() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("omlx-update-prefs-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let url = dir.appendingPathComponent("update-prefs.json")
        try Data(
            #"{"channel":"stable","autoCheck":true,"autoDownload":true}"#.utf8
        ).write(to: url)

        let controller = UpdateController(storeURL: url, currentVersion: "0.0.0")
        XCTAssertTrue(controller.autoNotify)

        controller.autoNotify = false

        let saved = try JSONSerialization.jsonObject(
            with: Data(contentsOf: url)
        ) as? [String: Any]
        XCTAssertEqual(saved?["autoNotify"] as? Bool, false)
        XCTAssertNil(saved?["autoDownload"])
    }
}

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

    func testFindMatchingDMGSupportsMacOSRangeAssets() {
        let sequoia = "oMLX-0.4.4-macos15-sequoia.dmg"
        let tahoeAndNext = "oMLX-0.4.4-macos26-27.dmg"
        let assets = [
            asset(sequoia),
            asset(tahoeAndNext),
        ]

        XCTAssertEqual(
            ReleasesChecker.findMatchingDMG(
                assets: assets,
                macOSMajor: 15
            )?.name,
            sequoia
        )
        XCTAssertEqual(
            ReleasesChecker.findMatchingDMG(
                assets: assets,
                macOSMajor: 26
            )?.name,
            tahoeAndNext
        )
        XCTAssertEqual(
            ReleasesChecker.findMatchingDMG(
                assets: assets,
                macOSMajor: 27
            )?.name,
            tahoeAndNext
        )
        XCTAssertNil(
            ReleasesChecker.findMatchingDMG(
                assets: assets,
                macOSMajor: 28
            )
        )
    }

    func testFindMatchingDMGPrefersExactAssetOverRangeAsset() {
        let range = "oMLX-0.4.4-macos26-27.dmg"
        let exact = "oMLX-0.4.4-macos27-beta.dmg"
        let assets = [
            asset(range),
            asset(exact),
        ]

        XCTAssertEqual(
            ReleasesChecker.findMatchingDMG(
                assets: assets,
                macOSMajor: 27
            )?.name,
            exact
        )
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

    private func asset(_ name: String) -> GitHubRelease.Asset {
        GitHubRelease.Asset(
            name: name,
            browserDownloadURL: URL(string: "https://example.com/\(name)")!,
            size: 123
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

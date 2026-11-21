// Model folder detection for the Welcome wizard. Pinned here because the
// 2026-11-16 acceptance run showed the cost of getting it wrong: three Macs
// with 902 GB of models finished Welcome pointing at an empty
// `<base>/models`, so the first measurement reported `models: []`.

import XCTest
@testable import oMLX

final class ModelDirectoryDetectorTests: XCTestCase {
    private var root: URL!
    private var home: URL!
    private let fm = FileManager.default

    override func setUpWithError() throws {
        root = fm.temporaryDirectory.appendingPathComponent("modeldetect-\(UUID().uuidString)")
        home = root.appendingPathComponent("home", isDirectory: true)
        try fm.createDirectory(at: home, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let root { try? fm.removeItem(at: root) }
    }

    private func makeModel(_ path: String, marker: String = "config.json") throws {
        let dir = home.appendingPathComponent(path, isDirectory: true)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        if marker == "weight" {
            try Data().write(to: dir.appendingPathComponent("model-00001-of-00002.safetensors"))
        } else if marker == "none" {
            try Data().write(to: dir.appendingPathComponent("README.txt"))
        } else {
            try Data("{}".utf8).write(to: dir.appendingPathComponent(marker))
        }
    }

    func testDetectsBothConfigurationAndWeightModels() throws {
        try makeModel("Models/GLM5")
        try makeModel("Models/Qwen4", marker: "weight")

        let found = ModelDirectoryDetector.detect(home: home, fileManager: fm)

        XCTAssertEqual(found.count, 1)
        XCTAssertEqual(found.first?.modelCount, 2)
        XCTAssertEqual(found.first?.path, home.appendingPathComponent("Models").standardizedFileURL.path)
        XCTAssertEqual(found.first?.isHuggingFaceCache, false)
    }

    func testIgnoresFoldersThatAreNotModels() throws {
        try makeModel("Models/notes", marker: "none")
        try fm.createDirectory(at: home.appendingPathComponent("models"),
                               withIntermediateDirectories: true)

        XCTAssertTrue(ModelDirectoryDetector.detect(home: home, fileManager: fm).isEmpty)
    }

    func testCountsHuggingFaceCacheEntries() throws {
        let snapshots = home.appendingPathComponent(
            ".cache/huggingface/hub/models--mlx-community--GLM-5.3/snapshots/abc",
            isDirectory: true
        )
        try fm.createDirectory(at: snapshots, withIntermediateDirectories: true)

        let found = ModelDirectoryDetector.detect(home: home, fileManager: fm)

        XCTAssertEqual(found.count, 1)
        XCTAssertTrue(found.first?.isHuggingFaceCache ?? false)
        XCTAssertEqual(found.first?.modelCount, 1)
    }

    func testLargestRootComesFirst() throws {
        try makeModel("Models/only-one")
        for name in ["a", "b", "c"] { try makeModel("mlx-models/\(name)") }

        let found = ModelDirectoryDetector.detect(home: home, fileManager: fm)

        XCTAssertEqual(found.count, 2)
        XCTAssertEqual(found.first?.modelCount, 3)
        XCTAssertEqual(found.first?.path, home.appendingPathComponent("mlx-models").standardizedFileURL.path)
    }

    func testRealModelFolderBeatsALargerHuggingFaceCache() throws {
        // Measured 2026-11-16 on .11: the cache held more `models--*` entries
        // than ~/Models had checkpoints, and the wizard adopted the cache.
        try makeModel("Models/GLM5")
        for name in ["a", "b", "c", "d", "e"] {
            let snapshots = home.appendingPathComponent(
                ".cache/huggingface/hub/models--org--\(name)/snapshots/abc", isDirectory: true)
            try fm.createDirectory(at: snapshots, withIntermediateDirectories: true)
        }

        let found = ModelDirectoryDetector.detect(home: home, fileManager: fm)

        XCTAssertEqual(found.first?.path,
                       home.appendingPathComponent("Models").standardizedFileURL.path)
        XCTAssertEqual(found.first?.isHuggingFaceCache, false)
        XCTAssertEqual(found.count, 2)
    }

    func testOneFolderIsReportedOnceAcrossCaseInsensitiveRoots() throws {
        // `~/Models` and `~/models` are the same folder on a default APFS
        // volume; the detector must not count it twice.
        try makeModel("Models/GLM5")
        try makeModel("models/Qwen4", marker: "weight")

        let found = ModelDirectoryDetector.detect(home: home, fileManager: fm)

        XCTAssertEqual(found.count, 1)
        XCTAssertEqual(found.first?.modelCount, 2)
    }

    func testAdoptsOnlyAnEmptyOrUntouchedDefaultField() throws {
        let base = root.appendingPathComponent("base", isDirectory: true).path
        let defaultDir = AppConfig.defaultModelDir(forBasePath: base)

        // empty field: adopt
        XCTAssertTrue(ModelDirectoryDetector.shouldAdopt(
            current: "", basePath: base, fileManager: fm))

        // untouched default with nothing in it: adopt
        XCTAssertTrue(ModelDirectoryDetector.shouldAdopt(
            current: defaultDir, basePath: base, fileManager: fm))

        // a folder the user chose: never adopt
        let chosen = home.appendingPathComponent("Models", isDirectory: true)
        XCTAssertFalse(ModelDirectoryDetector.shouldAdopt(
            current: chosen.path, basePath: base, fileManager: fm))

        // the default that already holds a model: keep it
        let populated = URL(fileURLWithPath: defaultDir, isDirectory: true)
            .appendingPathComponent("Already", isDirectory: true)
        try fm.createDirectory(at: populated, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: populated.appendingPathComponent("config.json"))
        XCTAssertFalse(ModelDirectoryDetector.shouldAdopt(
            current: defaultDir, basePath: base, fileManager: fm))
    }

    func testDisplayPathAbbreviatesTheHomeDirectory() {
        let detectorPath = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Models", isDirectory: true).path
        let entry = DetectedModelDirectory(path: detectorPath, modelCount: 3,
                                           isHuggingFaceCache: false)
        XCTAssertEqual(entry.displayPath, "~/Models")
    }
}
// First-run runtime installer (slim distribution). These tests pin the
// offline-testable surface: the exact `uv` command plan, the pinned
// interpreter, the installation check, and the fail-closed order that keeps a
// missing bundled installer from creating a half-built venv. The download
// itself needs a network and is exercised by hand (SLIM-RUNTIME.md).

import XCTest
@testable import oMLX

final class RuntimeInstallerTests: XCTestCase {

    private var root: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("omlx-runtime-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let root {
            try? FileManager.default.removeItem(at: root)
        }
        try super.tearDownWithError()
    }

    /// A tree that looks like `Contents/Resources/runtime-project`.
    @discardableResult
    private func makeProject(
        pythonVersion: String? = "3.12.16",
        withLock: Bool = true
    ) throws -> URL {
        let project = root.appendingPathComponent("runtime-project", isDirectory: true)
        try FileManager.default.createDirectory(at: project, withIntermediateDirectories: true)
        if withLock {
            try "lock".write(
                to: project.appendingPathComponent("uv.lock"),
                atomically: true,
                encoding: .utf8
            )
        }
        if let pythonVersion {
            try "\(pythonVersion)\n".write(
                to: project.appendingPathComponent(".python-version"),
                atomically: true,
                encoding: .utf8
            )
        }
        return project
    }

    @discardableResult
    private func makeUV() throws -> URL {
        let uv = root.appendingPathComponent("uv")
        try "#!/bin/sh\nexit 0\n".write(to: uv, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755], ofItemAtPath: uv.path
        )
        return uv
    }

    private func installer(project: URL?, uv: URL?) -> RuntimeInstaller {
        RuntimeInstaller(
            runtimeRoot: root.appendingPathComponent("oMLX/runtime", isDirectory: true),
            projectDirectory: project,
            uvCandidates: uv.map { [$0] } ?? []
        )
    }

    // MARK: - command plan

    func testPlanPinsTheLockedUvSurface() {
        let venv = URL(fileURLWithPath: "/tmp/v/oMLX/runtime/venv")
        let steps = RuntimeInstaller.plan(pythonVersion: "3.12.16", venv: venv)

        XCTAssertEqual(steps.count, 3)
        XCTAssertEqual(steps[0].args, ["python", "install", "3.12.16"])
        XCTAssertEqual(
            steps[1].args,
            ["venv", "--clear", "--python", "3.12.16", venv.path]
        )
        XCTAssertEqual(
            steps[2].args,
            [
                "sync", "--frozen", "--no-dev", "--no-install-project",
                "--extra", "cluster", "--python",
                venv.appendingPathComponent("bin/python3").path,
            ]
        )
        // The venv is the sync target; the app's own source is never installed.
        XCTAssertEqual(steps[2].environment["UV_PROJECT_ENVIRONMENT"], venv.path)
        XCTAssertNil(steps[0].environment["UV_PROJECT_ENVIRONMENT"])
        XCTAssertNil(steps[1].environment["UV_PROJECT_ENVIRONMENT"])
        XCTAssertTrue(steps.flatMap(\.args).contains("--no-install-project"))
    }

    // MARK: - pinned interpreter

    func testPinnedPythonComesFromTheBundledVersionFile() throws {
        let project = try makeProject(pythonVersion: "3.12.16\n")
        let sut = installer(project: project, uv: nil)
        XCTAssertEqual(sut.pinnedPythonVersion(), "3.12.16")
    }

    func testPinnedPythonFallsBackForAMissingOrEmptyFile() throws {
        let missing = try makeProject(pythonVersion: nil)
        XCTAssertEqual(
            installer(project: missing, uv: nil).pinnedPythonVersion(),
            RuntimeInstaller.defaultPythonVersion
        )

        let empty = try makeProject(pythonVersion: "   ")
        XCTAssertEqual(
            installer(project: empty, uv: nil).pinnedPythonVersion(),
            RuntimeInstaller.defaultPythonVersion
        )

        XCTAssertEqual(
            RuntimeInstaller(
                runtimeRoot: root, projectDirectory: nil, uvCandidates: []
            ).pinnedPythonVersion(),
            RuntimeInstaller.defaultPythonVersion
        )
    }

    // MARK: - installed check

    func testIsInstalledRequiresAnExecutableVenvInterpreter() throws {
        let sut = installer(project: nil, uv: nil)
        XCTAssertFalse(sut.isInstalled)

        let bin = sut.venvRoot.appendingPathComponent("bin", isDirectory: true)
        try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
        // A plain file is not enough: the Start Server path spawns it.
        try "not a binary".write(
            to: sut.venvPython, atomically: true, encoding: .utf8
        )
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o644], ofItemAtPath: sut.venvPython.path
        )
        XCTAssertFalse(sut.isInstalled)

        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755], ofItemAtPath: sut.venvPython.path
        )
        XCTAssertTrue(sut.isInstalled)

        // The app's call site keeps reading the production location.
        XCTAssertTrue(
            RuntimeInstaller.defaultRuntimeRoot.path.hasSuffix("/oMLX/runtime")
        )
    }

    // MARK: - clean first run fails closed

    func testMissingBundledUVFailsBeforeCreatingAnything() async {
        let sut = installer(project: nil, uv: nil)
        do {
            try await sut.install { _ in }
            XCTFail("install must fail without a bundled uv")
        } catch let error as RuntimeInstallError {
            guard case .missingUV = error else {
                return XCTFail("unexpected error \(error)")
            }
        } catch {
            XCTFail("unexpected error \(error)")
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: sut.runtimeRoot.path))
    }

    func testMissingProjectManifestFailsBeforeCreatingAnything() async throws {
        let uv = try makeUV()
        let project = try makeProject(withLock: false)
        let sut = installer(project: project, uv: uv)
        do {
            try await sut.install { _ in }
            XCTFail("install must fail without uv.lock")
        } catch let error as RuntimeInstallError {
            guard case .missingProject = error else {
                return XCTFail("unexpected error \(error)")
            }
        } catch {
            XCTFail("unexpected error \(error)")
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: sut.runtimeRoot.path))
    }

    func testMissingBundleFailClosedProjectLocationIsNil() async throws {
        let uv = try makeUV()
        let sut = RuntimeInstaller(
            runtimeRoot: root.appendingPathComponent("runtime", isDirectory: true),
            projectDirectory: nil,
            uvCandidates: [uv]
        )
        do {
            try await sut.install { _ in }
            XCTFail("install must fail when the bundle has no runtime-project")
        } catch let error as RuntimeInstallError {
            guard case .missingProject = error else {
                return XCTFail("unexpected error \(error)")
            }
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: sut.runtimeRoot.path))
    }
}
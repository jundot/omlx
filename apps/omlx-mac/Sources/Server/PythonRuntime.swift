// PR 2 (PR 11 follow-up) — locate the Python interpreter the parent will
// spawn.
//
// Resolution order (first match wins):
//   1. OMLX_PYTHON_OVERRIDE env var — dev escape hatch.
//   2. ~/Library/Application Support/oMLX/runtime/venv/bin/python3 — the
//      slim runtime provisioned by apps/omlx-mac/Scripts/provision_runtime.sh.
//      It keeps Python and MLX out of the app bundle and DMG.
//   3. Bundle.main/Contents/Resources/Python/cpython-3.11/bin/python3 — the
//      legacy embedded venvstacks layout.
//   4. Legacy bundle layouts under Contents/Python or Contents/Frameworks.
//
// In the bundled case the spawn environment also sets:
//   PYTHONHOME = Contents/Resources/Python/cpython-3.11
//     so the relocated interpreter finds its stdlib without grepping the
//     host system's /usr/lib.
//   PYTHONPATH = Contents/Resources : framework-mlx-base/site-packages
//     : __venvstacks__/site-customize
//     so `python -m omlx.cli` resolves both the omlx package (shipped as a
//     pure source tree in Resources/omlx/, matching today's Python build)
//     and the framework layer's wheels (mlx, transformers, fastapi, …).
//   PYTHONDONTWRITEBYTECODE = 1
//     so the read-only app bundle doesn't try to scribble .pyc files into
//     itself at first import.
//   OMLX_SUPERVISED = menubar
//     so the admin restart endpoint knows the parent app can respawn the
//     server after its delayed self-SIGTERM.

import Foundation

struct PythonRuntime {
    let executable: URL
    /// Extra PATH entries to prepend, matching today's Python menubar
    /// (server_manager.py:328-340 — Homebrew paths needed for ffmpeg, etc.).
    let homebrewPaths: [String]
    /// PYTHONPATH entries to prepend. Empty when the override path is used.
    let pythonPath: [URL]
    /// PYTHONHOME — the cpython layer root. nil when using a system Python.
    let pythonHome: URL?
    /// True when the bundled runtime was found; false if we fell back.
    let isBundled: Bool

    enum ResolutionError: Error, CustomStringConvertible {
        case notFound(triedPaths: [String])

        var description: String {
            switch self {
            case .notFound(let paths):
                return "Python runtime not found. Tried: \(paths.joined(separator: ", "))"
            }
        }
    }

    static func resolve() throws -> PythonRuntime {
        let env = ProcessInfo.processInfo.environment
        var tried: [String] = []

        if let override = env["OMLX_PYTHON_OVERRIDE"], !override.isEmpty {
            let url = URL(fileURLWithPath: override)
            tried.append(override)
            if FileManager.default.isExecutableFile(atPath: url.path) {
                return PythonRuntime(
                    executable: url,
                    homebrewPaths: defaultHomebrewPaths,
                    pythonPath: [],
                    pythonHome: nil,
                    isBundled: false
                )
            }
        }

        let bundleRoot = Bundle.main.bundleURL
        let resources = bundleRoot.appendingPathComponent("Contents/Resources")

        // Slim distribution: Python and the MLX layer live in the user's
        // provisioned runtime, not in the app bundle. The app still ships the
        // omlx/omlx_web source in Resources, so PYTHONPATH points there.
        if let support = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first {
            let venvPython = support
                .appendingPathComponent("oMLX/runtime/venv/bin/python3")
            tried.append(venvPython.path)
            if FileManager.default.isExecutableFile(atPath: venvPython.path) {
                return PythonRuntime(
                    executable: venvPython,
                    homebrewPaths: defaultHomebrewPaths,
                    pythonPath: [resources],
                    pythonHome: nil,
                    isBundled: false
                )
            }
        }

        let pythonRoots = [
            resources.appendingPathComponent("Python"),
            bundleRoot.appendingPathComponent("Contents/Python"),
            bundleRoot.appendingPathComponent("Contents/Frameworks"),
        ]

        for pythonRoot in pythonRoots {
            let cpython = pythonRoot.appendingPathComponent("cpython-3.11")
            let bundled = cpython.appendingPathComponent("bin/python3")
            tried.append(bundled.path)
            guard FileManager.default.isExecutableFile(atPath: bundled.path) else {
                continue
            }

            let mlxFramework = pythonRoot
                .appendingPathComponent("framework-mlx-base/lib/python3.11/site-packages")
            return PythonRuntime(
                executable: bundled,
                homebrewPaths: defaultHomebrewPaths,
                pythonPath: [resources, mlxFramework],
                pythonHome: cpython,
                isBundled: true
            )
        }

        throw ResolutionError.notFound(triedPaths: tried)
    }

    /// Build the spawn environment: parent env + supervisor marker +
    /// Homebrew PATH + PYTHONPATH + PYTHONHOME. `PYTHONDONTWRITEBYTECODE=1`
    /// is set in bundled mode so the read-only app bundle doesn't try to
    /// scribble `__pycache__/` into itself.
    func makeEnvironment() -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        env["OMLX_SUPERVISED"] = "menubar"
        // macOS malloc otherwise keeps large empty arenas resident after
        // repeated model load/unload cycles. This must be set before Python
        // starts; setting it inside omlx.cli is too late for malloc init.
        env["MallocSpaceEfficient"] = env["MallocSpaceEfficient"] ?? "1"
        // Never write __pycache__ into the app bundle. This matters for the
        // external runtime too: the Python child would otherwise modify
        // Contents/Resources and invalidate the code signature.
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        var path = env["PATH"] ?? ""
        for prefix in homebrewPaths.reversed() where !path.contains(prefix) {
            path = path.isEmpty ? prefix : "\(prefix):\(path)"
        }
        env["PATH"] = path

        if !pythonPath.isEmpty {
            let joined = pythonPath.map(\.path).joined(separator: ":")
            if let existing = env["PYTHONPATH"], !existing.isEmpty {
                env["PYTHONPATH"] = "\(joined):\(existing)"
            } else {
                env["PYTHONPATH"] = joined
            }
        }

        if let home = pythonHome {
            env["PYTHONHOME"] = home.path
        }

        return env
    }

    private static let defaultHomebrewPaths = [
        "/opt/homebrew/bin",
        "/opt/homebrew/sbin",
        "/usr/local/bin",
    ]
}

// MARK: - First-run runtime installer

enum RuntimeInstallError: LocalizedError {
    case missingUV
    case missingProject
    case failed(String)

    var errorDescription: String? {
        switch self {
        case .missingUV:
            return "The bundled uv installer is missing."
        case .missingProject:
            return "The bundled runtime manifest is missing."
        case .failed(let message):
            return message.isEmpty ? "Runtime installation failed." : message
        }
    }
}

/// One external command the first-run installer runs. Kept as data so the
/// exact `uv` surface is testable without downloading anything.
struct RuntimeInstallStep: Equatable {
    let args: [String]
    let environment: [String: String]
}

/// Slim distribution helper: install a pinned CPython and the locked
/// dependencies into Application Support, outside the app bundle. The UI
/// calls this once on first run; `PythonRuntime.resolve()` then finds it.
///
/// The locations are injected so the clean-first-run path can be exercised
/// against a temporary directory: production uses `RuntimeInstaller.shared`,
/// which resolves the user's Application Support folder and the app bundle.
struct RuntimeInstaller {
    /// The runtime root, e.g. `~/Library/Application Support/oMLX/runtime`.
    let runtimeRoot: URL
    /// `Contents/Resources/runtime-project` in the app bundle, or a directory
    /// holding `uv.lock`/`.python-version`.
    let projectDirectory: URL?
    /// `Contents/Resources/runtime/uv` plus the host fallbacks, in order.
    let uvCandidates: [URL]

    static let defaultPythonVersion = "3.12.16"

    static var defaultRuntimeRoot: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)
            .first!
            .appendingPathComponent("oMLX/runtime", isDirectory: true)
    }

    static var shared: RuntimeInstaller {
        RuntimeInstaller(
            runtimeRoot: defaultRuntimeRoot,
            projectDirectory: Bundle.main.resourceURL?
                .appendingPathComponent("runtime-project"),
            uvCandidates: defaultUVCandidates
        )
    }

    static var defaultUVCandidates: [URL] {
        var candidates: [URL] = []
        if let resources = Bundle.main.resourceURL {
            candidates.append(resources.appendingPathComponent("runtime/uv"))
        }
        candidates.append(URL(fileURLWithPath: "/opt/homebrew/bin/uv"))
        candidates.append(URL(fileURLWithPath: "/usr/local/bin/uv"))
        candidates.append(
            FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".local/bin/uv")
        )
        return candidates
    }

    var venvRoot: URL {
        runtimeRoot.appendingPathComponent("venv", isDirectory: true)
    }

    var venvPython: URL {
        venvRoot.appendingPathComponent("bin/python3")
    }

    var isInstalled: Bool {
        FileManager.default.isExecutableFile(atPath: venvPython.path)
    }

    /// Kept for the Welcome call sites; same meaning as `shared.isInstalled`.
    static var isInstalled: Bool { shared.isInstalled }

    /// The pinned interpreter: the bundled `.python-version`, else 3.12.16.
    func pinnedPythonVersion() -> String {
        guard let project = projectDirectory else {
            return Self.defaultPythonVersion
        }
        let versionFile = project.appendingPathComponent(".python-version")
        let version = (try? String(contentsOf: versionFile, encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return (version?.isEmpty == false ? version! : Self.defaultPythonVersion)
    }

    /// The three `uv` invocations, in order: install the interpreter, create
    /// the venv, sync the locked dependencies. `--frozen --no-dev
    /// --no-install-project` keeps the app's own source out of the venv (it is
    /// shipped in `Contents/Resources` and put on `PYTHONPATH`).
    static func plan(pythonVersion: String, venv: URL) -> [RuntimeInstallStep] {
        [
            RuntimeInstallStep(
                args: ["python", "install", pythonVersion],
                environment: [:]
            ),
            RuntimeInstallStep(
                args: ["venv", "--clear", "--python", pythonVersion, venv.path],
                environment: [:]
            ),
            RuntimeInstallStep(
                args: [
                    "sync", "--frozen", "--no-dev", "--no-install-project",
                    "--extra", "cluster", "--python",
                    venv.appendingPathComponent("bin/python3").path,
                ],
                environment: ["UV_PROJECT_ENVIRONMENT": venv.path]
            ),
        ]
    }

    static func install(
        progress: @escaping @MainActor @Sendable (String) -> Void
    ) async throws {
        try await shared.install(progress: progress)
    }

    func install(
        progress: @escaping @MainActor @Sendable (String) -> Void
    ) async throws {
        // Both inputs are checked before the first download: a missing bundled
        // installer must fail with one clear error, not a half-finished venv.
        let uv = try locateUV()
        let project = try locateProject()
        let pythonVersion = pinnedPythonVersion()

        try FileManager.default.createDirectory(
            at: runtimeRoot,
            withIntermediateDirectories: true
        )
        await progress("Preparing Python \(pythonVersion)…\n")

        let steps = Self.plan(pythonVersion: pythonVersion, venv: venvRoot)
        for (index, step) in steps.enumerated() {
            switch index {
            case 1:
                await progress("Creating the runtime environment…\n")
            case 2:
                await progress("Installing locked dependencies (this downloads MLX)…\n")
            default:
                break
            }
            try await Self.run(
                uv,
                step.args,
                cwd: project,
                environment: step.environment,
                progress: progress
            )
        }

        guard isInstalled else {
            throw RuntimeInstallError.failed("The runtime venv was not created.")
        }
        await progress("Runtime ready.\n")
    }

    func locateUV() throws -> URL {
        for candidate in uvCandidates
        where FileManager.default.isExecutableFile(atPath: candidate.path) {
            return candidate
        }
        throw RuntimeInstallError.missingUV
    }

    func locateProject() throws -> URL {
        guard let project = projectDirectory,
              FileManager.default.fileExists(
                  atPath: project.appendingPathComponent("uv.lock").path
              )
        else {
            throw RuntimeInstallError.missingProject
        }
        return project
    }

    private static func run(
        _ executable: URL,
        _ arguments: [String],
        cwd: URL,
        environment: [String: String],
        progress: @escaping @MainActor @Sendable (String) -> Void
    ) async throws {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let process = Process()
                process.executableURL = executable
                process.arguments = arguments
                process.currentDirectoryURL = cwd
                var env = ProcessInfo.processInfo.environment
                for (key, value) in environment { env[key] = value }
                process.environment = env

                let pipe = Pipe()
                process.standardOutput = pipe
                process.standardError = pipe
                pipe.fileHandleForReading.readabilityHandler = { handle in
                    let data = handle.availableData
                    guard !data.isEmpty,
                          let text = String(data: data, encoding: .utf8),
                          !text.isEmpty
                    else { return }
                    Task { @MainActor in progress(text) }
                }

                do {
                    try process.run()
                    process.waitUntilExit()
                    pipe.fileHandleForReading.readabilityHandler = nil
                    if process.terminationStatus == 0 {
                        continuation.resume()
                    } else {
                        continuation.resume(
                            throwing: RuntimeInstallError.failed(
                                "\(executable.lastPathComponent) \(arguments.first ?? "") exited with \(process.terminationStatus)."
                            )
                        )
                    }
                } catch {
                    pipe.fileHandleForReading.readabilityHandler = nil
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}

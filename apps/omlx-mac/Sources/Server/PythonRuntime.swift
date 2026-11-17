// PR 2 (PR 11 follow-up) — locate the Python interpreter the parent will
// spawn.
//
// Resolution order (first match wins):
//   1. OMLX_PYTHON_OVERRIDE env var — dev escape hatch.
//   2. Bundle.main/Contents/Resources/Python/cpython-3.11/bin/python3 — production.
//      Layout matches the venvstacks export tree, which
//      apps/omlx-mac/Scripts/build.sh copies verbatim into the Swift .app.
//   3. Legacy bundle layouts under Contents/Python or Contents/Frameworks.
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
            env["PYTHONDONTWRITEBYTECODE"] = "1"
        }

        return env
    }

    private static let defaultHomebrewPaths = [
        "/opt/homebrew/bin",
        "/opt/homebrew/sbin",
        "/usr/local/bin",
    ]
}

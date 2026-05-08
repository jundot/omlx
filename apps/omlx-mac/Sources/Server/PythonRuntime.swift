// PR 2 — locate the Python interpreter the parent will spawn.
//
// Resolution order (first match wins):
//   1. OMLX_PYTHON_OVERRIDE env var — dev escape hatch.
//   2. Bundle.main/Contents/Frameworks/cpython-3.11/bin/python3 — production.
//      Layout matches the venvstacks export tree, which packaging/build.py
//      copies verbatim into the Swift .app at the --swift-next step. The
//      bundled interpreter resolves the framework layer via venvstacks's
//      metadata in Contents/Frameworks/__venvstacks__/, so we don't set
//      PYTHONPATH ourselves in the bundled case.
//
// PR 5 grows this with port/PATH plumbing; PR 12 collapses the two paths once
// the venvstacks runtime is the only supported deployment.

import Foundation

struct PythonRuntime {
    let executable: URL
    /// Extra PATH entries to prepend, matching today's Python menubar
    /// (server_manager.py:328-340 — Homebrew paths needed for ffmpeg, etc.).
    let homebrewPaths: [String]
    /// PYTHONPATH entries to prepend. Empty when the override path is used.
    let pythonPath: [URL]
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
                    isBundled: false
                )
            }
        }

        let bundleRoot = Bundle.main.bundleURL
        let bundled = bundleRoot
            .appendingPathComponent("Contents/Frameworks/cpython-3.11/bin/python3")
        tried.append(bundled.path)
        if FileManager.default.isExecutableFile(atPath: bundled.path) {
            return PythonRuntime(
                executable: bundled,
                homebrewPaths: defaultHomebrewPaths,
                pythonPath: [],   // venvstacks metadata handles site-packages
                isBundled: true
            )
        }

        throw ResolutionError.notFound(triedPaths: tried)
    }

    /// Build the spawn environment: parent env + Homebrew PATH + PYTHONPATH.
    func makeEnvironment() -> [String: String] {
        var env = ProcessInfo.processInfo.environment

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

        return env
    }

    private static let defaultHomebrewPaths = [
        "/opt/homebrew/bin",
        "/opt/homebrew/sbin",
        "/usr/local/bin",
    ]
}

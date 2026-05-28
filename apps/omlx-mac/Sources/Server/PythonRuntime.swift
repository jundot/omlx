// Locate the Python interpreter the parent will spawn `omlx serve` with.
//
// Thin-shell model: we do NOT embed a Python runtime in the .app. Instead
// we shell out to the project's own virtualenv on the host, so editing a
// .py file in the worktree + restarting the server takes effect immediately
// (the bundle-hack workflow). The interpreter lives at
// `~/Code/omlx/.venv/bin/python` on the deploy machines (see CLAUDE.md —
// servers run at ~/Code/omlx).
//
// Resolution order (first match wins):
//   1. OMLX_PYTHON_OVERRIDE env var — dev/test escape hatch.
//   2. ~/Code/omlx/.venv/bin/python — the project venv. The venv already
//      has omlx importable, so no PYTHONHOME/PYTHONPATH wiring is needed:
//      running the venv's own python binary makes its site-packages the
//      default search path.

import Foundation

struct PythonRuntime {
    let executable: URL
    /// Extra PATH entries to prepend so the server finds Homebrew tools
    /// (ffmpeg, etc.) the way the legacy Python menubar did.
    let homebrewPaths: [String]
    /// Always false in the thin shell — we never embed an interpreter.
    /// Kept so callers that distinguish bundled vs. host Python still build.
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
                    isBundled: false
                )
            }
        }

        let venvPython = defaultVenvPython
        tried.append(venvPython.path)
        if FileManager.default.isExecutableFile(atPath: venvPython.path) {
            return PythonRuntime(
                executable: venvPython,
                homebrewPaths: defaultHomebrewPaths,
                isBundled: false
            )
        }

        throw ResolutionError.notFound(triedPaths: tried)
    }

    /// Build the spawn environment: parent env + Homebrew PATH. The venv's
    /// own python resolves omlx and its deps without PYTHONHOME/PYTHONPATH,
    /// so we touch only PATH here.
    func makeEnvironment() -> [String: String] {
        var env = ProcessInfo.processInfo.environment

        var path = env["PATH"] ?? ""
        for prefix in homebrewPaths.reversed() where !path.contains(prefix) {
            path = path.isEmpty ? prefix : "\(prefix):\(path)"
        }
        env["PATH"] = path

        return env
    }

    /// `~/Code/omlx/.venv/bin/python` — the project virtualenv interpreter.
    private static var defaultVenvPython: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Code/omlx/.venv/bin/python")
    }

    private static let defaultHomebrewPaths = [
        "/opt/homebrew/bin",
        "/opt/homebrew/sbin",
        "/usr/local/bin",
    ]
}

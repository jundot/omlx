// ShellEnvWriter owns the app-managed CLI shim.
//
// The app writes only under `~/.omlx/bin`. It never writes to shared bin dirs
// such as `/opt/homebrew/bin`, which package managers own. Shell rc edits and
// removal of a link left by older app versions happen only after a prompt.

import Foundation

enum ShellEnvWriter {
    static let variableName = "OMLX_BASE_PATH"
    nonisolated(unsafe) static var homeOverrideForTests: URL?
    nonisolated(unsafe) static var shellOverrideForTests: String?
    nonisolated(unsafe) static var publicBinDirsOverrideForTests: [URL]?
    nonisolated(unsafe) static var cliPathPrefsURLOverrideForTests: URL?

    enum CLISetupResult: Equatable {
        case publicCommandReady(path: String)
        case needsShellPathPrompt(reason: String)
        case legacyPublicLink(path: String, shellPathInstalled: Bool)
    }

    private enum WriterError: LocalizedError {
        case cliWrapperNotExecutable(String)

        var errorDescription: String? {
            switch self {
            case .cliWrapperNotExecutable(let path):
                return "App-bundle CLI wrapper is not executable: \(path)"
            }
        }
    }

    private static let cliShimBeginMarker = "# oMLX: CLI shim path begin"
    private static let cliShimEndMarker = "# oMLX: CLI shim path end"
    // Read-only. A Homebrew cask `binary` stanza links the bundle CLI here,
    // and app versions before #4341 linked the shim here.
    private static let publicBinCandidates = [
        "/opt/homebrew/bin",
        "/usr/local/bin",
    ]

    /// Install/update `~/.omlx/bin/omlx` so app-only installs still expose
    /// the same terminal command as pip/Homebrew installs.
    @discardableResult
    static func ensureCLIShim(appBundleURL: URL = Bundle.main.bundleURL) throws -> CLISetupResult {
        let shimURL = cliShimURL()
        let shimDir = shimURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(
            at: shimDir,
            withIntermediateDirectories: true
        )

        let bundleCLI = appBundleURL
            .appendingPathComponent("Contents", isDirectory: true)
            .appendingPathComponent("MacOS", isDirectory: true)
            .appendingPathComponent("omlx-cli")
        guard FileManager.default.isExecutableFile(atPath: bundleCLI.path) else {
            throw WriterError.cliWrapperNotExecutable(bundleCLI.path)
        }
        try writeLauncherShim(at: shimURL, forwardingTo: bundleCLI)

        // Another Mac's coordinator discovers this node by running
        // `~/.omlx/bin/omlx-cluster-python -c 'import omlx'` over SSH. The CLI
        // wrapper above cannot answer that — it hardcodes `-m omlx.cli` — so a
        // peer with the app installed failed every discovery candidate and was
        // reported as "worker runtime is not installed" (#2680). Publish the
        // bundle's plain interpreter under the name discovery looks for.
        let clusterPython = appBundleURL
            .appendingPathComponent("Contents", isDirectory: true)
            .appendingPathComponent("MacOS", isDirectory: true)
            .appendingPathComponent("omlx-cluster-python")
        if FileManager.default.isExecutableFile(atPath: clusterPython.path) {
            // Best effort: an older bundle predates this wrapper, and failing
            // to publish it must not stop the CLI shim from installing.
            try? writeLauncherShim(
                at: shimDir.appendingPathComponent("omlx-cluster-python"),
                forwardingTo: clusterPython
            )
        }

        for dir in publicBinDirs() {
            let link = dir.appendingPathComponent("omlx")
            if isLegacyPublicLink(link) {
                return .legacyPublicLink(
                    path: link.path,
                    shellPathInstalled: shellPathExportAlreadyInstalled()
                )
            }
        }

        let appCLIs = [shimURL, bundleCLI]
        if let path = firstCLIPathInCurrentPath(), isAppCLI(path, appCLIs: appCLIs) {
            return .publicCommandReady(path: path.path)
        }

        // A GUI launch only sees the launchd PATH, so check the shared bin
        // dirs directly for a cask link.
        var conflicts: [String] = []
        for dir in publicBinDirs() {
            let link = dir.appendingPathComponent("omlx")
            guard FileManager.default.isExecutableFile(atPath: link.path) else { continue }
            if isAppCLI(link, appCLIs: appCLIs) {
                return .publicCommandReady(path: link.path)
            }
            conflicts.append("\(link.path) is a different omlx install.")
        }

        // The shim marker in a shell file is the only signal that "Update
        // Shell File" already ran.
        if shellPathExportAlreadyInstalled() {
            return .publicCommandReady(path: shimURL.path)
        }
        return .needsShellPathPrompt(reason: conflicts.joined(separator: "\n"))
    }

    /// Remove `path` only if it is still a link that an older app version
    /// created. Any other file is left in place.
    static func removeLegacyPublicLink(atPath path: String) throws {
        let link = URL(fileURLWithPath: path)
        guard isLegacyPublicLink(link) else { return }
        try FileManager.default.removeItem(at: link)
    }

    static func shouldSuppressCLIPathPrompt() -> Bool {
        readCLIPathPrefs().suppressShellPathPrompt
    }

    static func suppressCLIPathPromptForever() {
        var prefs = readCLIPathPrefs()
        prefs.suppressShellPathPrompt = true
        writeCLIPathPrefs(prefs)
    }

    static func ensureShellPathExport() throws {
        try ensureCLIPathExport()
    }

    /// Write an executable `/bin/sh` shim that restores `OMLX_BASE_PATH` from
    /// the app's bootstrap file and then hands argv to a bundle executable.
    private static func writeLauncherShim(at shimURL: URL, forwardingTo target: URL) throws {
        let script = """
        #!/bin/sh
        BOOTSTRAP="$HOME/Library/Application Support/oMLX/base-path"
        if [ -r "$BOOTSTRAP" ]; then
            IFS= read -r \(variableName) < "$BOOTSTRAP" || \(variableName)=""
            if [ -n "$\(variableName)" ]; then
                export \(variableName)
            fi
        fi
        exec \(shellQuote(target.path)) "$@"
        """
        try script.write(to: shimURL, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755],
            ofItemAtPath: shimURL.path
        )
    }

    // MARK: - File targets

    private static func home() -> URL {
        homeOverrideForTests ?? FileManager.default.homeDirectoryForCurrentUser
    }

    private static func cliShimURL() -> URL {
        home()
            .appendingPathComponent(".omlx", isDirectory: true)
            .appendingPathComponent("bin", isDirectory: true)
            .appendingPathComponent("omlx")
    }

    private static func candidateFiles() -> [URL] {
        let names = [
            ".zshrc", ".zprofile", ".zshenv",
            ".bashrc", ".bash_profile", ".profile",
        ]
        return names.map { home().appendingPathComponent($0) }
    }

    /// Prefer the rc file matching the user's `$SHELL`. zsh users get
    /// `.zshrc`, bash users get `.bashrc`, anyone else falls back below.
    private static func primaryFile() -> URL? {
        let shell = shellOverrideForTests ?? ProcessInfo.processInfo.environment["SHELL"] ?? ""
        if shell.contains("zsh") {
            return home().appendingPathComponent(".zshrc")
        }
        if shell.contains("bash") {
            // On macOS, GUI-launched terminals run login shells, so
            // .bash_profile is what gets sourced. Prefer it when it
            // already exists.
            let profile = home().appendingPathComponent(".bash_profile")
            if FileManager.default.fileExists(atPath: profile.path) {
                return profile
            }
            return home().appendingPathComponent(".bashrc")
        }
        return nil
    }

    // MARK: - File mutation

    private struct CLIPathPrefs: Codable {
        var suppressShellPathPrompt: Bool = false
    }

    private static func publicBinDirs() -> [URL] {
        publicBinDirsOverrideForTests
            ?? publicBinCandidates.map { URL(fileURLWithPath: $0, isDirectory: true) }
    }

    /// Older app versions linked `<public bin>/omlx` to the absolute shim
    /// path. Compare the raw link text so no other link matches.
    private static func isLegacyPublicLink(_ link: URL) -> Bool {
        guard let destination = try? FileManager.default
            .destinationOfSymbolicLink(atPath: link.path)
        else {
            return false
        }
        return destination == cliShimURL().path
    }

    private static func firstCLIPathInCurrentPath() -> URL? {
        let current = getenv("PATH").map { String(cString: $0) } ?? ""
        for part in current.split(separator: ":").map(String.init) {
            let candidate = URL(fileURLWithPath: part, isDirectory: true)
                .appendingPathComponent("omlx")
            guard FileManager.default.isExecutableFile(atPath: candidate.path) else {
                continue
            }
            return candidate
        }
        return nil
    }

    private static func isAppCLI(_ path: URL, appCLIs: [URL]) -> Bool {
        let resolved = path.resolvingSymlinksInPath().standardizedFileURL.path
        return appCLIs.contains {
            $0.resolvingSymlinksInPath().standardizedFileURL.path == resolved
        }
    }

    private static func cliPathPrefsURL() -> URL {
        cliPathPrefsURLOverrideForTests
            ?? AppConfig.appSupportURL().appendingPathComponent("cli-path-prefs.json")
    }

    private static func readCLIPathPrefs() -> CLIPathPrefs {
        let url = cliPathPrefsURL()
        guard let data = try? Data(contentsOf: url),
              let prefs = try? JSONDecoder().decode(CLIPathPrefs.self, from: data)
        else {
            return CLIPathPrefs()
        }
        return prefs
    }

    private static func writeCLIPathPrefs(_ prefs: CLIPathPrefs) {
        let url = cliPathPrefsURL()
        guard let data = try? JSONEncoder().encode(prefs) else { return }
        try? FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try? data.write(to: url, options: [.atomic])
    }

    private static func shellPathExportAlreadyInstalled() -> Bool {
        for url in candidateFiles() where FileManager.default.fileExists(atPath: url.path) {
            let raw = (try? String(contentsOf: url, encoding: .utf8)) ?? ""
            if raw.contains(cliShimBeginMarker) {
                return true
            }
        }
        return false
    }

    private static func ensureCLIPathExport() throws {
        if shellPathExportAlreadyInstalled() {
            return
        }

        let files = candidateFiles()
        let target = primaryFile() ?? files.first(where: {
            FileManager.default.fileExists(atPath: $0.path)
        }) ?? files.first!
        let block = """
        \(cliShimBeginMarker)
        case ":$PATH:" in
          *":$HOME/.omlx/bin:"*) ;;
          *) export PATH="$HOME/.omlx/bin:$PATH" ;;
        esac
        \(cliShimEndMarker)
        """
        try appendRawBlock(to: target, block: block)
    }

    private static func appendRawBlock(to url: URL, block: String) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        var existing = (try? String(contentsOf: url, encoding: .utf8)) ?? ""
        if !existing.hasSuffix("\n"), !existing.isEmpty {
            existing.append("\n")
        }
        existing.append(block)
        if !existing.hasSuffix("\n") {
            existing.append("\n")
        }
        try existing.write(to: url, atomically: true, encoding: .utf8)
    }

    private static func shellQuote(_ value: String) -> String {
        if value.isEmpty { return "''" }
        return "'" + value.replacingOccurrences(of: "'", with: "'\"'\"'") + "'"
    }
}

// ShellEnvWriter owns the app-managed CLI shim.
//
// Default launch behavior must not edit shell rc files. The app first creates
// `~/.omlx/bin/omlx`, then tries to expose it through a safe public symlink.
// Shell rc edits are kept behind an explicit user prompt only.

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
    private static let publicBinCandidates = [
        "/opt/homebrew/bin",
        "/usr/local/bin",
    ]

    /// Install/update `~/.omlx/bin/omlx` so app-only installs still expose
    /// the same terminal command as pip/Homebrew installs.
    @discardableResult
    static func ensureCLIShim(appBundleURL: URL = Bundle.main.bundleURL) throws -> CLISetupResult {
        let shimDir = home()
            .appendingPathComponent(".omlx", isDirectory: true)
            .appendingPathComponent("bin", isDirectory: true)
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
        let shimURL = shimDir.appendingPathComponent("omlx")
        let script = """
        #!/bin/sh
        BOOTSTRAP="$HOME/Library/Application Support/oMLX/base-path"
        if [ -r "$BOOTSTRAP" ]; then
            IFS= read -r \(variableName) < "$BOOTSTRAP" || \(variableName)=""
            if [ -n "$\(variableName)" ]; then
                export \(variableName)
            fi
        fi
        exec \(shellQuote(bundleCLI.path)) "$@"
        """
        try script.write(to: shimURL, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755],
            ofItemAtPath: shimURL.path
        )

        if let path = firstCLIPathInCurrentPath() {
            if isManagedCLI(path: path, shimURL: shimURL) {
                return .publicCommandReady(path: path.path)
            }
        }

        let symlinkResult = ensurePublicSymlink(to: shimURL)

        switch symlinkResult {
        case .installed(let path):
            if let first = firstCLIPathInCurrentPath(),
               !isManagedCLI(path: first, shimURL: shimURL) {
                return .needsShellPathPrompt(
                    reason: "\(first.path) appears before the oMLX app-managed command on PATH."
                )
            }
            return .publicCommandReady(path: path.path)
        case .failed(let reasons):
            // A GUI launch only sees the launchd PATH, so an rc-based
            // install is invisible to the PATH scan above. The shim
            // marker in a shell file is the only signal that "Update
            // Shell File" already ran.
            if shellPathExportAlreadyInstalled() {
                return .publicCommandReady(path: shimURL.path)
            }
            return .needsShellPathPrompt(reason: reasons.joined(separator: "\n"))
        }
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

    // MARK: - File targets

    private static func home() -> URL {
        homeOverrideForTests ?? FileManager.default.homeDirectoryForCurrentUser
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

    private enum PublicSymlinkResult {
        case installed(URL)
        case failed([String])
    }

    private struct CLIPathPrefs: Codable {
        var suppressShellPathPrompt: Bool = false
    }

    private static func ensurePublicSymlink(to shimURL: URL) -> PublicSymlinkResult {
        let fm = FileManager.default
        var reasons: [String] = []

        for dir in publicBinDirs() {
            guard fm.fileExists(atPath: dir.path) else {
                reasons.append("\(dir.path) does not exist.")
                continue
            }

            let link = dir.appendingPathComponent("omlx")
            if fm.fileExists(atPath: link.path) {
                if isManagedCLI(path: link, shimURL: shimURL) {
                    return .installed(link)
                }
                reasons.append("\(link.path) already exists and is not managed by oMLX.")
                continue
            }
            guard fm.isWritableFile(atPath: dir.path) else {
                reasons.append("\(dir.path) is not writable.")
                continue
            }

            do {
                try fm.createSymbolicLink(at: link, withDestinationURL: shimURL)
                return .installed(link)
            } catch {
                reasons.append("Failed to create \(link.path): \(error.localizedDescription)")
            }
        }

        if reasons.isEmpty {
            reasons.append("No writable public PATH directory was available.")
        }
        return .failed(reasons)
    }

    private static func publicBinDirs() -> [URL] {
        if let override = publicBinDirsOverrideForTests {
            return override
        }

        var seen = Set<String>()
        var dirs: [URL] = []
        let pathParts = (getenv("PATH").map { String(cString: $0) } ?? "")
            .split(separator: ":")
            .map(String.init)

        for path in pathParts where publicBinCandidates.contains(path) {
            if seen.insert(path).inserted {
                dirs.append(URL(fileURLWithPath: path, isDirectory: true))
            }
        }
        for path in publicBinCandidates where seen.insert(path).inserted {
            dirs.append(URL(fileURLWithPath: path, isDirectory: true))
        }
        return dirs
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

    private static func isManagedCLI(path: URL, shimURL: URL) -> Bool {
        path.resolvingSymlinksInPath().standardizedFileURL.path
            == shimURL.resolvingSymlinksInPath().standardizedFileURL.path
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

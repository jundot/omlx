// ShellEnvWriter — minimal helper for persisting `export OMLX_BASE_PATH=…`
// across login shells.
//
// The user-stated rule:
//   • When a non-default basePath is set, write `export OMLX_BASE_PATH="…"`
//     to the user's primary rc file so future shells (and future launches
//     of this app from a terminal) inherit it.
//   • When the path is reset to the default (`~/.omlx`), strip every
//     `OMLX_BASE_PATH` declaration from all known rc files — the default
//     doesn't need an env override.
//
// Files we touch (in order; primary write target is the first existing one
// that matches the user's $SHELL):
//   ~/.zshrc, ~/.zprofile, ~/.zshenv,
//   ~/.bashrc, ~/.bash_profile, ~/.profile.

import Foundation

enum ShellEnvWriter {
    static let variableName = "OMLX_BASE_PATH"

    /// Set or clear the variable across the user's shell rc files.
    /// - When `value` is non-nil, removes any existing line and appends
    ///   `export <NAME>="<value>"` to the user's primary rc file.
    /// - When `value` is nil, strips the variable from every known rc file
    ///   (no append).
    static func apply(value: String?) {
        let files = candidateFiles()
        // 1. Strip existing declarations from every file we know about.
        for url in files {
            try? stripDeclarations(from: url, name: variableName)
        }
        // 2. Append a fresh export to the user's primary rc file.
        guard let value, !value.isEmpty else { return }
        let target = primaryFile() ?? files.first(where: {
            FileManager.default.fileExists(atPath: $0.path)
        }) ?? files.first!  // fall back to ~/.zshrc; create on demand
        try? appendExport(to: target, name: variableName, value: value)
    }

    // MARK: - File targets

    private static func home() -> URL {
        FileManager.default.homeDirectoryForCurrentUser
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
        let shell = ProcessInfo.processInfo.environment["SHELL"] ?? ""
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

    /// Remove every line that exports the named variable. Matches:
    ///   `export NAME=value`
    ///   `export NAME="value"`
    ///   `export NAME='value'`
    /// with optional leading whitespace.
    private static func stripDeclarations(from url: URL, name: String) throws {
        guard FileManager.default.fileExists(atPath: url.path) else { return }
        let raw = try String(contentsOf: url, encoding: .utf8)
        let pattern = #"^[ \t]*export[ \t]+"# + NSRegularExpression.escapedPattern(for: name) + #"=[^\n]*\n?"#
        let regex = try NSRegularExpression(pattern: pattern, options: [.anchorsMatchLines])
        let range = NSRange(raw.startIndex..., in: raw)
        let stripped = regex.stringByReplacingMatches(
            in: raw, options: [], range: range, withTemplate: ""
        )
        if stripped == raw { return }
        try stripped.write(to: url, atomically: true, encoding: .utf8)
    }

    /// Append `export NAME="value"` to the file, creating it (and its
    /// parent directory if needed) on demand. The value is wrapped in
    /// double quotes so paths with spaces survive sourcing.
    private static func appendExport(to url: URL, name: String, value: String) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        let escaped = value
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")

        var existing = (try? String(contentsOf: url, encoding: .utf8)) ?? ""
        if !existing.hasSuffix("\n"), !existing.isEmpty {
            existing.append("\n")
        }
        existing.append("# oMLX: persisted base path override\n")
        existing.append("export \(name)=\"\(escaped)\"\n")

        try existing.write(to: url, atomically: true, encoding: .utf8)
    }
}

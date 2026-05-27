// AppConfig — single source of truth: `<basePath>/settings.json`.
//
// basePath resolution (first match wins):
//   1. OMLX_BASE_PATH env var. macOS Finder/Dock launches inherit only the
//      launchd environment — not the user's shell env — so this is mostly
//      relevant when the app was launched from a terminal or the user has
//      `launchctl setenv` configured.
//   2. ~/Library/Application Support/oMLX/base-path — a one-line
//      bootstrap file Swift owns. Written when the user changes the base
//      path through the in-app Storage row, so Finder relaunches still
//      land on the right data root. This is the *only* Swift-side config
//      we keep — its sole job is to tell us where settings.json lives.
//   3. Default `~/.omlx`.
//
// Every other field (host, port, api_key, model_dir, hf_endpoint) lives
// in `<basePath>/settings.json` — owned by the running Python server,
// written by AppConfig.save() when the server is offline. Unknown keys
// in settings.json (cache, integrations, ui, …) are preserved verbatim.
// While the server is running, prefer the HTTP PATCH path
// (/admin/api/global-settings) — the server is the owner.
//
// Env overrides for host/port/api_key are still honored last so dev
// escape-hatches always win.

import Foundation

struct AppConfig: Sendable, Equatable, Codable {
    var host: String
    var port: Int
    var apiKey: String?
    /// Always `OMLX_BASE_PATH` if set, else `~/.omlx`. Set at load() time
    /// from the current process env so the running app sees a consistent
    /// view; `AppServices.changeBasePath` updates the env in place when
    /// the user moves their data root.
    var basePath: String
    /// Always a literal path. Defaults to `<basePath>/models` and can be
    /// pointed at any folder; never empty in memory or on disk.
    var modelDir: String
    /// HuggingFace endpoint override. Empty = default `huggingface.co`.
    var hfEndpoint: String

    static var `default`: AppConfig {
        let base = currentBasePath()
        return AppConfig(
            host: "127.0.0.1",
            port: 8080,
            apiKey: nil,
            basePath: base,
            modelDir: defaultModelDir(forBasePath: base),
            hfEndpoint: ""
        )
    }

    /// `<basePath>/models` — the literal the server falls back to when the
    /// user hasn't pointed model storage somewhere else.
    static func defaultModelDir(forBasePath base: String) -> String {
        URL(fileURLWithPath: base, isDirectory: true)
            .appendingPathComponent("models", isDirectory: true)
            .path
    }

    var baseURL: URL? {
        URL(string: "http://\(host):\(port)")
    }

    // MARK: - Path resolution

    /// Resolve the user-effective base path. `OMLX_BASE_PATH` env wins,
    /// then the bootstrap file under Library/Application Support, then the
    /// `~/.omlx` default.
    static func currentBasePath() -> String {
        let env = ProcessInfo.processInfo.environment
        if let custom = env["OMLX_BASE_PATH"], !custom.isEmpty {
            return normalize(path: custom)
        }
        if let stored = readBootstrapBasePath() {
            return stored
        }
        return defaultBasePath()
    }

    private static func normalize(path: String) -> String {
        ((path as NSString).expandingTildeInPath as NSString).standardizingPath
    }

    // MARK: - Bootstrap file (basePath only)

    /// One-line file that survives Finder relaunches when the launchd env
    /// has no `OMLX_BASE_PATH`. The single field it carries is the path
    /// itself; nothing else lives here.
    static func bootstrapFileURL() -> URL {
        appSupportURL().appendingPathComponent("base-path")
    }

    /// Returns the basePath stored in the bootstrap file, normalized.
    /// `nil` when the file is missing or empty.
    static func readBootstrapBasePath() -> String? {
        let url = bootstrapFileURL()
        guard FileManager.default.fileExists(atPath: url.path),
              let raw = try? String(contentsOf: url, encoding: .utf8)
        else { return nil }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : normalize(path: trimmed)
    }

    /// Persist the basePath to the bootstrap file so future Finder launches
    /// see it. Pass `nil` to delete the file (for the "reset to default"
    /// flow).
    static func writeBootstrapBasePath(_ value: String?) throws {
        let url = bootstrapFileURL()
        if let value, !value.isEmpty {
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try (value + "\n").write(to: url, atomically: true, encoding: .utf8)
        } else if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
    }

    /// Apply a basePath choice across every layer that contributes to
    /// `currentBasePath()` resolution on the next launch:
    ///   • process env via `setenv`/`unsetenv` (so the spawned child
    ///     server inherits the choice immediately)
    ///   • bootstrap file (so Finder relaunches see it; launchd does not
    ///     inherit shell rc)
    ///   • shell rc (so terminal-launched `omlx` invocations agree)
    /// Pass `nil` (or an empty string) to clear every override — the
    /// "reset to ~/.omlx default" flow. Callers should compare against
    /// `defaultBasePath()` first and pass `nil` when the user chose the
    /// default so a default install isn't left with stale state.
    static func persistBasePath(_ path: String?) {
        let value = (path?.isEmpty ?? true) ? nil : path
        if let value {
            setenv(ShellEnvWriter.variableName, value, 1)
        } else {
            unsetenv(ShellEnvWriter.variableName)
        }
        try? writeBootstrapBasePath(value)
        ShellEnvWriter.apply(value: value)
    }

    static func defaultBasePath() -> String {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent(".omlx", isDirectory: true).path
    }

    static func settingsURL(basePath: String) -> URL {
        URL(fileURLWithPath: basePath, isDirectory: true)
            .appendingPathComponent("settings.json")
    }

    static func settingsURL() -> URL {
        settingsURL(basePath: currentBasePath())
    }

    /// Library/Application Support directory — still used for the Swift
    /// parent's stdout/stderr capture of the Python child (logs, not
    /// configuration). The base directory is created on demand.
    static func appSupportURL() -> URL {
        let base = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first ?? FileManager.default.temporaryDirectory
        let dir = base.appendingPathComponent("oMLX", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// True when settings.json exists at the current basePath. Used to
    /// gate the Welcome wizard.
    static var hasExistingConfig: Bool {
        FileManager.default.fileExists(atPath: settingsURL().path)
    }

    // MARK: - Load

    static func load() -> AppConfig {
        var c = Self.default

        if let slice = try? readSettings(basePath: c.basePath) {
            if let h = slice.host { c.host = h }
            if let p = slice.port { c.port = p }
            if let k = slice.apiKey, !k.isEmpty { c.apiKey = k }
            // settings.json may not have model_dir on a brand-new install;
            // in that case `c.modelDir` keeps the `<basePath>/models`
            // default that `Self.default` already populated.
            if let m = slice.modelDir, !m.isEmpty { c.modelDir = m }
            if let hf = slice.hfEndpoint { c.hfEndpoint = hf }
        }

        // Env overrides for non-path fields. basePath is already env-driven
        // via currentBasePath().
        let env = ProcessInfo.processInfo.environment
        if let h = env["OMLX_HOST"], !h.isEmpty { c.host = h }
        if let pStr = env["OMLX_PORT"], let p = Int(pStr) { c.port = p }
        if let k = env["OMLX_API_KEY"], !k.isEmpty { c.apiKey = k }
        return c
    }

    // MARK: - Save

    /// Write our slice into `<basePath>/settings.json`, preserving every
    /// other key in the file (cache, claude_code, integrations, …).
    func save() throws {
        let url = AppConfig.settingsURL(basePath: basePath)

        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        var json: [String: Any] = [:]
        if FileManager.default.fileExists(atPath: url.path),
           let data = try? Data(contentsOf: url),
           let existing = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            json = existing
        }

        var server = (json["server"] as? [String: Any]) ?? [:]
        server["host"] = host
        server["port"] = port
        json["server"] = server

        var auth = (json["auth"] as? [String: Any]) ?? [:]
        auth["api_key"] = apiKey ?? ""
        json["auth"] = auth

        var model = (json["model"] as? [String: Any]) ?? [:]
        // `modelDir` is always a literal path (never empty). On a basePath
        // move, AppServices.relocate() rewrites it so the persisted value
        // tracks the new root when modelDir lived inside basePath.
        model["model_dir"] = modelDir
        json["model"] = model

        var hf = (json["huggingface"] as? [String: Any]) ?? [:]
        hf["endpoint"] = hfEndpoint
        json["huggingface"] = hf

        if json["version"] == nil { json["version"] = "1.0" }

        let out = try JSONSerialization.data(
            withJSONObject: json,
            options: [.prettyPrinted]
        )
        try out.write(to: url, options: [.atomic])
    }

    // MARK: - Internal

    /// Subset of `<basePath>/settings.json` we project into AppConfig.
    struct ServerSettingsSlice {
        var host: String?
        var port: Int?
        var apiKey: String?
        var modelDir: String?
        var hfEndpoint: String?
    }

    private static func readSettings(basePath: String) throws -> ServerSettingsSlice {
        let url = settingsURL(basePath: basePath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return ServerSettingsSlice()
        }
        let data = try Data(contentsOf: url)
        let json = (try JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
        let server = json["server"] as? [String: Any]
        let auth   = json["auth"]   as? [String: Any]
        let model  = json["model"]  as? [String: Any]
        let hf     = json["huggingface"] as? [String: Any]
        return ServerSettingsSlice(
            host: server?["host"] as? String,
            port: server?["port"] as? Int,
            apiKey: auth?["api_key"] as? String,
            modelDir: model?["model_dir"] as? String,
            hfEndpoint: hf?["endpoint"] as? String
        )
    }
}

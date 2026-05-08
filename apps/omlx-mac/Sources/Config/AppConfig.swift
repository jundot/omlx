// PR 4 (PR 10 update) — minimal app configuration.
//
// Resolution order (first match wins):
//   1. Environment overrides (OMLX_HOST, OMLX_PORT, OMLX_API_KEY) — dev.
//   2. ~/Library/Application Support/oMLX-next/config.json — Welcome wizard
//      writes this in PR 10. Format mirrors today's Python config.json.
//   3. Compiled defaults (host=127.0.0.1, port=8080, apiKey=nil).
//
// PR 10 added `basePath` + `modelDir` (matches `packaging/omlx_app/config.py`'s
// ServerConfig fields), a `save()` writer, and a `configFileExists` flag used
// by the first-run check in AppDelegate.

import Foundation

struct AppConfig: Sendable, Equatable, Codable {
    var host: String
    var port: Int
    var apiKey: String?
    /// `~/.omlx` by default. The `omlx serve` child stores its `settings.json`,
    /// `models/`, logs, and SSD cache under this root.
    var basePath: String
    /// Optional override. Empty string means "use `<basePath>/models`".
    var modelDir: String
    /// HuggingFace endpoint override (`hf_endpoint`). Empty = default
    /// `huggingface.co`. Mirror users in restricted networks set this to
    /// `https://hf-mirror.com`.
    var hfEndpoint: String

    static let `default` = AppConfig(
        host: "127.0.0.1",
        port: 8080,
        apiKey: nil,
        basePath: defaultBasePath(),
        modelDir: "",
        hfEndpoint: ""
    )

    var baseURL: URL? {
        URL(string: "http://\(host):\(port)")
    }

    /// Application Support directory for the Swift rewrite. Distinct from
    /// the Python menubar's ~/Library/Application Support/oMLX/ during the
    /// side-by-side rollout (plan.md §7).
    static func appSupportURL() -> URL {
        let base = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first ?? FileManager.default.temporaryDirectory
        let dir = base.appendingPathComponent("oMLX-next", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    static func configFileURL() -> URL {
        appSupportURL().appendingPathComponent("config.json")
    }

    static var configFileExists: Bool {
        FileManager.default.fileExists(atPath: configFileURL().path)
    }

    static func defaultBasePath() -> String {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent(".omlx", isDirectory: true).path
    }

    static func load() -> AppConfig {
        var c = Self.default
        if let json = try? readConfigFile() {
            if let host = json["host"] as? String { c.host = host }
            if let port = json["port"] as? Int { c.port = port }
            if let key = json["api_key"] as? String, !key.isEmpty { c.apiKey = key }
            if let bp = json["base_path"] as? String, !bp.isEmpty { c.basePath = bp }
            if let md = json["model_dir"] as? String { c.modelDir = md }
            if let hf = json["hf_endpoint"] as? String { c.hfEndpoint = hf }
        }
        let env = ProcessInfo.processInfo.environment
        if let host = env["OMLX_HOST"], !host.isEmpty { c.host = host }
        if let portStr = env["OMLX_PORT"], let port = Int(portStr) { c.port = port }
        if let key = env["OMLX_API_KEY"], !key.isEmpty { c.apiKey = key }
        return c
    }

    func save() throws {
        let payload: [String: Any] = [
            "host": host,
            "port": port,
            "api_key": apiKey ?? "",
            "base_path": basePath,
            "model_dir": modelDir,
            "hf_endpoint": hfEndpoint,
        ]
        let data = try JSONSerialization.data(
            withJSONObject: payload,
            options: [.prettyPrinted, .sortedKeys]
        )
        try data.write(to: AppConfig.configFileURL(), options: [.atomic])
    }

    private static func readConfigFile() throws -> [String: Any] {
        let url = configFileURL()
        guard FileManager.default.fileExists(atPath: url.path) else { return [:] }
        let data = try Data(contentsOf: url)
        return (try JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
    }
}

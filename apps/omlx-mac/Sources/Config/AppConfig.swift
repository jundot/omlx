// PR 4 — minimal app configuration. Just enough to drive the menubar's
// stats poller. PR 6+ grows this with full read/write of the AppView's
// global-settings surface; PR 10 (Welcome) populates the file on first run.
//
// Resolution order (first match wins):
//   1. Environment overrides (OMLX_HOST, OMLX_PORT, OMLX_API_KEY) — dev.
//   2. ~/Library/Application Support/oMLX-next/config.json — Welcome wizard
//      writes this in PR 10. Format mirrors today's Python config.json.
//   3. Compiled defaults (host=127.0.0.1, port=8080, apiKey=nil).

import Foundation

struct AppConfig: Sendable, Equatable {
    var host: String
    var port: Int
    var apiKey: String?

    static let `default` = AppConfig(host: "127.0.0.1", port: 8080, apiKey: nil)

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

    static func load() -> AppConfig {
        var c = Self.default
        if let json = try? readConfigFile() {
            if let host = json["host"] as? String { c.host = host }
            if let port = json["port"] as? Int { c.port = port }
            if let key = json["api_key"] as? String, !key.isEmpty { c.apiKey = key }
        }
        let env = ProcessInfo.processInfo.environment
        if let host = env["OMLX_HOST"], !host.isEmpty { c.host = host }
        if let portStr = env["OMLX_PORT"], let port = Int(portStr) { c.port = port }
        if let key = env["OMLX_API_KEY"], !key.isEmpty { c.apiKey = key }
        return c
    }

    private static func readConfigFile() throws -> [String: Any] {
        let url = configFileURL()
        guard FileManager.default.fileExists(atPath: url.path) else { return [:] }
        let data = try Data(contentsOf: url)
        return (try JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
    }
}

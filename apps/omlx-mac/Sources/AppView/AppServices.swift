// PR 7 — wires AppDelegate-owned runtime objects (ServerProcess, AppConfig)
// to the SwiftUI side. AppView mounts a single instance via `.environmentObject`
// so screens can pull whatever they need without prop drilling. The screens
// keep their own data + polling state in their own view models.
//
// `serverState` republishes ServerProcess.State on every state change so a
// view can `@EnvironmentObject` AppServices and use it as a SwiftUI source
// of truth. ServerProcess itself stays NSNotification-driven (no Combine
// retrofit).

import Foundation
import SwiftUI

@MainActor
final class AppServices: NSObject, ObservableObject {
    @Published var config: AppConfig
    @Published var serverState: ServerProcess.State = .stopped
    /// PR 8 — when non-nil, the AppView swaps the Models screen for the
    /// per-model ModelSettingsScreen drilled to this id.
    @Published var modelDetailID: String?
    /// When set, AppView pulls the sidebar selection to this section on
    /// the next runloop tick and clears the request. Lets a screen
    /// imperatively navigate the user (e.g. the Profiles tab's
    /// "Edit on Server →" link) without prop-drilling a `Binding<AppSection>`.
    @Published var requestedSection: AppSection?
    /// Pair with `requestedSection` to scroll the Server screen to a
    /// specific section after the deep-link lands. ContentScaffold's
    /// `ScrollViewReader` observes this, scrolls, then nils it. Only the
    /// Default Profile anchor is wired today — extend the enum as more
    /// deep links land.
    @Published var requestedServerAnchor: ServerAnchor?

    let client: OMLXClient
    let updates: UpdateController
    /// Read-only preset bundle (sourced from the shipped JSON + remote
    /// refresh). The per-model settings preset chip strip subscribes via
    /// `@EnvironmentObject` to react to refreshes.
    let presetBundle = PresetBundleStore()

    /// Long-lived view models for the Bench screens. Owned here (not by
    /// the screen's `@StateObject`) so a running benchmark survives
    /// leaving the screen — the server keeps producing results while
    /// we're off-screen and the poll task continues updating these VMs,
    /// so coming back shows the in-flight state instead of an empty
    /// form. The HTML admin panel got this for free via Alpine's
    /// app-scoped store; SwiftUI needs the lifetime promoted manually.
    let throughputBench = ThroughputBenchScreenVM()
    let accuracyBench   = AccuracyBenchScreenVM()

    private weak var server: ServerProcess?

    init(config: AppConfig = .default, server: ServerProcess? = nil) {
        self.config = config
        self.client = OMLXClient(host: config.host, port: config.port, apiKey: config.apiKey)
        self.updates = UpdateController()
        super.init()
        self.bind(server: server)
        // Wire Sparkle (or its stub) on the next runloop so any user prefs
        // saved on disk are applied before the first background check.
        DispatchQueue.main.async { [weak self] in
            self?.updates.bootstrap()
        }
    }

    func bind(server: ServerProcess?) {
        // Detach from the previous server (if any) before re-attaching.
        if self.server != nil {
            NotificationCenter.default.removeObserver(
                self,
                name: ServerProcess.stateDidChangeNotification,
                object: nil
            )
        }
        self.server = server
        if let server {
            self.serverState = server.state
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(serverStateDidChange(_:)),
                name: ServerProcess.stateDidChangeNotification,
                object: server
            )
        }
    }

    @objc private func serverStateDidChange(_ note: Notification) {
        guard let proc = note.object as? ServerProcess, proc === server else { return }
        // ServerProcess posts on the main queue (via DispatchQueue.main.async
        // in terminationHandler / @MainActor health-check Task), so we're
        // already on the main thread here.
        serverState = proc.state
    }

    func updateConfig(_ next: AppConfig) {
        self.config = next
        client.configure(host: next.host, port: next.port, apiKey: next.apiKey)
    }

    // MARK: - Server lifecycle (proxied to ServerProcess)

    var hasServer: Bool { server != nil }

    @discardableResult
    func startServer() throws -> ServerProcess.StartResult? {
        try server?.start()
    }

    func stopServer() async {
        await server?.stop()
    }

    func restartServer() async throws {
        await server?.stop()
        _ = try server?.start()
    }

    func forceRestartServer() async throws {
        _ = try await server?.forceRestart()
    }

    enum BasePathChangeError: LocalizedError {
        case sameAsCurrent
        case destinationNotEmpty(String)
        case destinationNotWritable(String)
        case moveFailed(String)

        var errorDescription: String? {
            switch self {
            case .sameAsCurrent:
                return "Base path is unchanged."
            case .destinationNotEmpty(let p):
                return "\(p) already exists and isn't empty. Pick an unused folder."
            case .destinationNotWritable(let p):
                return "Can't write to \(p)."
            case .moveFailed(let m):
                return "Move failed: \(m)"
            }
        }
    }

    /// Apply pending edits to the storage layout. Both arguments are
    /// optional so the caller (Server screen → Apply) can submit only
    /// what actually changed:
    ///   • `basePath`: relocates every file under the current root, sets
    ///     `OMLX_BASE_PATH` (env + bootstrap file + shell rc), and
    ///     reconfigures the spawn args.
    ///   • `modelDir`: writes the explicit override into
    ///     `<basePath>/settings.json`. Empty string clears it back to the
    ///     server's default (`<basePath>/models`).
    /// The server is stopped once before any mutation and restarted once
    /// at the end — the user-stated rule: restart only fires when at
    /// least one of the inputs actually differs from the current config.
    func applyStorageChanges(basePath: String? = nil, modelDir: String? = nil) async throws {
        let normalizedBase = basePath.map(Self.normalize)
        let trimmedDir = modelDir?.trimmingCharacters(in: .whitespacesAndNewlines)

        let basePathChanging: Bool = {
            guard let normalizedBase else { return false }
            return normalizedBase != Self.normalize(config.basePath)
        }()
        let modelDirChanging: Bool = {
            guard let trimmedDir else { return false }
            return trimmedDir != config.modelDir
        }()

        guard basePathChanging || modelDirChanging else {
            throw BasePathChangeError.sameAsCurrent
        }

        // Stop the server BEFORE any filesystem mutation so an open log
        // file or SSD cache doesn't corrupt the move.
        if let server { await server.stop() }

        if basePathChanging, let newPath = normalizedBase {
            try migrateBasePath(to: newPath)
        }

        if modelDirChanging, let newDir = trimmedDir {
            var updated = config
            updated.modelDir = newDir
            try updated.save()
            self.config = updated
        }

        if let server {
            let baseURL = URL(fileURLWithPath: config.basePath, isDirectory: true)
            try server.reconfigure(basePath: baseURL)
            _ = try server.start()
        }
    }

    /// Move every file under the current basePath to `newPath` and persist
    /// the choice. Caller must have already stopped the server.
    private func migrateBasePath(to newPath: String) throws {
        let fm = FileManager.default
        let oldPath = Self.normalize(config.basePath)
        let oldURL = URL(fileURLWithPath: oldPath, isDirectory: true)
        let newURL = URL(fileURLWithPath: newPath, isDirectory: true)

        // Ensure the destination's parent exists. If the destination itself
        // already exists we require it to be empty so we don't accidentally
        // overwrite an unrelated folder.
        try fm.createDirectory(
            at: newURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        if fm.fileExists(atPath: newURL.path) {
            let entries = (try? fm.contentsOfDirectory(atPath: newURL.path)) ?? []
            if !entries.isEmpty {
                let preview = entries.prefix(4).joined(separator: ", ")
                let suffix = entries.count > 4 ? ", …" : ""
                throw BasePathChangeError.destinationNotEmpty(
                    "\(newPath) (\(entries.count) item\(entries.count == 1 ? "" : "s"): \(preview)\(suffix))"
                )
            }
            try? fm.removeItem(at: newURL)
        }

        do {
            if fm.fileExists(atPath: oldURL.path) {
                try fm.moveItem(at: oldURL, to: newURL)
            } else {
                try fm.createDirectory(at: newURL, withIntermediateDirectories: true)
            }
        } catch {
            throw BasePathChangeError.moveFailed(error.localizedDescription)
        }

        // When the user resets to the `~/.omlx` default, clear every
        // override so a default install isn't left with stale state.
        let isDefault = (newPath == AppConfig.defaultBasePath())
        AppConfig.persistBasePath(isDefault ? nil : newPath)

        var updated = config
        updated.basePath = newPath
        // If the explicit modelDir lived under the OLD basePath, rewrite
        // its prefix so it tags along — files were physically moved by the
        // moveItem() above, so the new path is where they actually live.
        // A modelDir outside the old basePath (e.g. /Volumes/SSD/models)
        // stays put untouched.
        updated.modelDir = Self.relocate(path: config.modelDir, oldBase: oldPath, newBase: newPath)
        // Persist any unknown server keys at the new location — settings.json
        // moved with the directory, so this is mostly a refresh of our slice
        // for first installs that didn't have one yet.
        try? updated.save()
        self.config = updated

        // AppConfig.save() only owns its own slice; settings.json also carries
        // path-bearing fields the Python server owns (model.model_dirs list,
        // cache.ssd_cache_dir, logging.log_dir). When those were persisted as
        // absolute paths under the old basePath, the server reads them after
        // the move and recreates dirs at the stale path. Rewrite them here.
        // Errors are surfaced via NSLog so a silent failure is debuggable in
        // Console.app — but we don't fail the migration (move already worked).
        do {
            try Self.relocateOrphanPaths(in: AppConfig.settingsURL(basePath: newPath),
                                         oldBase: oldPath, newBase: newPath)
        } catch {
            NSLog("oMLX: relocateOrphanPaths failed: %@", String(describing: error))
        }
    }

    /// If `path` is inside `oldBase`, swap the prefix to `newBase`.
    /// Returns the input unchanged when it's empty or sits outside the
    /// migrated tree. Internal so unit tests can drive it directly. Pure —
    /// `nonisolated` so it's callable without bouncing onto MainActor.
    nonisolated static func relocate(path: String, oldBase: String, newBase: String) -> String {
        guard !path.isEmpty else { return path }
        let normalized = normalize(path)
        let oldRoot = oldBase
        if normalized == oldRoot {
            return newBase
        }
        let oldPrefix = oldRoot.hasSuffix("/") ? oldRoot : oldRoot + "/"
        if normalized.hasPrefix(oldPrefix) {
            let suffix = String(normalized.dropFirst(oldPrefix.count))
            return URL(fileURLWithPath: newBase, isDirectory: true)
                .appendingPathComponent(suffix).path
        }
        return path
    }

    /// Rewrite path-bearing fields in `<basePath>/settings.json` that
    /// AppConfig doesn't own (model.model_dirs, cache.ssd_cache_dir,
    /// logging.log_dir). Paths outside the migrated tree are left alone.
    nonisolated static func relocateOrphanPaths(in url: URL, oldBase: String, newBase: String) throws {
        NSLog("oMLX: relocateOrphanPaths in=%@ old=%@ new=%@",
              url.path, oldBase, newBase)
        guard FileManager.default.fileExists(atPath: url.path) else {
            NSLog("oMLX: relocateOrphanPaths skipped — file does not exist")
            return
        }
        let data = try Data(contentsOf: url)
        guard var json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            NSLog("oMLX: relocateOrphanPaths skipped — root is not an object")
            return
        }

        if var model = json["model"] as? [String: Any] {
            if let dirs = model["model_dirs"] as? [String] {
                model["model_dirs"] = dirs.map {
                    Self.relocate(path: $0, oldBase: oldBase, newBase: newBase)
                }
            }
            if let dir = model["model_dir"] as? String, !dir.isEmpty {
                model["model_dir"] = Self.relocate(path: dir, oldBase: oldBase, newBase: newBase)
            }
            json["model"] = model
        }

        if var cache = json["cache"] as? [String: Any] {
            if let dir = cache["ssd_cache_dir"] as? String, !dir.isEmpty {
                cache["ssd_cache_dir"] = Self.relocate(path: dir, oldBase: oldBase, newBase: newBase)
            }
            json["cache"] = cache
        }

        if var logging = json["logging"] as? [String: Any] {
            if let dir = logging["log_dir"] as? String, !dir.isEmpty {
                logging["log_dir"] = Self.relocate(path: dir, oldBase: oldBase, newBase: newBase)
            }
            json["logging"] = logging
        }

        let out = try JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted])
        try out.write(to: url, options: [.atomic])
        NSLog("oMLX: relocateOrphanPaths wrote %d bytes", out.count)
    }

    nonisolated private static func normalize(_ path: String) -> String {
        ((path as NSString).expandingTildeInPath as NSString).standardizingPath
    }

    /// Persist a new host/port to AppConfig, reconfigure the running server
    /// process, and bounce it. Without this, ServerScreenVM.savePort would
    /// only update the server's `settings.json`, but the next spawn still
    /// uses the cached --host / --port arguments captured at app launch.
    ///
    /// The Python server is the canonical writer of `settings.json` while
    /// it's running — the caller already PATCHed it before us, so we don't
    /// double-write here. When the server is offline (wizard dropouts,
    /// dev), we DO write so the next spawn reads the right values.
    func applyServerEndpoint(host: String? = nil, port: Int? = nil) async throws {
        let resolvedHost = host ?? config.host
        let resolvedPort = port ?? config.port

        var updated = config
        updated.host = resolvedHost
        updated.port = resolvedPort
        if server == nil {
            try updated.save()
        }
        self.config = updated

        // The HTTP client also needs to know about the new endpoint so future
        // admin calls land on the new bind address.
        client.configure(host: resolvedHost, port: resolvedPort, apiKey: updated.apiKey)

        if let server {
            await server.stop()
            try server.reconfigure(host: resolvedHost, port: resolvedPort)
            _ = try server.start()
        }
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
    }
}

/// Scroll anchors inside the Server screen that other screens can deep
/// link to via `AppServices.requestedServerAnchor`. The raw value is the
/// `.id(_:)` attached to the corresponding `SectionHeader`.
enum ServerAnchor: String, Sendable {
    case defaultProfile = "server.defaultProfile"
}

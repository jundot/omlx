// D1 — the app uninstalls itself, with explicit keep options.
//
// One button (Settings → Advanced → Uninstall…) opens a sheet that lists every
// scope with its own keep switch, shows a byte summary before anything is
// removed, and names the model directories it keeps verbatim. Models and
// cluster/identity.json are kept by default; model deletion is opt-in per
// directory. The helper is the only admin prompt; a cancel leaves the rest
// done and is reported. LaunchServices hygiene (lsregister -u per removed
// bundle, -gc, Dock repaint) is part of "uninstalled", and the app quits last.
//
// The model layer is split so XCTests can drive it without deleting anything
// real: `UninstallLocator` carries injected paths, `UninstallPlan.build` is a
// read-only enumeration, and `UninstallRunner.run` performs every side effect
// through an injectable `UninstallActions` seam. The lab/CI equivalent is
// scripts/cleanup-cluster-macs.sh, which removes the same scope set.

import AppKit
import Foundation

// MARK: - Locator (injected paths)

/// Every path the uninstall can touch. Production wires it from `AppConfig`
/// and the running bundle; tests point it at temporary directories.
struct UninstallLocator: Sendable {
    var appBundleURL: URL
    var helperToolPath: String
    var helperDaemonPlistPath: String
    /// ~/Library/Application Support/oMLX — runtime venv, logs, updater state,
    /// the base-path bootstrap file. Removed wholesale by the Runtime scope.
    var appSupportURL: URL
    /// The data root, default ~/.omlx.
    var basePath: String
    /// User-chosen model roots, named verbatim in the "what is kept" line.
    var modelDirs: [String]
    /// ~/Library/Logs/oMLX — defensive; the lab script removes it too.
    var diagnosticsURL: URL
    var cachesURL: URL
    var savedStateURLs: [URL]
    var lsregisterPath: String

    static func live(
        config: AppConfig,
        bundleURL: URL = Bundle.main.bundleURL
    ) -> UninstallLocator {
        let fm = FileManager.default
        let home = fm.homeDirectoryForCurrentUser
        let library = fm.urls(for: .libraryDirectory, in: .userDomainMask).first
            ?? home.appendingPathComponent("Library", isDirectory: true)
        return UninstallLocator(
            appBundleURL: bundleURL,
            helperToolPath: PrivilegedHelper.installedPath,
            helperDaemonPlistPath: "/Library/LaunchDaemons/app.omlx.helper.plist",
            appSupportURL: AppConfig.appSupportURL(),
            basePath: config.basePath,
            modelDirs: config.effectiveModelDirs,
            diagnosticsURL: library.appendingPathComponent("Logs/oMLX", isDirectory: true),
            cachesURL: library.appendingPathComponent("Caches/oMLX", isDirectory: true),
            savedStateURLs: [
                library.appendingPathComponent("Saved Application State/omlx.oMLX.savedState"),
                library.appendingPathComponent("Saved Application State/app.omlx.oMLX.savedState"),
            ],
            lsregisterPath: "/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
        )
    }

    var baseDirectoryURL: URL {
        URL(fileURLWithPath: basePath, isDirectory: true)
    }
    var clusterDirectoryURL: URL {
        baseDirectoryURL.appendingPathComponent("cluster", isDirectory: true)
    }
    var identityURL: URL {
        clusterDirectoryURL.appendingPathComponent("identity.json")
    }
}

// MARK: - Options (keep switches)

/// Defaults mirror the frozen decision: remove app/helper/runtime/settings/
/// cluster-state/diagnostics, **keep** identity and every model directory.
struct UninstallOptions: Equatable, Sendable {
    var removeApp = true
    var removeHelper = true
    var removeRuntime = true
    var removeSettings = true
    var removeClusterState = true
    var removeIdentity = false
    var removeDiagnostics = true
    /// Opt-in per directory; empty keeps every model root.
    var modelDirsToRemove: Set<String> = []

    static let `default` = UninstallOptions()
}

// MARK: - Plan (read-only enumeration + byte counts)

struct UninstallItem: Equatable, Sendable {
    enum Method: Equatable, Sendable {
        /// Move to Trash (the running app bundle).
        case trash
        /// Remove from disk.
        case delete
        /// The one admin dialog (SMJobRemove bootouts + removes the tool).
        case helperAdmin
    }
    /// Scope raw value, or "models" for an opted-in model directory.
    let scope: String
    let url: URL
    let bytes: Int64
    let method: Method
}

struct UninstallPlan: Sendable {
    let items: [UninstallItem]
    let keptModelDirs: [String]
    let keepsIdentity: Bool

    var totalBytes: Int64 { items.reduce(0) { $0 + $1.bytes } }
    var removesApp: Bool { items.contains { $0.scope == "app" } }
    var appBundleURL: URL? { items.first { $0.scope == "app" }?.url }
}

enum UninstallPlanner {
    /// Enumerate what the chosen options would remove, with byte counts. Reads
    /// the filesystem but never mutates it, so the sheet can show the summary
    /// before the user confirms. Injected FileManager keeps tests in temp dirs.
    static func build(
        locator: UninstallLocator,
        options: UninstallOptions,
        fileManager: FileManager = .default
    ) -> UninstallPlan {
        var items: [UninstallItem] = []
        let fm = fileManager

        func exists(_ url: URL) -> Bool {
            fm.fileExists(atPath: url.path)
        }
        func add(_ scope: String, _ url: URL, _ method: UninstallItem.Method) {
            guard exists(url) else { return }
            items.append(UninstallItem(scope: scope, url: url, bytes: size(of: url, fm: fm), method: method))
        }

        if options.removeApp {
            add("app", locator.appBundleURL, .trash)
        }
        if options.removeHelper {
            let tool = URL(fileURLWithPath: locator.helperToolPath)
            let plist = URL(fileURLWithPath: locator.helperDaemonPlistPath)
            if exists(tool) || exists(plist) {
                let bytes = size(of: tool, fm: fm) + size(of: plist, fm: fm)
                items.append(UninstallItem(scope: "helper", url: tool, bytes: bytes, method: .helperAdmin))
            }
        }
        if options.removeRuntime {
            add("runtime", locator.appSupportURL, .delete)
        }
        if options.removeSettings {
            // Everything at the data root except the cluster dir and backups —
            // settings.json, model_settings.json, logs, bin, assistant-*.json.
            // Mirrors the lab script's "remove ~/.omlx/* except cluster".
            let children = (try? fm.contentsOfDirectory(
                at: locator.baseDirectoryURL,
                includingPropertiesForKeys: nil,
                options: []
            )) ?? []
            for child in children.sorted(by: { $0.path < $1.path }) {
                let name = child.lastPathComponent
                if name == "cluster" || name.hasPrefix("backup-cluster-") { continue }
                add("settings", child, .delete)
            }
        }
        if options.removeClusterState {
            // cluster/* except identity.json — identity has its own switch.
            let children = (try? fm.contentsOfDirectory(
                at: locator.clusterDirectoryURL,
                includingPropertiesForKeys: nil,
                options: []
            )) ?? []
            for child in children.sorted(by: { $0.path < $1.path }) {
                if child.lastPathComponent == "identity.json" { continue }
                add("clusterState", child, .delete)
            }
        }
        if options.removeIdentity {
            add("identity", locator.identityURL, .delete)
        }
        if options.removeDiagnostics {
            add("diagnostics", locator.diagnosticsURL, .delete)
            add("diagnostics", locator.cachesURL, .delete)
            for url in locator.savedStateURLs { add("diagnostics", url, .delete) }
        }
        for dir in locator.modelDirs where options.modelDirsToRemove.contains(dir) {
            add("models", URL(fileURLWithPath: dir, isDirectory: true), .delete)
        }

        let keptModels = locator.modelDirs.filter { !options.modelDirsToRemove.contains($0) }
        return UninstallPlan(
            items: items,
            keptModelDirs: keptModels,
            keepsIdentity: !options.removeIdentity
        )
    }

    /// Sum of regular-file sizes under a URL (0 when absent). Directory walks
    /// follow the same enumerator the byte summary needs; no mutation.
    static func size(of url: URL, fm: FileManager = .default) -> Int64 {
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: url.path, isDirectory: &isDir) else { return 0 }
        if !isDir.boolValue {
            let attrs = try? fm.attributesOfItem(atPath: url.path)
            return (attrs?[.size] as? Int64) ?? 0
        }
        var total: Int64 = 0
        guard let enumerator = fm.enumerator(
            at: url,
            includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey],
            options: []
        ) else { return 0 }
        for case let fileURL as URL in enumerator {
            let values = try? fileURL.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
            if values?.isRegularFile == true {
                total += Int64(values?.fileSize ?? 0)
            }
        }
        return total
    }
}

// MARK: - Runner (injectable side effects)

/// Every effect the uninstall performs, behind one seam. Production wires the
/// real FileManager / privileged removal / lsregister / quit; tests inject a
/// recorder and temporary directories so nothing real is deleted.
struct UninstallActions {
    var trash: (URL) throws -> Void
    var delete: (URL) throws -> Void
    var removeHelper: () -> HelperRemovalOutcome
    var unregisterBundle: (URL) -> Void
    var garbageCollectLaunchServices: () -> Void
    var repaintDock: () -> Void
    var quitApp: () -> Void

    static func live(fileManager: FileManager = .default) -> UninstallActions {
        UninstallActions(
            trash: { url in
                var resulting: NSURL?
                try fileManager.trashItem(at: url, resultingItemURL: &resulting)
            },
            delete: { url in try fileManager.removeItem(at: url) },
            removeHelper: { PrivilegedHelper.removeForUninstall() },
            unregisterBundle: { url in
                runProcess(UninstallLocator.live(config: .default).lsregisterPath, ["-u", url.path])
            },
            garbageCollectLaunchServices: {
                let path = UninstallLocator.live(config: .default).lsregisterPath
                runProcess(path, ["-gc", "-f", "-r", "-domain", "user", "-domain", "local"])
            },
            repaintDock: { runProcess("/usr/bin/killall", ["Dock"]) },
            quitApp: { NSApp.terminate(nil) }
        )
    }

    /// Synchronous, fire-and-forget helper for the LS hygiene commands. A
    /// failure here is cosmetic (an icon), never fatal to the uninstall.
    private static func runProcess(_ launchPath: String, _ arguments: [String]) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: launchPath)
        process.arguments = arguments
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try? process.run()
        process.waitUntilExit()
    }
}

struct UninstallReport: Sendable {
    var removedScopes: [String] = []
    var removedURLs: [URL] = []
    var helperOutcome: HelperRemovalOutcome?
    var keptModelDirs: [String] = []
    var keepsIdentity = true
    var errors: [String] = []
    var didQuit = false
    /// The app bundle was moved to the Trash; quitting is the final step.
    var appRemoved = false

    var helperLeftInPlace: Bool {
        switch helperOutcome {
        case .cancelled, .failed: return true
        default: return false
        }
    }
}

enum UninstallRunner {
    /// Execute the plan. Each item is independent: a failure (or a cancelled
    /// helper dialog) is recorded and the rest still runs. LaunchServices
    /// hygiene follows removal. With `quitAfter` (the default) the app quits
    /// last; the sheet passes `false` so it can show the report — including a
    /// left-in-place helper — before the explicit final quit.
    static func run(
        plan: UninstallPlan,
        actions: UninstallActions,
        quitAfter: Bool = true
    ) -> UninstallReport {
        var report = UninstallReport()
        report.keptModelDirs = plan.keptModelDirs
        report.keepsIdentity = plan.keepsIdentity

        for item in plan.items {
            switch item.method {
            case .trash:
                do {
                    try actions.trash(item.url)
                    report.removedScopes.append(item.scope)
                    report.removedURLs.append(item.url)
                } catch {
                    report.errors.append("\(item.scope): \(error.localizedDescription)")
                }
            case .delete:
                do {
                    try actions.delete(item.url)
                    report.removedScopes.append(item.scope)
                    report.removedURLs.append(item.url)
                } catch {
                    report.errors.append("\(item.scope): \(error.localizedDescription)")
                }
            case .helperAdmin:
                let outcome = actions.removeHelper()
                report.helperOutcome = outcome
                switch outcome {
                case .removed:
                    report.removedScopes.append(item.scope)
                    report.removedURLs.append(item.url)
                case .cancelled:
                    report.errors.append("helper: the administrator prompt was cancelled; the helper was left in place.")
                case .failed(let message):
                    report.errors.append("helper: \(message)")
                }
            }
        }

        // LaunchServices hygiene: unregister the removed bundle, garbage
        // collect, repaint the Dock. A missing icon is part of "uninstalled".
        if let appURL = plan.appBundleURL {
            actions.unregisterBundle(appURL)
        }
        actions.garbageCollectLaunchServices()
        actions.repaintDock()

        // The app quits last, after everything else is done.
        report.appRemoved = plan.removesApp
        if plan.removesApp && quitAfter {
            actions.quitApp()
            report.didQuit = true
        }
        return report
    }
}

// MARK: - Formatting

enum UninstallFormat {
    /// Human-readable byte count for the summary (GB/MB/KB, one decimal).
    static func bytes(_ value: Int64) -> String {
        let gb = Double(value) / 1_073_741_824.0
        if gb >= 1 { return String(format: "%.1f GB", gb) }
        let mb = Double(value) / 1_048_576.0
        if mb >= 1 { return String(format: "%.1f MB", mb) }
        let kb = Double(value) / 1024.0
        if kb >= 1 { return String(format: "%.1f KB", kb) }
        return "\(value) B"
    }
}
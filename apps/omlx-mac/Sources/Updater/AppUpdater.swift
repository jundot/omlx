// In-place auto-updater — direct port of the PyObjC menubar's updater.
//
// Flow: download .dmg → hdiutil attach → copy the inner oMLX.app next to
// the running bundle as `.oMLX-update.app` → hdiutil detach → on
// confirmation, spawn a detached bash script that waits for our PID to
// exit, swaps the staged bundle into place, strips the quarantine xattr,
// and `open`s the new bundle. No EdDSA signature check — Apple's
// notarization stapled to the DMG is the trust boundary.
//
// Cancellation: `cancel()` is best-effort; an in-flight download exits at
// the next stream chunk. A staged copy that's already on disk gets
// cleaned up by `cleanupStaged()` on the next launch.

import AppKit
import Foundation

@MainActor
final class AppUpdater {
    enum UpdateError: Error, CustomStringConvertible {
        case notWritable(String)
        case downloadFailed(String)
        case mountFailed(String)
        case appNotFoundInVolume
        case stageFailed(String)
        case cancelled

        var description: String {
            switch self {
            case .notWritable(let path):
                return "Cannot write to \(path). Move oMLX.app to a writable location and try again."
            case .downloadFailed(let m): return "Download failed: \(m)"
            case .mountFailed(let m): return "Could not mount DMG: \(m)"
            case .appNotFoundInVolume: return "oMLX.app not found inside the downloaded DMG"
            case .stageFailed(let m): return "Could not stage the update: \(m)"
            case .cancelled: return "Update cancelled"
            }
        }
    }

    enum Progress: Sendable {
        case starting
        case downloading(percent: Int, receivedBytes: Int64, totalBytes: Int64)
        case mounting
        case staging
        case ready
    }

    static let stagedAppName = ".oMLX-update.app"

    private let dmgURL: URL
    private let version: String
    private let onProgress: @MainActor (Progress) -> Void
    private let onError: @MainActor (UpdateError) -> Void
    private let onReady: @MainActor () -> Void

    private var task: Task<Void, Never>?
    private var session: URLSession?
    private var downloadTask: URLSessionDownloadTask?
    private var cancelled = false

    init(
        dmgURL: URL,
        version: String,
        onProgress: @escaping @MainActor (Progress) -> Void,
        onError: @escaping @MainActor (UpdateError) -> Void,
        onReady: @escaping @MainActor () -> Void
    ) {
        self.dmgURL = dmgURL
        self.version = version
        self.onProgress = onProgress
        self.onError = onError
        self.onReady = onReady
    }

    static func appBundleURL() -> URL {
        Bundle.main.bundleURL
    }

    static func isWritable(_ app: URL) -> Bool {
        FileManager.default.isWritableFile(atPath: app.deletingLastPathComponent().path)
    }

    /// Best-effort cleanup of a leftover staged bundle from a prior attempt.
    /// Call once on launch.
    static func cleanupStaged() {
        let app = appBundleURL()
        let staged = app.deletingLastPathComponent().appendingPathComponent(stagedAppName)
        try? FileManager.default.removeItem(at: staged)
    }

    func start() {
        let app = Self.appBundleURL()
        guard Self.isWritable(app) else {
            onError(.notWritable(app.deletingLastPathComponent().path))
            return
        }

        task = Task { [weak self] in
            guard let self else { return }
            await self.run(app: app)
        }
    }

    func cancel() {
        cancelled = true
        downloadTask?.cancel()
        task?.cancel()
    }

    private func run(app: URL) async {
        onProgress(.starting)

        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("omlx-update-\(UUID().uuidString)")
        do {
            try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        } catch {
            onError(.downloadFailed("Could not create temp dir: \(error.localizedDescription)"))
            return
        }
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let dmgPath = tmpDir.appendingPathComponent("oMLX-\(version).dmg")

        do {
            try await downloadDMG(to: dmgPath)
        } catch let err as UpdateError {
            if !cancelled { onError(err) }
            return
        } catch {
            onError(.downloadFailed(error.localizedDescription))
            return
        }

        if cancelled { return }
        onProgress(.mounting)

        let mountPoint: URL
        do {
            mountPoint = try mountDMG(at: dmgPath)
        } catch let err as UpdateError {
            onError(err); return
        } catch {
            onError(.mountFailed(error.localizedDescription)); return
        }

        defer { _ = try? unmountDMG(at: mountPoint) }

        if cancelled { return }
        onProgress(.staging)

        let stagedApp = app.deletingLastPathComponent().appendingPathComponent(Self.stagedAppName)
        do {
            try stageApp(fromMount: mountPoint, to: stagedApp)
        } catch let err as UpdateError {
            onError(err); return
        } catch {
            onError(.stageFailed(error.localizedDescription)); return
        }

        if cancelled { return }
        onProgress(.ready)
        onReady()
    }

    // MARK: - Download

    private func downloadDMG(to dest: URL) async throws {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 3600
        let session = URLSession(configuration: config)
        self.session = session
        defer { session.invalidateAndCancel(); self.session = nil }

        let (bytes, response) = try await session.bytes(from: dmgURL)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            throw UpdateError.downloadFailed("HTTP \(code)")
        }

        let total = response.expectedContentLength
        var received: Int64 = 0
        var lastReportedPct = -1

        FileManager.default.createFile(atPath: dest.path, contents: nil)
        guard let handle = try? FileHandle(forWritingTo: dest) else {
            throw UpdateError.downloadFailed("Could not open \(dest.lastPathComponent) for writing")
        }
        defer { try? handle.close() }

        var chunk = Data()
        chunk.reserveCapacity(256 * 1024)
        for try await byte in bytes {
            if cancelled { throw UpdateError.cancelled }
            chunk.append(byte)
            received += 1
            if chunk.count >= 256 * 1024 {
                try handle.write(contentsOf: chunk)
                chunk.removeAll(keepingCapacity: true)
                if total > 0 {
                    let pct = Int(received * 100 / total)
                    if pct != lastReportedPct {
                        lastReportedPct = pct
                        await MainActor.run {
                            self.onProgress(.downloading(percent: pct, receivedBytes: received, totalBytes: total))
                        }
                    }
                }
            }
        }
        if !chunk.isEmpty {
            try handle.write(contentsOf: chunk)
        }
    }

    // MARK: - Mount / unmount

    private func mountDMG(at dmg: URL) throws -> URL {
        let result = try runProcess(
            "/usr/bin/hdiutil",
            args: ["attach", "-nobrowse", "-noverify", "-noautoopen", "-mountrandom", "/tmp", dmg.path]
        )
        guard result.status == 0 else {
            throw UpdateError.mountFailed(result.stderr.isEmpty ? result.stdout : result.stderr)
        }
        // hdiutil prints `<dev>\t<protocol>\t<mountpoint>` lines. Mount
        // point is the trailing column of the last line that names a
        // directory under /tmp.
        for line in result.stdout.split(whereSeparator: \.isNewline).reversed() {
            let cols = line.components(separatedBy: "\t").map { $0.trimmingCharacters(in: .whitespaces) }
            if let last = cols.last, !last.isEmpty {
                var isDir: ObjCBool = false
                if FileManager.default.fileExists(atPath: last, isDirectory: &isDir), isDir.boolValue {
                    return URL(fileURLWithPath: last)
                }
            }
        }
        throw UpdateError.mountFailed("Could not parse hdiutil output")
    }

    @discardableResult
    private func unmountDMG(at mountPoint: URL) throws -> Bool {
        let result = try runProcess(
            "/usr/bin/hdiutil",
            args: ["detach", mountPoint.path, "-force"]
        )
        return result.status == 0
    }

    // MARK: - Stage

    private func stageApp(fromMount mountPoint: URL, to stagedApp: URL) throws {
        let appInVolume = try findAppInVolume(mountPoint)
        if FileManager.default.fileExists(atPath: stagedApp.path) {
            try FileManager.default.removeItem(at: stagedApp)
        }
        // `ditto` preserves resource forks, extended attributes, and
        // symlinks — straight `FileManager.copyItem` is known to drop
        // some of those on .app bundles.
        let result = try runProcess(
            "/usr/bin/ditto",
            args: [appInVolume.path, stagedApp.path]
        )
        guard result.status == 0 else {
            throw UpdateError.stageFailed(result.stderr.isEmpty ? result.stdout : result.stderr)
        }
    }

    private func findAppInVolume(_ mountPoint: URL) throws -> URL {
        let preferred = mountPoint.appendingPathComponent("oMLX.app")
        if FileManager.default.fileExists(atPath: preferred.path) { return preferred }
        let entries = (try? FileManager.default.contentsOfDirectory(atPath: mountPoint.path)) ?? []
        for name in entries where name.hasSuffix(".app") {
            return mountPoint.appendingPathComponent(name)
        }
        throw UpdateError.appNotFoundInVolume
    }

    // MARK: - Swap + relaunch (called from outside, right before terminate)

    /// Spawns a detached bash script that:
    ///   1. waits for our PID to exit
    ///   2. replaces the running .app with the staged one
    ///   3. strips com.apple.quarantine
    ///   4. `open`s the replaced .app
    /// Must be called immediately before `NSApp.terminate(nil)`.
    @discardableResult
    static func performSwapAndRelaunch() -> Bool {
        let app = appBundleURL()
        let staged = app.deletingLastPathComponent().appendingPathComponent(stagedAppName)
        guard FileManager.default.fileExists(atPath: staged.path) else { return false }

        let pid = ProcessInfo.processInfo.processIdentifier
        let appPath = app.path.replacingOccurrences(of: "\"", with: "\\\"")
        let stagedPath = staged.path.replacingOccurrences(of: "\"", with: "\\\"")
        let script = """
        #!/bin/bash
        while kill -0 \(pid) 2>/dev/null; do
            sleep 0.2
        done
        sleep 0.5
        rm -rf "\(appPath)"
        mv "\(stagedPath)" "\(appPath)"
        xattr -rd com.apple.quarantine "\(appPath)" 2>/dev/null
        open "\(appPath)"
        """

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = ["-c", script]
        // Detach: orphan into a new session so the script outlives our process.
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
        } catch {
            NSLog("oMLX: failed to spawn swap script: %@", error.localizedDescription)
            return false
        }
        return true
    }
}

// MARK: - Process helper

private struct ProcessResult {
    let status: Int32
    let stdout: String
    let stderr: String
}

private func runProcess(_ executable: String, args: [String]) throws -> ProcessResult {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: executable)
    process.arguments = args
    let stdoutPipe = Pipe()
    let stderrPipe = Pipe()
    process.standardOutput = stdoutPipe
    process.standardError = stderrPipe
    try process.run()
    process.waitUntilExit()
    let stdout = String(data: stdoutPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
    let stderr = String(data: stderrPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
    return ProcessResult(status: process.terminationStatus, stdout: stdout, stderr: stderr)
}

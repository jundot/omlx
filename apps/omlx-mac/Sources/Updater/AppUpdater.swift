// In-place auto-updater.
//
// Flow: download .dmg -> hdiutil attach -> copy the inner oMLX.app next to
// the running bundle as `.oMLX-update.app` -> check its signature against
// the release team -> hdiutil detach -> on confirmation, register a one-shot
// launchd worker that waits for our PID to exit, atomically swaps the staged
// bundle into place, strips the quarantine xattr, and `open`s the new bundle.
// The quarantine strip skips Gatekeeper, so the signature check is the trust
// boundary.
//
// Cancellation: `cancel()` is best-effort; an in-flight download exits at
// the next stream chunk.
// A staged copy newer than the running app stays on disk across launches.
// The next check reuses it only after a fresh signature check.

import AppKit
import Foundation

@MainActor
final class AppUpdater {
    enum UpdateError: Error, CustomStringConvertible {
        case notWritable(String)
        case downloadFailed(String)
        case mountFailed(String)
        case unmountFailed(String)
        case cleanupFailed(String)
        case appNotFoundInVolume
        case stageFailed(String)
        case signatureInvalid(String)
        case cancelled

        var description: String {
            switch self {
            case .notWritable(let path):
                return "Cannot write to \(path). Move oMLX.app to a writable location and try again."
            case .downloadFailed(let m): return "Download failed: \(m)"
            case .mountFailed(let m): return "Could not mount DMG: \(m)"
            case .unmountFailed(let m): return "Could not unmount DMG: \(m)"
            case .cleanupFailed(let m): return "Could not clean up update files: \(m)"
            case .appNotFoundInVolume: return "oMLX.app not found inside the downloaded DMG"
            case .stageFailed(let m): return "Could not stage the update: \(m)"
            case .signatureInvalid(let m):
                return "The downloaded update failed signature verification and was discarded: \(m)"
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

    nonisolated static let stagedAppName = UpdateInstaller.stagedAppName
    private static var swapScheduled = false

    private let dmgURL: URL
    private let version: String
    private let allowsMeteredNetwork: Bool
    private let onProgress: @MainActor (Progress) -> Void
    private let onError: @MainActor (UpdateError) -> Void
    private let onReady: @MainActor () -> Void

    private var task: Task<Void, Never>?
    private var session: URLSession?
    private var downloadTask: URLSessionTask?
    private var cancelled = false

    /// `allowsMeteredNetwork: false` fails the download on Low Data Mode and hotspot connections.
    init(
        dmgURL: URL,
        version: String,
        allowsMeteredNetwork: Bool,
        onProgress: @escaping @MainActor (Progress) -> Void,
        onError: @escaping @MainActor (UpdateError) -> Void,
        onReady: @escaping @MainActor () -> Void
    ) {
        self.dmgURL = dmgURL
        self.version = version
        self.allowsMeteredNetwork = allowsMeteredNetwork
        self.onProgress = onProgress
        self.onError = onError
        self.onReady = onReady
    }

    nonisolated static func appBundleURL() -> URL {
        Bundle.main.bundleURL
    }

    static func isWritable(_ app: URL) -> Bool {
        FileManager.default.isWritableFile(atPath: app.deletingLastPathComponent().path)
    }

    private nonisolated static var stagedAppURL: URL {
        appBundleURL().deletingLastPathComponent().appendingPathComponent(stagedAppName)
    }

    /// Call once on launch.
    /// With `keepPending`, keeps a staged bundle newer than the running app.
    /// Anything else is the old app left by a swap, or an older pending update.
    static func cleanupStaged(currentVersion: String, keepPending: Bool) {
        UpdateInstaller.cleanupLaunchAgents(
            in: AppConfig.appSupportURL().appendingPathComponent(
                UpdateInstaller.jobsDirectoryName
            )
        )
        if keepPending, let staged = stagedVersion(),
           ReleasesChecker.isNewer(staged, than: currentVersion) {
            return
        }
        Task { await discardStaged() }
    }

    nonisolated static func stagedVersion() -> String? {
        let plist = stagedAppURL.appendingPathComponent("Contents/Info.plist")
        return NSDictionary(contentsOf: plist)?["CFBundleShortVersionString"] as? String
    }

    /// True when the staged bundle is `version` and still passes the signature check.
    /// A copy cut short by a quit or crash fails the check.
    nonisolated static func stagedBundleMatches(version: String) async -> Bool {
        guard stagedVersion() == version else { return false }
        return await Task.detached(priority: .utility) {
            (try? verifyAppSignature(at: stagedAppURL.path)) != nil
        }.value
    }

    /// Removes the staged bundle off the main thread.
    /// The bundle has tens of thousands of files.
    nonisolated static func discardStaged() async {
        await Task.detached(priority: .utility) {
            try? FileManager.default.removeItem(at: stagedAppURL)
        }.value
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
        var temporaryDirectoryNeedsCleanup = true
        defer {
            if temporaryDirectoryNeedsCleanup {
                try? FileManager.default.removeItem(at: tmpDir)
            }
        }

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

        // hdiutil attach, ditto, and codesign block for seconds each.
        // A background download runs while the app is in use.
        let mountPoint: URL
        do {
            mountPoint = try await Task.detached(priority: .utility) {
                try Self.mountDMG(at: dmgPath)
            }.value
        } catch let err as UpdateError {
            onError(err); return
        } catch {
            onError(.mountFailed(error.localizedDescription)); return
        }

        var mountedDMGNeedsCleanup = true
        defer {
            if mountedDMGNeedsCleanup {
                try? Self.unmountDMG(at: mountPoint)
            }
        }

        if cancelled { return }
        onProgress(.staging)

        let stagedApp = app.deletingLastPathComponent().appendingPathComponent(Self.stagedAppName)
        do {
            try await Task.detached(priority: .utility) {
                do {
                    try Self.stageApp(fromMount: mountPoint, to: stagedApp)
                    try Self.verifyAppSignature(at: stagedApp.path)
                } catch {
                    try? FileManager.default.removeItem(at: stagedApp)
                    throw error
                }
            }.value
        } catch {
            onError(error as? UpdateError ?? .stageFailed(error.localizedDescription))
            return
        }

        if cancelled { return }
        do {
            try Self.finishStagedUpdate(
                detach: {
                    try Self.unmountDMG(at: mountPoint)
                    mountedDMGNeedsCleanup = false
                },
                removeTemporaryFiles: {
                    try FileManager.default.removeItem(at: tmpDir)
                    temporaryDirectoryNeedsCleanup = false
                },
                notifyReady: {
                    onProgress(.ready)
                    onReady()
                }
            )
        } catch let err as UpdateError {
            onError(err)
        } catch {
            onError(.cleanupFailed(error.localizedDescription))
        }
    }

    /// Releases every resource owned by a staged update before notifying the
    /// controller. `notifyReady` may synchronously terminate the process, so
    /// moving either cleanup step after it would leak the mounted image and
    /// its downloaded backing file.
    static func finishStagedUpdate(
        detach: () throws -> Void,
        removeTemporaryFiles: () throws -> Void,
        notifyReady: () -> Void
    ) throws {
        try detach()
        try removeTemporaryFiles()
        notifyReady()
    }

    // MARK: - Download

    func downloadDMG(
        to dest: URL,
        configuration config: URLSessionConfiguration = .default
    ) async throws {
        // Idle timeout only. Throttled CDN routes can need hours for the DMG.
        config.timeoutIntervalForRequest = 30
        config.allowsConstrainedNetworkAccess = allowsMeteredNetwork
        config.allowsExpensiveNetworkAccess = allowsMeteredNetwork
        let delegate = try DMGDownloadDelegate(destination: dest) { [weak self] pct, received, total in
            Task { @MainActor [weak self] in
                guard let self, !self.cancelled else { return }
                self.onProgress(.downloading(percent: pct, receivedBytes: received, totalBytes: total))
            }
        }
        let session = URLSession(configuration: config, delegate: delegate, delegateQueue: nil)
        self.session = session
        defer {
            session.invalidateAndCancel()
            self.session = nil
            self.downloadTask = nil
        }

        // A stalled transfer resumes from the bytes on disk with a new request.
        // URLSession resume data replays the signed CDN URL, which expires.
        var stalls = 0
        while true {
            let offset = delegate.received
            var request = URLRequest(url: dmgURL)
            if offset > 0 {
                request.setValue("bytes=\(offset)-", forHTTPHeaderField: "Range")
            }
            do {
                try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                    delegate.continuation = continuation
                    let task = session.dataTask(with: request)
                    self.downloadTask = task
                    task.resume()
                }
                break
            } catch let error as URLError where !cancelled {
                stalls = delegate.received > offset ? 0 : stalls + 1
                guard stalls <= 5 else { throw error }
                if stalls > 0 {
                    try await Task.sleep(for: .seconds(1 << stalls))
                }
            }
        }
        if cancelled {
            throw UpdateError.cancelled
        }
    }

    private final class DMGDownloadDelegate: NSObject, URLSessionDataDelegate {
        let file: FileHandle
        let onProgress: @Sendable (Int, Int64, Int64) -> Void
        var continuation: CheckedContinuation<Void, Error>?
        private(set) var received: Int64 = 0
        private var total: Int64 = 0
        private var etag: String?

        private let lock = NSLock()
        private var lastReportedPct = -1

        init(
            destination: URL,
            onProgress: @escaping @Sendable (Int, Int64, Int64) -> Void
        ) throws {
            FileManager.default.createFile(atPath: destination.path, contents: nil)
            file = try FileHandle(forWritingTo: destination)
            self.onProgress = onProgress
        }

        func urlSession(
            _ session: URLSession,
            dataTask: URLSessionDataTask,
            didReceive response: URLResponse,
            completionHandler: @escaping @Sendable (URLSession.ResponseDisposition) -> Void
        ) {
            let http = response as? HTTPURLResponse
            let etag = http?.value(forHTTPHeaderField: "ETag")
            // The CDN ignores If-Range, so a resumed body is checked by ETag.
            let continuesFile = received == 0
                ? http?.statusCode == 200
                : http?.statusCode == 206 && etag == self.etag
            guard continuesFile else {
                finish(.failure(UpdateError.downloadFailed("HTTP \(http?.statusCode ?? -1)")))
                completionHandler(.cancel)
                return
            }
            if received == 0 {
                total = response.expectedContentLength
                self.etag = etag
            }
            completionHandler(.allow)
        }

        func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
            do {
                try file.write(contentsOf: data)
            } catch {
                finish(.failure(UpdateError.downloadFailed(error.localizedDescription)))
                dataTask.cancel()
                return
            }
            received += Int64(data.count)
            guard total > 0 else { return }
            let pct = Int(received * 100 / total)
            lock.lock()
            let shouldReport = pct != lastReportedPct
            if shouldReport { lastReportedPct = pct }
            lock.unlock()
            if shouldReport {
                onProgress(pct, received, total)
            }
        }

        func urlSession(
            _ session: URLSession,
            task: URLSessionTask,
            didCompleteWithError error: Error?
        ) {
            if let error {
                finish(.failure(error))
            } else if received < total {
                finish(.failure(URLError(.networkConnectionLost)))
            } else {
                finish(.success(()))
            }
        }

        private func finish(_ result: Result<Void, Error>) {
            lock.lock()
            let continuation = self.continuation
            self.continuation = nil
            lock.unlock()

            switch result {
            case .success:
                continuation?.resume()
            case .failure(let error):
                continuation?.resume(throwing: error)
            }
        }
    }

    // MARK: - Mount / unmount

    private nonisolated static func mountDMG(at dmg: URL) throws -> URL {
        let result = try runProcess(
            "/usr/bin/hdiutil",
            args: ["attach", "-nobrowse", "-noautoopen", "-mountrandom", "/tmp", dmg.path]
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

    private nonisolated static func unmountDMG(at mountPoint: URL) throws {
        let result = try runProcess(
            "/usr/bin/hdiutil",
            args: ["detach", mountPoint.path, "-force"]
        )
        guard result.status == 0 else {
            throw UpdateError.unmountFailed(
                result.stderr.isEmpty ? result.stdout : result.stderr
            )
        }
    }

    // MARK: - Stage

    private nonisolated static func stageApp(fromMount mountPoint: URL, to stagedApp: URL) throws {
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

    private nonisolated static func findAppInVolume(_ mountPoint: URL) throws -> URL {
        let preferred = mountPoint.appendingPathComponent("oMLX.app")
        if FileManager.default.fileExists(atPath: preferred.path) { return preferred }
        let entries = (try? FileManager.default.contentsOfDirectory(atPath: mountPoint.path)) ?? []
        for name in entries where name.hasSuffix(".app") {
            return mountPoint.appendingPathComponent(name)
        }
        throw UpdateError.appNotFoundInVolume
    }

    // MARK: - Signature verification

    /// A plain `--verify` also accepts ad-hoc or foreign signatures, so the
    /// check pins the Developer ID team that signs releases.
    nonisolated static let releaseRequirement =
        #"=anchor apple generic and identifier "app.omlx" and certificate leaf[subject.OU] = "PSK5Q5T46L""#

    nonisolated static func verifyAppSignature(at appPath: String) throws {
        let result = try runProcess(
            "/usr/bin/codesign",
            args: ["--verify", "--deep", "--strict", "-R", Self.releaseRequirement, appPath]
        )
        guard result.status == 0 else {
            let detail = result.stderr.isEmpty ? result.stdout : result.stderr
            NSLog("oMLX updater: signature verification failed: %@", detail)
            throw UpdateError.signatureInvalid(
                detail.isEmpty ? "codesign exited \(result.status)" : detail
            )
        }
    }

    // MARK: - Swap + relaunch (called from outside, right before terminate)

    /// Registers a one-shot launchd worker that:
    ///   1. waits for our PID to exit
    ///   2. replaces the running .app with the staged one
    ///   3. strips com.apple.quarantine
    ///   4. `open`s the replaced .app
    /// Must be called immediately before `NSApp.terminate(nil)`.
    @discardableResult
    static func performSwapAndRelaunch() -> Bool {
        if swapScheduled { return true }

        let app = appBundleURL()
        let staged = app.deletingLastPathComponent().appendingPathComponent(stagedAppName)
        guard FileManager.default.fileExists(atPath: staged.path) else { return false }

        guard let executable = Bundle.main.executableURL else {
            NSLog("oMLX: failed to locate executable for updater worker")
            return false
        }
        do {
            try UpdateInstaller.submitWorker(
                parentPID: ProcessInfo.processInfo.processIdentifier,
                liveApp: app,
                stagedApp: staged,
                executable: executable,
                jobsDirectory: AppConfig.appSupportURL().appendingPathComponent(
                    UpdateInstaller.jobsDirectoryName
                )
            )
            swapScheduled = true
        } catch {
            NSLog("oMLX: failed to start updater worker: %@", error.localizedDescription)
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

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
// A download that stops is resumed from the bytes already on disk, a bounded
// number of times without progress. There is no whole-transfer deadline: the
// idle timeout is the stall detector.
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

    /// Consecutive attempts that may end without new bytes before the update
    /// is reported as failed. An attempt that made progress resets the count.
    nonisolated static let maxDownloadResumes = 5

    /// The release CDN throttles some routes to a few dozen KB/s per
    /// connection, so an 830 MB DMG can take hours. The request timeout is an
    /// idle timeout (its timer resets whenever bytes arrive) and is the stall
    /// detector. The resource timeout is only a backstop: a one-hour deadline
    /// here killed transfers that were still receiving bytes (#4199).
    nonisolated static func downloadSessionConfiguration(
        allowsMeteredNetwork: Bool
    ) -> URLSessionConfiguration {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 7 * 24 * 60 * 60
        config.urlCache = nil
        config.allowsConstrainedNetworkAccess = allowsMeteredNetwork
        config.allowsExpensiveNetworkAccess = allowsMeteredNetwork
        return config
    }

    /// `bytes <first>-<last>/<total>` from a 206; `total` is nil for `*`.
    nonisolated static func parseContentRange(_ header: String?) -> (first: Int64, total: Int64?)? {
        guard let header, header.hasPrefix("bytes ") else { return nil }
        let parts = header.dropFirst(6).split(separator: "/", maxSplits: 1)
        guard parts.count == 2,
              let dash = parts[0].firstIndex(of: "-"),
              let first = Int64(parts[0][..<dash])
        else { return nil }
        return (first, Int64(parts[1]))
    }

    /// A transfer that stopped after `received` bytes were written to disk.
    /// The next attempt continues from there while the asset still has `etag`.
    struct DownloadInterrupted: Error {
        let received: Int64
        let etag: String?
        let underlying: Error
    }

    /// Writes the DMG to `dest`. A transfer that stops is picked up from the
    /// bytes already on disk with a fresh request to the release URL and a
    /// `Range` header. The redirect target behind that URL is signed and
    /// expires, so URLSession's resume data (which replays the signed URL)
    /// would stop working after the first hour. A 200 or a changed ETag on a
    /// resumed request starts the file over. The pause before a resume starts
    /// at `resumeDelay` and doubles while attempts make no progress, so a
    /// network that is re-associating gets about a minute.
    func downloadDMG(
        to dest: URL,
        configuration: URLSessionConfiguration? = nil,
        resumeDelay: Duration = .seconds(2)
    ) async throws {
        let config = configuration
            ?? Self.downloadSessionConfiguration(allowsMeteredNetwork: allowsMeteredNetwork)
        var received: Int64 = 0
        var etag: String?
        var attemptsWithoutProgress = 0
        while true {
            if cancelled { throw UpdateError.cancelled }
            do {
                try await downloadAttempt(to: dest, configuration: config, offset: received, etag: etag)
                return
            } catch let interrupted as DownloadInterrupted {
                if cancelled { throw UpdateError.cancelled }
                attemptsWithoutProgress =
                    interrupted.received > received ? 0 : attemptsWithoutProgress + 1
                guard attemptsWithoutProgress <= Self.maxDownloadResumes else {
                    throw interrupted.underlying
                }
                received = interrupted.received
                etag = interrupted.etag
                NSLog(
                    "oMLX updater: download stopped after %lld bytes (%@), resuming",
                    received, interrupted.underlying.localizedDescription
                )
                do {
                    try await Task.sleep(for: resumeDelay * (1 << max(attemptsWithoutProgress - 1, 0)))
                } catch {
                    throw UpdateError.cancelled
                }
            }
        }
    }

    private func downloadAttempt(
        to dest: URL,
        configuration: URLSessionConfiguration,
        offset: Int64,
        etag: String?
    ) async throws {
        var request = URLRequest(url: dmgURL)
        request.cachePolicy = .reloadIgnoringLocalCacheData
        if offset > 0 {
            request.setValue("bytes=\(offset)-", forHTTPHeaderField: "Range")
            if let etag {
                request.setValue(etag, forHTTPHeaderField: "If-Range")
            }
        }

        let delegate: DMGDownloadDelegate
        do {
            delegate = try DMGDownloadDelegate(destination: dest, offset: offset, etag: etag) {
                [weak self] pct, received, total in
                Task { @MainActor [weak self] in
                    guard let self, !self.cancelled else { return }
                    self.onProgress(.downloading(percent: pct, receivedBytes: received, totalBytes: total))
                }
            }
        } catch {
            throw UpdateError.downloadFailed(
                "Could not open \(dest.lastPathComponent): \(error.localizedDescription)"
            )
        }
        let session = URLSession(configuration: configuration, delegate: delegate, delegateQueue: nil)
        self.session = session
        defer {
            session.invalidateAndCancel()
            self.session = nil
            self.downloadTask = nil
        }

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            delegate.begin(continuation)
            let task = session.dataTask(with: request)
            self.downloadTask = task
            task.resume()
        }
    }

    /// Streams the response body into the destination file. Callbacks arrive
    /// on the session's serial delegate queue; `lock` guards every stored
    /// property, and `begin` runs before the task is resumed.
    private final class DMGDownloadDelegate: NSObject, URLSessionDataDelegate, @unchecked Sendable {
        private let lock = NSLock()
        private let file: FileHandle
        private let resumingFrom: Int64
        private let onProgress: @Sendable (Int, Int64, Int64) -> Void
        private var continuation: CheckedContinuation<Void, Error>?
        private var received: Int64
        private var total: Int64?
        private var etag: String?
        private var lastReportedPct = -1
        private var completed = false

        init(
            destination: URL,
            offset: Int64,
            etag: String?,
            onProgress: @escaping @Sendable (Int, Int64, Int64) -> Void
        ) throws {
            if offset == 0 || !FileManager.default.fileExists(atPath: destination.path) {
                FileManager.default.createFile(atPath: destination.path, contents: nil)
            }
            file = try FileHandle(forWritingTo: destination)
            try file.truncate(atOffset: UInt64(offset))
            try file.seek(toOffset: UInt64(offset))
            resumingFrom = offset
            received = offset
            self.etag = etag
            self.onProgress = onProgress
        }

        func begin(_ continuation: CheckedContinuation<Void, Error>) {
            lock.lock()
            self.continuation = continuation
            lock.unlock()
        }

        func urlSession(
            _ session: URLSession,
            dataTask: URLSessionDataTask,
            didReceive response: URLResponse,
            completionHandler: @escaping @Sendable (URLSession.ResponseDisposition) -> Void
        ) {
            guard let http = response as? HTTPURLResponse else {
                finish(.failure(UpdateError.downloadFailed("Not an HTTP response")))
                completionHandler(.cancel)
                return
            }
            let responseETag = http.value(forHTTPHeaderField: "ETag")
            lock.lock()
            let expectedETag = etag
            lock.unlock()

            switch http.statusCode {
            case 206 where resumingFrom > 0:
                guard let range = AppUpdater.parseContentRange(http.value(forHTTPHeaderField: "Content-Range")),
                      range.first == resumingFrom,
                      expectedETag == nil || responseETag == expectedETag
                else {
                    // Not the continuation of our file (the asset changed and
                    // the server ignored If-Range, or it picked another
                    // range): drop this body and start over on the next attempt.
                    let changed = UpdateError.downloadFailed("Asset changed while resuming at byte \(resumingFrom)")
                    finish(.failure(DownloadInterrupted(received: 0, etag: nil, underlying: changed)))
                    completionHandler(.cancel)
                    return
                }
                lock.lock()
                total = range.total
                lock.unlock()
            case 200:
                // The whole asset: first attempt, or the server did not honour
                // the range, or the asset changed. Start the file over.
                do {
                    try restartFile()
                } catch {
                    finish(.failure(UpdateError.downloadFailed(error.localizedDescription)))
                    completionHandler(.cancel)
                    return
                }
                lock.lock()
                total = http.expectedContentLength > 0 ? http.expectedContentLength : nil
                etag = responseETag
                lock.unlock()
            default:
                finish(.failure(UpdateError.downloadFailed("HTTP \(http.statusCode)")))
                completionHandler(.cancel)
                return
            }
            completionHandler(.allow)
        }

        func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
            lock.lock()
            do {
                try file.write(contentsOf: data)
            } catch {
                lock.unlock()
                finish(.failure(UpdateError.downloadFailed(error.localizedDescription)))
                dataTask.cancel()
                return
            }
            received += Int64(data.count)
            var report: (Int, Int64, Int64)?
            if let total, total > 0 {
                let pct = Int(received * 100 / total)
                if pct != lastReportedPct {
                    lastReportedPct = pct
                    report = (pct, received, total)
                }
            }
            lock.unlock()
            if let report {
                onProgress(report.0, report.1, report.2)
            }
        }

        func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
            lock.lock()
            let received = self.received
            let total = self.total
            let etag = self.etag
            lock.unlock()
            if let error {
                finish(.failure(DownloadInterrupted(received: received, etag: etag, underlying: error)))
            } else if let total, received != total {
                // A clean end before the announced length: resume from here.
                let short = UpdateError.downloadFailed("Received \(received) of \(total) bytes")
                finish(.failure(DownloadInterrupted(received: received, etag: etag, underlying: short)))
            } else {
                finish(.success(()))
            }
        }

        private func restartFile() throws {
            lock.lock()
            defer { lock.unlock() }
            try file.truncate(atOffset: 0)
            try file.seek(toOffset: 0)
            received = 0
            lastReportedPct = -1
        }

        private func finish(_ result: Result<Void, Error>) {
            lock.lock()
            guard !completed else {
                lock.unlock()
                return
            }
            completed = true
            let continuation = self.continuation
            self.continuation = nil
            try? file.close()
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

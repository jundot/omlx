// In-place auto-updater.
//
// Flow: download .dmg → Gatekeeper-assess the notarized image → mount with
// image verification enabled → validate the inner app's stapled ticket and
// Developer ID requirement against the running app's team → copy it next to
// the running bundle as `.oMLX-update.app` → validate the copy again → on
// confirmation, atomically swap and relaunch. The downloaded payload never
// becomes executable merely because it came from a GitHub release URL.
//
// Cancellation: `cancel()` is best-effort; an in-flight download exits at
// the next stream chunk. A staged copy that's already on disk gets
// cleaned up by `cleanupStaged()` on the next launch.

import AppKit
import Foundation
import Security

@MainActor
final class AppUpdater {
    enum UpdateError: Error, CustomStringConvertible {
        case notWritable(String)
        case downloadFailed(String)
        case mountFailed(String)
        case appNotFoundInVolume
        case verificationFailed(String)
        case stageFailed(String)
        case cancelled

        var description: String {
            switch self {
            case .notWritable(let path):
                return "Cannot write to \(path). Move oMLX.app to a writable location and try again."
            case .downloadFailed(let m): return "Download failed: \(m)"
            case .mountFailed(let m): return "Could not mount DMG: \(m)"
            case .appNotFoundInVolume: return "oMLX.app not found inside the downloaded DMG"
            case .verificationFailed(let m): return "Update verification failed: \(m)"
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

    static let stagedAppName = UpdateInstaller.stagedAppName
    private static var swapScheduled = false

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
        UpdateInstaller.cleanupLaunchAgents(
            in: AppConfig.appSupportURL().appendingPathComponent(
                UpdateInstaller.jobsDirectoryName
            )
        )
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
            try Self.validateDiskImage(dmgPath)
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
            let identity = try Self.expectedUpdateIdentity()
            let appInVolume = try findAppInVolume(mountPoint)
            try Self.validateApplication(appInVolume, expected: identity)
            try stageApp(appInVolume, to: stagedApp)
            try Self.validateApplication(stagedApp, expected: identity)
        } catch let err as UpdateError {
            try? FileManager.default.removeItem(at: stagedApp)
            onError(err); return
        } catch {
            try? FileManager.default.removeItem(at: stagedApp)
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
        let delegate = DMGDownloadDelegate(destination: dest) { [weak self] pct, received, total in
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

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            delegate.continuation = continuation
            let task = session.downloadTask(with: dmgURL)
            self.downloadTask = task
            task.resume()
        }
        if cancelled {
            throw UpdateError.cancelled
        }
    }

    private final class DMGDownloadDelegate: NSObject, URLSessionDownloadDelegate {
        let destination: URL
        let onProgress: @Sendable (Int, Int64, Int64) -> Void
        var continuation: CheckedContinuation<Void, Error>?

        private let lock = NSLock()
        private var completed = false
        private var lastReportedPct = -1

        init(
            destination: URL,
            onProgress: @escaping @Sendable (Int, Int64, Int64) -> Void
        ) {
            self.destination = destination
            self.onProgress = onProgress
        }

        func urlSession(
            _ session: URLSession,
            downloadTask: URLSessionDownloadTask,
            didWriteData bytesWritten: Int64,
            totalBytesWritten: Int64,
            totalBytesExpectedToWrite: Int64
        ) {
            guard totalBytesExpectedToWrite > 0 else { return }
            let pct = Int(totalBytesWritten * 100 / totalBytesExpectedToWrite)
            lock.lock()
            let shouldReport = pct != lastReportedPct
            if shouldReport { lastReportedPct = pct }
            lock.unlock()
            if shouldReport {
                onProgress(pct, totalBytesWritten, totalBytesExpectedToWrite)
            }
        }

        func urlSession(
            _ session: URLSession,
            downloadTask: URLSessionDownloadTask,
            didFinishDownloadingTo location: URL
        ) {
            guard let http = downloadTask.response as? HTTPURLResponse,
                  http.statusCode == 200
            else {
                let code = (downloadTask.response as? HTTPURLResponse)?.statusCode ?? -1
                finish(.failure(UpdateError.downloadFailed("HTTP \(code)")))
                return
            }
            do {
                if FileManager.default.fileExists(atPath: destination.path) {
                    try FileManager.default.removeItem(at: destination)
                }
                try FileManager.default.moveItem(at: location, to: destination)
                finish(.success(()))
            } catch {
                finish(.failure(UpdateError.downloadFailed(error.localizedDescription)))
            }
        }

        func urlSession(
            _ session: URLSession,
            task: URLSessionTask,
            didCompleteWithError error: Error?
        ) {
            if let error {
                finish(.failure(error))
            }
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

    nonisolated static func diskImageAssessmentArguments(_ dmg: URL) -> [String] {
        [
            "--assess", "--type", "open", "--verbose=2",
            "--context", "context:primary-signature", dmg.path,
        ]
    }

    nonisolated static func mountArguments(_ dmg: URL) -> [String] {
        ["attach", "-readonly", "-nobrowse", "-noautoopen", "-mountrandom", "/tmp", dmg.path]
    }

    private static func validateDiskImage(_ dmg: URL) throws {
        let result = try runProcess(
            "/usr/sbin/spctl",
            args: diskImageAssessmentArguments(dmg)
        )
        guard result.status == 0 else {
            let detail = result.stderr.isEmpty ? result.stdout : result.stderr
            throw UpdateError.verificationFailed(
                detail.isEmpty ? "Gatekeeper rejected the disk image" : detail
            )
        }
    }

    private func mountDMG(at dmg: URL) throws -> URL {
        let result = try runProcess(
            "/usr/bin/hdiutil",
            args: Self.mountArguments(dmg)
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

    private func stageApp(_ appInVolume: URL, to stagedApp: URL) throws {
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

    struct UpdateIdentity: Equatable {
        let bundleIdentifier: String
        let teamIdentifier: String
    }

    nonisolated static func developerIDRequirement(for identity: UpdateIdentity) throws -> String {
        let bundleAllowed = CharacterSet.alphanumerics.union(
            CharacterSet(charactersIn: ".-")
        )
        guard !identity.bundleIdentifier.isEmpty,
              identity.bundleIdentifier.unicodeScalars.allSatisfy({ bundleAllowed.contains($0) })
        else {
            throw UpdateError.verificationFailed("invalid bundle identifier in trust policy")
        }

        let teamAllowed = CharacterSet(charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        guard identity.teamIdentifier.count == 10,
              identity.teamIdentifier.unicodeScalars.allSatisfy({ teamAllowed.contains($0) })
        else {
            throw UpdateError.verificationFailed("invalid Developer Team ID in trust policy")
        }

        return "anchor apple generic"
            + " and identifier \"\(identity.bundleIdentifier)\""
            + " and certificate 1[field.1.2.840.113635.100.6.2.6] exists"
            + " and certificate leaf[field.1.2.840.113635.100.6.1.13] exists"
            + " and certificate leaf[subject.OU] = \"\(identity.teamIdentifier)\""
    }

    private static func expectedUpdateIdentity() throws -> UpdateIdentity {
        guard let bundleIdentifier = Bundle.main.bundleIdentifier,
              !bundleIdentifier.isEmpty
        else {
            throw UpdateError.verificationFailed("running app has no bundle identifier")
        }

        var dynamicCode: SecCode?
        var status = SecCodeCopySelf(SecCSFlags(), &dynamicCode)
        guard status == errSecSuccess, let dynamicCode else {
            throw securityError("could not inspect the running app signature", status)
        }

        var staticCode: SecStaticCode?
        status = SecCodeCopyStaticCode(dynamicCode, SecCSFlags(), &staticCode)
        guard status == errSecSuccess, let staticCode else {
            throw securityError("could not inspect the running app on disk", status)
        }

        var information: CFDictionary?
        status = SecCodeCopySigningInformation(
            staticCode,
            SecCSFlags(rawValue: kSecCSSigningInformation),
            &information
        )
        guard status == errSecSuccess,
              let dictionary = information as? [CFString: Any],
              let teamIdentifier = dictionary[kSecCodeInfoTeamIdentifier] as? String,
              !teamIdentifier.isEmpty
        else {
            if status == errSecSuccess {
                throw UpdateError.verificationFailed(
                    "running app is not signed with a Developer ID team"
                )
            }
            throw securityError("could not read the running app signing team", status)
        }
        return UpdateIdentity(
            bundleIdentifier: bundleIdentifier,
            teamIdentifier: teamIdentifier
        )
    }

    private static func validateApplication(
        _ application: URL,
        expected identity: UpdateIdentity
    ) throws {
        let policy = try developerIDRequirement(for: identity)
        var requirement: SecRequirement?
        var status = SecRequirementCreateWithString(
            policy as CFString,
            SecCSFlags(),
            &requirement
        )
        guard status == errSecSuccess, let requirement else {
            throw securityError("could not construct the update trust policy", status)
        }

        var staticCode: SecStaticCode?
        status = SecStaticCodeCreateWithPath(
            application as CFURL,
            SecCSFlags(),
            &staticCode
        )
        guard status == errSecSuccess, let staticCode else {
            throw securityError("downloaded app has no readable code signature", status)
        }

        let validationFlags = SecCSFlags(
            rawValue: kSecCSCheckNestedCode
                | kSecCSStrictValidate
                | kSecCSCheckAllArchitectures
                | kSecCSRestrictSymlinks
        )
        status = SecStaticCodeCheckValidity(staticCode, validationFlags, requirement)
        guard status == errSecSuccess else {
            throw securityError(
                "downloaded app is not valid Developer ID code from the expected team",
                status
            )
        }

        let systemPolicy = try runProcess(
            "/usr/bin/syspolicy_check",
            args: ["distribution", application.path]
        )
        guard systemPolicy.status == 0 else {
            let detail = systemPolicy.stderr.isEmpty
                ? systemPolicy.stdout
                : systemPolicy.stderr
            throw UpdateError.verificationFailed(
                detail.isEmpty
                    ? "downloaded app failed Apple's distribution policy check"
                    : detail
            )
        }
    }

    private static func securityError(_ message: String, _ status: OSStatus) -> UpdateError {
        let detail = SecCopyErrorMessageString(status, nil) as String? ?? "OSStatus \(status)"
        return .verificationFailed("\(message): \(detail)")
    }

    // MARK: - Swap + relaunch (called from outside, right before terminate)

    /// Registers a one-shot launchd worker that:
    ///   1. waits for our PID to exit
    ///   2. replaces the running .app with the staged one
    ///   3. `open`s the replaced .app (with quarantine metadata preserved)
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

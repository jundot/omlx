// Crash-safe bundle installer used by the in-app updater.
//
// The main app registers its own executable as a one-shot launchd job before
// terminating. The worker mode waits for the app process to exit, atomically
// exchanges the live and staged bundles, and relaunches the live path. launchd
// gives the worker its own job and process coalition, so app teardown cannot
// reap it as a process-scoped child.

import Darwin
import Foundation

enum UpdateInstaller {
    static let workerModeArgument = "--omlx-update-worker"
    static let stagedAppName = ".oMLX-update.app"
    static let jobsDirectoryName = "updater-jobs"

    struct WorkerRequest: Equatable {
        let parentPID: pid_t
        let liveApp: URL
        let stagedApp: URL
    }

    enum InstallerError: LocalizedError {
        case invalidWorkerArguments
        case invalidParentPID(String)
        case invalidBundlePaths(String)
        case parentExitTimedOut(pid_t)
        case atomicSwapFailed(Int32, String)
        case launchAgentFailed(String)
        case relaunchFailed(String)

        var errorDescription: String? {
            switch self {
            case .invalidWorkerArguments:
                return "Invalid updater worker arguments"
            case .invalidParentPID(let value):
                return "Invalid updater parent PID: \(value)"
            case .invalidBundlePaths(let message):
                return "Invalid updater bundle paths: \(message)"
            case .parentExitTimedOut(let pid):
                return "Timed out waiting for oMLX process \(pid) to exit"
            case .atomicSwapFailed(let code, let message):
                return "Atomic app swap failed with errno \(code): \(message)"
            case .launchAgentFailed(let message):
                return "Could not start the updater launch agent: \(message)"
            case .relaunchFailed(let message):
                return "Could not relaunch oMLX: \(message)"
            }
        }
    }

    private struct ProcessResult {
        let status: Int32
        let stdout: String
        let stderr: String
    }

    /// Parses the private worker invocation while leaving normal app launches
    /// untouched. A malformed worker invocation fails closed instead of
    /// accidentally starting the full app.
    static func workerRequest(from arguments: [String]) throws -> WorkerRequest? {
        guard arguments.dropFirst().first == workerModeArgument else {
            return nil
        }
        guard arguments.count == 5 else {
            throw InstallerError.invalidWorkerArguments
        }
        guard let parentPID = pid_t(arguments[2]), parentPID > 1 else {
            throw InstallerError.invalidParentPID(arguments[2])
        }

        let request = WorkerRequest(
            parentPID: parentPID,
            liveApp: URL(fileURLWithPath: arguments[3]).standardizedFileURL,
            stagedApp: URL(fileURLWithPath: arguments[4]).standardizedFileURL
        )
        try validateBundlePaths(liveApp: request.liveApp, stagedApp: request.stagedApp)
        return request
    }

    /// Registers a non-keepalive LaunchAgent. Unlike `launchctl submit`, a
    /// plist with `KeepAlive = false` runs exactly once even when the worker
    /// exits before launchd's ten-second minimum-runtime threshold.
    static func submitWorker(
        parentPID: pid_t,
        liveApp: URL,
        stagedApp: URL,
        executable: URL,
        jobsDirectory: URL
    ) throws {
        let liveApp = liveApp.standardizedFileURL
        let stagedApp = stagedApp.standardizedFileURL
        try validateBundlePaths(liveApp: liveApp, stagedApp: stagedApp)

        let fileManager = FileManager.default
        try fileManager.createDirectory(
            at: jobsDirectory,
            withIntermediateDirectories: true
        )

        let identifier = UUID().uuidString.lowercased()
        let label = "app.omlx.updater.\(identifier)"
        let plistURL = jobsDirectory.appendingPathComponent("\(label).plist")
        let plist = launchAgentPropertyList(
            label: label,
            executable: executable,
            parentPID: parentPID,
            liveApp: liveApp,
            stagedApp: stagedApp
        )
        let data = try PropertyListSerialization.data(
            fromPropertyList: plist,
            format: .xml,
            options: 0
        )
        try data.write(to: plistURL, options: .atomic)

        do {
            let result = try runProcess(
                "/bin/launchctl",
                arguments: ["bootstrap", launchDomain, plistURL.path]
            )
            guard result.status == 0 else {
                throw InstallerError.launchAgentFailed(
                    result.stderr.isEmpty ? result.stdout : result.stderr
                )
            }
        } catch {
            try? fileManager.removeItem(at: plistURL)
            throw error
        }
    }

    static func launchAgentPropertyList(
        label: String,
        executable: URL,
        parentPID: pid_t,
        liveApp: URL,
        stagedApp: URL
    ) -> [String: Any] {
        [
            "Label": label,
            "ProgramArguments": [
                executable.path,
                workerModeArgument,
                String(parentPID),
                liveApp.path,
                stagedApp.path,
            ],
            "RunAtLoad": true,
            "KeepAlive": false,
            "ProcessType": "Background",
        ]
    }

    /// Removes inactive jobs from earlier updater runs. This is called only
    /// after an app has launched successfully, so stopping a still-finishing
    /// worker cannot make the live bundle unavailable.
    static func cleanupLaunchAgents(in jobsDirectory: URL) {
        let fileManager = FileManager.default
        guard let entries = try? fileManager.contentsOfDirectory(
            at: jobsDirectory,
            includingPropertiesForKeys: nil
        ) else {
            return
        }

        for plistURL in entries where plistURL.pathExtension == "plist" {
            if let data = try? Data(contentsOf: plistURL),
               let plist = try? PropertyListSerialization.propertyList(
                   from: data,
                   options: [],
                   format: nil
               ) as? [String: Any],
               let label = plist["Label"] as? String,
               label.hasPrefix("app.omlx.updater.") {
                _ = try? runProcess(
                    "/bin/launchctl",
                    arguments: ["bootout", "\(launchDomain)/\(label)"]
                )
            }
            try? fileManager.removeItem(at: plistURL)
        }

        if (try? fileManager.contentsOfDirectory(
            atPath: jobsDirectory.path
        ).isEmpty) == true {
            try? fileManager.removeItem(at: jobsDirectory)
        }
    }

    static func runWorker(_ request: WorkerRequest) -> Int32 {
        do {
            guard waitForProcessExit(request.parentPID, timeout: 60) else {
                throw InstallerError.parentExitTimedOut(request.parentPID)
            }

            try atomicSwap(
                liveApp: request.liveApp,
                stagedApp: request.stagedApp
            )
            clearQuarantine(at: request.liveApp)
            try relaunch(request.liveApp)
            return EXIT_SUCCESS
        } catch {
            NSLog("oMLX updater: %@", error.localizedDescription)
            try? relaunch(request.liveApp)
            return EXIT_FAILURE
        }
    }

    static func waitForProcessExit(
        _ pid: pid_t,
        timeout: TimeInterval
    ) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            errno = 0
            if kill(pid, 0) == -1, errno == ESRCH {
                return true
            }
            usleep(100_000)
        }
        return false
    }

    /// `RENAME_SWAP` is a single filesystem transaction. Before the call the
    /// old app is live; after it the new app is live and the old app occupies
    /// the staged path. There is no partially deleted bundle or missing-path
    /// interval, even if the worker is killed or the machine loses power.
    static func atomicSwap(liveApp: URL, stagedApp: URL) throws {
        let liveApp = liveApp.standardizedFileURL
        let stagedApp = stagedApp.standardizedFileURL
        try validateBundlePaths(liveApp: liveApp, stagedApp: stagedApp)

        errno = 0
        let result = liveApp.path.withCString { livePath in
            stagedApp.path.withCString { stagedPath in
                renamex_np(livePath, stagedPath, UInt32(RENAME_SWAP))
            }
        }
        guard result == 0 else {
            let code = errno
            throw InstallerError.atomicSwapFailed(
                code,
                String(cString: strerror(code))
            )
        }
    }

    private static var launchDomain: String {
        "gui/\(getuid())"
    }

    private static func validateBundlePaths(
        liveApp: URL,
        stagedApp: URL
    ) throws {
        guard liveApp.isFileURL, stagedApp.isFileURL else {
            throw InstallerError.invalidBundlePaths("paths must be local files")
        }
        guard liveApp.deletingLastPathComponent().standardizedFileURL
                == stagedApp.deletingLastPathComponent().standardizedFileURL else {
            throw InstallerError.invalidBundlePaths("bundles must share a parent directory")
        }
        guard stagedApp.lastPathComponent == stagedAppName else {
            throw InstallerError.invalidBundlePaths(
                "staged bundle must be named \(stagedAppName)"
            )
        }
        guard liveApp.pathExtension == "app" else {
            throw InstallerError.invalidBundlePaths("live bundle must be an app")
        }
    }

    private static func clearQuarantine(at app: URL) {
        guard let result = try? runProcess(
            "/usr/bin/xattr",
            arguments: ["-rd", "com.apple.quarantine", app.path]
        ), result.status != 0 else {
            return
        }
        NSLog(
            "oMLX updater: could not clear quarantine: %@",
            result.stderr.isEmpty ? result.stdout : result.stderr
        )
    }

    private static func relaunch(_ app: URL) throws {
        let result = try runProcess("/usr/bin/open", arguments: [app.path])
        guard result.status == 0 else {
            throw InstallerError.relaunchFailed(
                result.stderr.isEmpty ? result.stdout : result.stderr
            )
        }
    }

    private static func runProcess(
        _ executable: String,
        arguments: [String]
    ) throws -> ProcessResult {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe
        try process.run()
        process.waitUntilExit()
        let stdout = String(
            data: stdoutPipe.fileHandleForReading.readDataToEndOfFile(),
            encoding: .utf8
        ) ?? ""
        let stderr = String(
            data: stderrPipe.fileHandleForReading.readDataToEndOfFile(),
            encoding: .utf8
        ) ?? ""
        return ProcessResult(
            status: process.terminationStatus,
            stdout: stdout,
            stderr: stderr
        )
    }
}

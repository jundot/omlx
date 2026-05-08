// PR 2 — spawn-only wrapper around the omlx serve child process.
//
// Today: start(), terminate(), pid, isRunning. No health check, no auto-restart,
// no port-conflict handling — those land in PR 5 and replace the simple
// state machine here with a Combine-published lifecycle.
//
// Spawn semantics mirror packaging/omlx_app/server_manager.py:300-355:
//   <python> -m omlx.cli serve --base-path <base> --port <port>
//   stdout+stderr → ~/Library/Application Support/oMLX-next/logs/server.log
//   PATH = parent + Homebrew prefixes
//
// Dev override: set OMLX_DEV_SERVER_SCRIPT=<path> and we spawn
//   <python> <script> --port <port>
// instead, so the spawn path can be exercised without the full omlx stack
// (workspace .venv is currently mlx-lm-stale; see plan.md PR 2 verification).

import Foundation

// @unchecked Sendable: all state mutations either happen at construction
// time on the main thread, or in `handleTermination` which DispatchQueue.main
// .async's back. The Process terminationHandler is invoked on an arbitrary
// thread but only reads `process` to query terminationStatus / -Reason.
final class ServerProcess: @unchecked Sendable {
    enum State: Equatable {
        case idle
        case starting
        case running(pid: Int32)
        case stopped
        case failed(message: String)
    }

    /// Posted on `NotificationCenter.default` when `state` changes; the
    /// notification's `object` is the ServerProcess. Observers receive on
    /// the main queue (we dispatch via DispatchQueue.main inside
    /// `handleTermination`); use addObserver(_:selector:name:object:) for
    /// concurrency-safe target-action observation.
    static let stateDidChangeNotification = Notification.Name("OMLXServerProcessStateDidChange")

    private let runtime: PythonRuntime
    private(set) var state: State = .idle

    let host: String
    let port: Int
    let basePath: URL

    private var process: Process?
    private var logHandle: FileHandle?
    private let logURL: URL

    init(
        runtime: PythonRuntime,
        host: String = "127.0.0.1",
        port: Int = 8080,
        basePath: URL = ServerProcess.defaultBasePath()
    ) {
        self.runtime = runtime
        self.host = host
        self.port = port
        self.basePath = basePath
        self.logURL = ServerProcess.defaultLogURL()
    }

    var isRunning: Bool { process?.isRunning ?? false }
    var pid: Int32? { process?.processIdentifier }

    /// Start the child server. Returns immediately; readiness is verified by
    /// PR 5's health-check loop. Throws if the runtime can't be located or the
    /// spawn syscall fails.
    func start() throws {
        if isRunning { return }

        try ensureDirectory(basePath)
        try ensureDirectory(logURL.deletingLastPathComponent())

        // Open the log file in append mode so multiple sessions accumulate.
        if !FileManager.default.fileExists(atPath: logURL.path) {
            FileManager.default.createFile(atPath: logURL.path, contents: nil)
        }
        let handle = try FileHandle(forWritingTo: logURL)
        try handle.seekToEnd()
        logHandle = handle

        let proc = Process()
        proc.executableURL = runtime.executable
        proc.arguments = makeArguments()
        proc.environment = runtime.makeEnvironment()
        proc.standardOutput = handle
        proc.standardError = handle

        proc.terminationHandler = { [weak self] terminated in
            self?.handleTermination(terminated)
        }

        update(.starting)
        do {
            try proc.run()
        } catch {
            cleanup()
            update(.failed(message: "spawn failed: \(error.localizedDescription)"))
            throw error
        }
        process = proc
        update(.running(pid: proc.processIdentifier))
    }

    /// Send SIGTERM. Full SIGTERM→wait→SIGKILL chain lands in PR 5.
    func terminate() {
        guard let proc = process, proc.isRunning else { return }
        proc.terminate()
    }

    // MARK: - Private

    private func makeArguments() -> [String] {
        let env = ProcessInfo.processInfo.environment
        if let dev = env["OMLX_DEV_SERVER_SCRIPT"], !dev.isEmpty {
            return [dev, "--host", host, "--port", String(port)]
        }
        return [
            "-m", "omlx.cli", "serve",
            "--base-path", basePath.path,
            "--port", String(port),
        ]
    }

    private func update(_ next: State) {
        state = next
        NotificationCenter.default.post(name: Self.stateDidChangeNotification, object: self)
    }

    private func handleTermination(_ terminated: Process) {
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            let status = terminated.terminationStatus
            self.cleanup()
            if status == 0 || terminated.terminationReason == .uncaughtSignal {
                self.update(.stopped)
            } else {
                self.update(.failed(message: "exited with code \(status)"))
            }
        }
    }

    private func cleanup() {
        process = nil
        try? logHandle?.close()
        logHandle = nil
    }

    private func ensureDirectory(_ url: URL) throws {
        try FileManager.default.createDirectory(
            at: url, withIntermediateDirectories: true
        )
    }

    static func defaultBasePath() -> URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".omlx", isDirectory: true)
    }

    static func defaultLogURL() -> URL {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first ?? FileManager.default.temporaryDirectory
        return appSupport
            .appendingPathComponent("oMLX-next/logs/server.log")
    }
}

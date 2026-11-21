// Lifecycle owner for the `omlx serve` child process.
//
// State machine
//   stopped ─start()→ starting ─/health 200→ running ─/health fail×3→ unresponsive
//                       │                       │ ↑                       │
//                       │                       │ └─/health or status OK──┘
//                       │                       │
//                       │                       └─process exit → auto-restart
//                       └─process exit during startup → auto-restart
//
//   stop()  : * → stopping → SIGTERM → wait ≤10s → SIGKILL → stopped
//   forceRestart() : * → SIGKILL → start()
//   crashes : auto-restart with 5s/10s/20s backoff, max 3 attempts, counter
//             resets after 60s of stable .running
//
// Spawn invocation:
//   <python> -m omlx.cli serve --base-path <base> --port <port>
//   stdout+stderr → ~/Library/Application Support/oMLX/logs/server.log
//   PATH = parent + Homebrew prefixes
//
// Dev override: OMLX_DEV_SERVER_SCRIPT spawns <python> <script> --port …
// instead, used by Scripts/dev_server.py to exercise the spawn path without
// the full omlx stack.
//
// State changes posted via NotificationCenter so MenubarController and (in
// PR 6) the AppView shell can react without owning the lifecycle.

import Foundation
import AppKit
import Darwin
import IOKit
import os

/// Restart and health decisions are logged through `os.Logger` with public privacy:
/// `NSLog` output lands in the unified log as `<private>`, which makes a restart loop
/// undiagnosable exactly when it matters.
private let log = Logger(subsystem: "app.omlx", category: "server")

struct AutoRestartBudget {
    let maxAttempts: Int
    let stableThreshold: TimeInterval

    private(set) var attempts = 0
    private(set) var healthySince: Date?

    mutating func recordHealthy(at date: Date) {
        if healthySince == nil {
            healthySince = date
        }
        if attempts > 0,
           let since = healthySince,
           date.timeIntervalSince(since) >= stableThreshold {
            attempts = 0
            healthySince = date
        }
    }

    mutating func consumeRestart(at date: Date) -> Int? {
        if let since = healthySince,
           date.timeIntervalSince(since) >= stableThreshold {
            attempts = 0
        }
        healthySince = nil

        guard attempts < maxAttempts else { return nil }
        attempts += 1
        return attempts
    }

    mutating func reset() {
        attempts = 0
        healthySince = nil
    }
}

// @unchecked Sendable: state mutations either happen on the main thread
// (start, stop, force restart, callbacks dispatched via main) or inside
// the @MainActor health-check Task. Process termination handler bounces
// to main before touching state.
final class ServerProcess: @unchecked Sendable {
    enum State: Equatable, Sendable {
        case stopped
        case starting
        case running(pid: Int32)
        case stopping
        case unresponsive(pid: Int32)
        case failed(message: String)

        var isRunningLike: Bool {
            switch self {
            case .running, .unresponsive: return true
            default:                      return false
            }
        }
    }

    enum StartResult: Sendable {
        case started
        case alreadyRunning
        case portConflict(PortConflict)
    }

    enum StartError: Error, LocalizedError, CustomStringConvertible {
        case spawnFailed(String)
        case invalidPort

        var description: String {
            switch self {
            case .spawnFailed(let m): return "Spawn failed: \(m)"
            case .invalidPort: return "Port must be a number between 1 and 65535."
            }
        }

        var errorDescription: String? { description }
    }

    static let stateDidChangeNotification = Notification.Name("OMLXServerProcessStateDidChange")
    static let portConflictNotification   = Notification.Name("OMLXServerPortConflict")

    // Inputs

    private(set) var bindAddress: String
    /// The connectable host — normalises `0.0.0.0` → `127.0.0.1` because
    /// `0.0.0.0` is a bind wildcard, not a connectable address.
    var host: String {
        AppConfig.connectableHost(for: bindAddress)
    }
    private(set) var port: Int
    private(set) var basePath: URL
    private let runtime: PythonRuntime
    private var resolver: PortConflictResolver

    /// Apply a new host/port/basePath. Only legal while the server is
    /// stopped — won't take effect until the next `start()`. Caller is
    /// responsible for stopping first; we throw if asked to mutate a live
    /// process so a stale resolver / spawn args can never reach a running
    /// uvicorn.
    enum ReconfigureError: Error { case serverIsLive }
    func reconfigure(bindAddress: String? = nil, port: Int? = nil, basePath: URL? = nil) throws {
        switch state {
        case .running, .starting, .stopping, .unresponsive:
            throw ReconfigureError.serverIsLive
        case .stopped, .failed:
            break
        }
        if let bindAddress { self.bindAddress = bindAddress }
        if let port { self.port = port }
        if let basePath { self.basePath = basePath }
        self.resolver = PortConflictResolver(host: self.host, port: self.port)
    }

    // Tunables (mirror server_manager.py)

    private let healthCheckInterval: TimeInterval = 5
    private let maxHealthFailures = 3
    private let auxiliaryHealthFreshness: TimeInterval = 15
    private let stopGraceSeconds: TimeInterval = 10

    // State

    private(set) var state: State = .stopped
    private var process: Process?
    private(set) var startupNoticeURL: URL?
    private var logHandle: FileHandle?
    private var healthTask: Task<Void, Never>?
    private var consecutiveFailures = 0
    private var autoRestartBudget = AutoRestartBudget(
        maxAttempts: 3,
        stableThreshold: 60
    )
    private var lastAuxiliaryHealthyAt: Date?
    private var expectingExit       = false   // set by stop()/forceRestart() so terminationHandler doesn't trigger auto-restart
    private let logURL: URL
    /// How to run the server as the sandbox account while Secure Server mode is
    /// on, or nil to spawn it in process. Closures rather than a reference to the
    /// controller: the Server layer must not know about the privileged helper, and
    /// `SecureServerController` is a `@MainActor` object this class is not
    /// isolated to.
    var sandboxDirective: ServerSandbox?
    /// Pid of the helper-spawned server. Nothing else runs under the sandbox
    /// account, so this can never name somebody else's process.
    private var sandboxedPID: Int32?

    init(
        runtime: PythonRuntime,
        bindAddress: String = "127.0.0.1",
        port: Int = 8000,
        basePath: URL = ServerProcess.defaultBasePath()
    ) {
        self.runtime  = runtime
        self.bindAddress = bindAddress
        self.port     = port
        self.basePath = basePath
        self.logURL   = ServerProcess.defaultLogURL()
        self.resolver = PortConflictResolver(
            host: AppConfig.connectableHost(for: bindAddress),
            port: port
        )
    }

    // MARK: - Public surface

    var isRunning: Bool {
        if case .running = state { return true }
        if case .unresponsive = state { return true }
        return process?.isRunning == true || sandboxedPIDAlive
    }

    var pid: Int32? { process?.processIdentifier ?? sandboxedPID }
    var serverLogURL: URL { logURL }

    /// Liveness of the sandboxed server. A signal probe across uids returns
    /// EPERM, not success: permission denied *is* the answer that the process
    /// exists, and reading it as "gone" would turn every healthy sandboxed server
    /// into an instant crash-restart loop.
    private var sandboxedPIDAlive: Bool {
        guard let sandboxedPID else { return false }
        if kill(sandboxedPID, 0) == 0 { return true }
        return errno == EPERM
    }

    /// Start the server. Returns .started on success, .alreadyRunning if
    /// already up, or .portConflict if the port is busy. Throws on invalid
    /// persisted settings or spawn failure.
    @discardableResult
    func start() throws -> StartResult {
        switch state {
        case .running, .starting, .unresponsive:
            return .alreadyRunning
        default:
            break
        }

        // Web settings may have changed since the previous child was launched.
        let saved = try AppConfig.readSettings(basePath: basePath.path)
        let env = ProcessInfo.processInfo.environment
        let nextHost = env["OMLX_HOST"].flatMap { $0.isEmpty ? nil : $0 }
            ?? saved.bindAddress ?? bindAddress
        let nextPort = env["OMLX_PORT"].flatMap(Int.init) ?? saved.port ?? port
        guard (1...65535).contains(nextPort) else {
            throw StartError.invalidPort
        }
        try reconfigure(bindAddress: nextHost, port: nextPort)

        // The helper owns the port decision while the mode is on: its start() drains a
        // stale listener — including a uid-501 child of the server just stopped, which
        // `lsof` sees but a `_omlx`-scoped sweep cannot — and verifies the child bound
        // before it replies. An app-side "port busy" check here was both the false
        // "Port 8000 in use" alarm and a security hole: a leftover user-owned server
        // would read as "already running" and be left serving unsandboxed under a
        // padlock.
        if sandboxDirective == nil, resolver.isPortInUseSync() {
            let conflict = PortConflict(
                pid: resolver.findOwnerPIDSync(),
                isOMLX: resolver.isOMLXOnPortSync()
            )
            // An oMLX server holding the port is this app's server; attach to it.
            if conflict.isOMLX, let owner = conflict.pid, owner > 0 {
                log.notice("port \(self.port, privacy: .public) is held by oMLX itself (pid \(owner, privacy: .public)); adopting instead of reporting a conflict")
                sandboxedPID = owner
                update(.running(pid: owner))
                startHealthCheckLoop()
                return .alreadyRunning
            }
            update(.failed(message: "Port \(port) in use"))
            postPortConflict(conflict)
            return .portConflict(conflict)
        }

        // `.alreadyRunning` and `.portConflict` are the two ways a requested start
        // silently does nothing, and a handoff that leaves no server looks identical
        // from outside.
        log.notice("start requested from state \(String(describing: self.state), privacy: .public) → sandboxed=\(self.sandboxDirective != nil, privacy: .public)")
        try doStart()
        return .started
    }

    /// Everything `pid` forked, deepest first. uvicorn binds the listening socket at
    /// cli.py:238, before `init_server` forks the engine pool at 356, so those
    /// children inherit the fd: killing only the parent leaves :8000 held by a
    /// uid-501 process that a `_omlx`-scoped sweep cannot see.
    private func descendants(of pid: Int32, depth: Int = 0) -> [Int32] {
        guard depth < 8 else { return [] }
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/pgrep")
        task.arguments = ["-P", String(pid)]
        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = nil
        guard (try? task.run()) != nil else { return [] }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        task.waitUntilExit()
        let direct = String(data: data, encoding: .utf8)?
            .split(whereSeparator: \.isNewline)
            .compactMap { Int32($0) } ?? []
        return direct.flatMap { descendants(of: $0, depth: depth + 1) } + direct
    }

    /// Whether a socket with the same options uvicorn uses can bind right now.
    private func portIsBindable() -> Bool {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { return false }
        defer { close(fd) }
        var reuse: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout<Int32>.size))
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = in_port_t(port).bigEndian
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")
        let result = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        return result == 0
    }

    private func waitUntilPortBindable(seconds: Double = 20) async {
        let deadline = Date().addingTimeInterval(seconds)
        while Date() < deadline {
            if portIsBindable() { return }
            try? await Task.sleep(for: .milliseconds(200))
        }
        log.warning("port \(self.port, privacy: .public) never became bindable in \(Int(seconds), privacy: .public)s")
    }

    /// Graceful stop: SIGTERM → wait ≤ stopGraceSeconds → SIGKILL.
    func stop(timeout: TimeInterval? = nil) async {
        guard isRunning || state == .starting else { return }

        update(.stopping)
        expectingExit = true
        cancelHealthLoop()

        let timeout = timeout ?? stopGraceSeconds
        if let proc = process, proc.isRunning {
            let tree = descendants(of: proc.processIdentifier)
            kill(proc.processIdentifier, SIGTERM)
            for pid in tree where pidIsAlive(pid) { kill(pid, SIGTERM) }

            let deadline = Date().addingTimeInterval(timeout)
            while proc.isRunning && Date() < deadline {
                try? await Task.sleep(for: .milliseconds(100))
            }
            if proc.isRunning {
                kill(proc.processIdentifier, SIGKILL)
                for pid in tree where pidIsAlive(pid) { kill(pid, SIGKILL) }
                try? await Task.sleep(for: .seconds(0.5))
            }
            for pid in tree where pidIsAlive(pid) { kill(pid, SIGKILL) }
        } else if process == nil, let sandbox = sandboxDirective {
            // The sandboxed server belongs to another user: this process cannot
            // signal it, and the helper has to do the killing *and* the sweep, so
            // that nothing the server spawned (a worker, a tool subprocess) is
            // left holding a socket under rules that no longer have a purpose.
            _ = await sandbox.stop()
            sandboxedPID = nil
        } else if process == nil, sandboxDirective == nil, let adopted = sandboxedPID, adopted > 0 {
            // Adopted because it was already listening: same uid, so it can be
            // signalled directly. Without this an adopted server could be started again
            // but never stopped.
            kill(adopted, SIGTERM)
            let deadline = Date().addingTimeInterval(timeout)
            while pidIsAlive(adopted) && Date() < deadline {
                try? await Task.sleep(for: .milliseconds(100))
            }
            if pidIsAlive(adopted) { kill(adopted, SIGKILL) }
            sandboxedPID = nil
        }

        // terminationHandler updates state to .stopped; force in case it
        // didn't fire yet.
        if state != .stopped {
            update(.stopped)
        }
        expectingExit = false
        process = nil
        lastAuxiliaryHealthyAt = nil
        closeLog()
    }

    /// Force-restart: SIGKILL the child without waiting, reset counters,
    /// then start() fresh.
    @discardableResult
    func forceRestart() async throws -> StartResult {
        expectingExit = true
        cancelHealthLoop()
        if let proc = process, proc.isRunning {
            let tree = descendants(of: proc.processIdentifier)
            // SIGKILL closes the listening fd abruptly and leaves a bound-but-not-
            // listening socket that `lsof -sTCP:LISTEN` cannot see, yet still makes
            // the replacement's bind fail with EADDRINUSE. Let the server shut down
            // gracefully first; escalate only if it refuses.
            kill(proc.processIdentifier, SIGTERM)
            for pid in tree where pidIsAlive(pid) { kill(pid, SIGTERM) }
            let deadline = Date().addingTimeInterval(3)
            while proc.isRunning && Date() < deadline {
                try? await Task.sleep(for: .milliseconds(50))
            }
            if proc.isRunning {
                kill(proc.processIdentifier, SIGKILL)
                for pid in tree where pidIsAlive(pid) { kill(pid, SIGKILL) }
            }
            for pid in tree where pidIsAlive(pid) { kill(pid, SIGKILL) }
        } else if process == nil, let sandbox = sandboxDirective {
            // Not gated on `sandboxedPID`: if the app lost track of the pid (a helper
            // restart, a failed start) there is still very likely a sandboxed server
            // holding the port, and only the helper can stop it. That mismatch is
            // exactly why Force Restart used to collide while Start worked — Start
            // goes through `stop()`, which does not consult the pid.
            guard await sandbox.stop() else {
                // Starting now would collide with the survivor on the port, and the
                // failure would look like a port conflict rather than a server that
                // would not die.
                expectingExit = false
                update(.failed(message: "the sandboxed server did not stop, so it could not be restarted"))
                throw StartError.spawnFailed("the sandboxed server did not stop")
            }
            sandboxedPID = nil
        }
        if process == nil, sandboxDirective == nil, let adopted = sandboxedPID, adopted > 0 {
            kill(adopted, SIGKILL)
            let deadline = Date().addingTimeInterval(2)
            while pidIsAlive(adopted) && Date() < deadline {
                try? await Task.sleep(for: .milliseconds(50))
            }
            sandboxedPID = nil
        }
        process = nil
        closeLog()
        autoRestartBudget.reset()
        consecutiveFailures = 0
        lastAuxiliaryHealthyAt = nil
        expectingExit = false
        update(.stopped)
        // Ask the kernel the exact question the child will ask, *before* sending the
        // spawn request (no child exists yet, so this cannot compete with its bind).
        // Covers the FIN_WAIT / bound-not-listening socket a just-killed local server
        // leaves behind, which lsof cannot see.
        await waitUntilPortBindable()
        return try start()
    }

    /// Called by lightweight menubar status polling when the server answers
    /// `/api/status`. Under heavy generation load this keeps the UI from
    /// declaring the managed process unresponsive solely because `/health`
    /// was delayed.
    @MainActor
    func recordAuxiliaryHealthSuccess(at date: Date = Date()) {
        // A recent response from an exited child is not evidence of a live server.
        guard let process, process.isRunning else { return }
        lastAuxiliaryHealthyAt = date
        autoRestartBudget.recordHealthy(at: date)
        consecutiveFailures = 0
        switch state {
        case .starting:
            if let pid {
                update(.running(pid: pid))
            }
        case .unresponsive(let pid):
            update(.running(pid: pid))
        default:
            break
        }
    }

    /// Synchronous SIGTERM-then-SIGKILL of the child, used by signal
    /// handlers (which can't await). Returns when the kernel reports
    /// the PID as gone or after `timeout` seconds.
    func reapSync(timeout: TimeInterval = 5) {
        guard let proc = process, proc.isRunning else { return }
        let pid = proc.processIdentifier
        kill(pid, SIGTERM)
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline, kill(pid, 0) == 0 {
            usleep(100_000)                     // 100 ms
        }
        if kill(pid, 0) == 0 { kill(pid, SIGKILL) }
        // The app is quitting, so the async exit handler may never run.
        removeLeftoverGPUPowerCap(serverPID: pid)
    }

    // MARK: - Internal — spawn

    private func doStart() throws {
        try ensureDir(basePath)
        try ensureDir(logURL.deletingLastPathComponent())
        consecutiveFailures = 0
        lastAuxiliaryHealthyAt = nil

        if let sandbox = sandboxDirective {
            try doStartSandboxed(sandbox)
            return
        }
        // No directive means nobody attached the helper, but the record says the mode
        // is on. Spawning here would put a server with unrestricted network behind a
        // locked padlock — the one outcome this feature exists to prevent — so the
        // start fails where the problem is, instead of pretending to succeed.
        if SecureServer.enabledOnRecord {
            throw StartError.spawnFailed(String(
                localized: "security.secure_server.error.spawn_without_helper",
                defaultValue: "Secure Server is on, so the server can only be started through the oMLX privileged helper. Check that the helper is registered, or turn Secure Server off.",
                comment: "Shown when the server is asked to start while Secure Server is on but no helper is attached"
            ))
        }

        // Mode turned off since the last run: a leftover sandbox pid would keep
        // `pid`/`isRunning` reporting a process that no longer belongs to us.
        sandboxedPID = nil

        if !FileManager.default.fileExists(atPath: logURL.path) {
            FileManager.default.createFile(atPath: logURL.path, contents: nil)
        }
        let handle = try FileHandle(forWritingTo: logURL)
        try handle.seekToEnd()
        logHandle = handle

        let proc = Process()
        proc.executableURL = runtime.executable
        proc.arguments = makeArguments()
        // A per-launch notice keeps Python migration and the native alert in sync.
        var environment = runtime.makeEnvironment()
        let noticeURL = prepareStartupNotice()
        environment["OMLX_STARTUP_NOTICE_PATH"] = noticeURL.path
        proc.environment = environment
        proc.standardOutput = handle
        proc.standardError  = handle
        proc.terminationHandler = { [weak self] term in
            DispatchQueue.main.async {
                self?.handleProcessExit(code: term.terminationStatus, pid: term.processIdentifier)
            }
        }

        update(.starting)
        do {
            try proc.run()
        } catch {
            closeLog()
            update(.failed(message: "spawn failed: \(error.localizedDescription)"))
            throw StartError.spawnFailed(error.localizedDescription)
        }
        process = proc
        startHealthCheckLoop()
    }

    /// Liveness of any pid this process knows about, including a server it adopted.
    private func pidIsAlive(_ pid: Int32) -> Bool {
        if kill(pid, 0) == 0 { return true }
        return errno == EPERM
    }

    /// Whether a new listener could take the port right now.
    /// Takes over a sandboxed server this process did not spawn.
    ///
    /// The helper starts that server and the app can only be told its pid, so after a
    /// relaunch while the mode is on nothing in the app knows a server is running: the
    /// padlock says on, the UI says stopped, and flipping the mode off kills the
    /// server without anything taking its place. Adopting makes the existing pid the
    /// one the health loop watches and the one the handoff restarts.
    func adoptSandboxedServer(pid: Int32) {
        guard process == nil, sandboxedPID == nil else { return }
        // `.failed` is where the app lands when it probed the port at launch and found
        // its own sandboxed server listening — adopting is exactly the correction.
        switch state {
        case .stopped, .failed:
            break
        default:
            return
        }
        sandboxedPID = pid
        update(.running(pid: pid))
        startHealthCheckLoop()
    }

    /// Ask the privileged helper to spawn the server as the sandbox account.
    ///
    /// There is deliberately no fallback to the local spawn. A server started as
    /// the console user owns sockets that match none of the `user` rules, so it
    /// would keep full network access while the UI showed the lockdown as on —
    /// the one outcome worse than a server that fails to start.
    private func doStartSandboxed(_ sandbox: ServerSandbox) throws {
        // A sandboxed server that is already alive is reused rather than replaced:
        // starting one goes through the helper, whose first act is to stop whatever it
        // is running, so a restart triggered by a misread liveness check would kill a
        // healthy server — measured on this machine as four starts in one minute, each
        // one killing a boot that needs about nine seconds.
        if let pid = sandboxedPID, sandboxedPIDAlive {
            update(.running(pid: pid))
            startHealthCheckLoop()
            return
        }
        update(.starting)
        let request = sandboxLaunch()
        Task { @MainActor [weak self] in
            guard let self else { return }
            // The helper is the authority on what it is already running. At launch the
            // app's own adoption is still in flight — it awaits a measured status — while
            // auto-start runs in the same turn as bind(), so `sandboxedPID` above is nil
            // and the reuse guard cannot fire. `start()` stops whatever the helper has
            // before spawning, so without this check a relaunch kills a healthy server
            // and pays a full boot to replace it with the same thing. The helper
            // validates this pid (alive and owned by the sandbox uid) or reports none.
            if let running = await sandbox.runningPID() {
                guard case .starting = self.state else { return }
                self.sandboxedPID = running
                self.update(.running(pid: running))
                self.startHealthCheckLoop()
                log.notice(
                    "reusing sandboxed server \(running, privacy: .public) the helper already had running"
                )
                return
            }
            do {
                let pid = try await sandbox.start(request)
                // stop() or a restart may have intervened while the helper worked.
                guard case .starting = self.state else { return }
                self.sandboxedPID = pid
                self.update(.running(pid: pid))
                self.startHealthCheckLoop()
            } catch {
                self.sandboxedPID = nil
                self.update(
                    .failed(message: "sandboxed start failed: \(error.localizedDescription)")
                )
            }
        }
    }

    private func sandboxLaunch() -> SecureServerLaunch {
        var environment = runtime.makeEnvironment()
        // Logs and the disk cache default into the base directory, which the sandbox
        // cannot write. `omlx/settings.py` honours exactly these two variables, so
        // what the server creates goes into the sandbox's own home instead.
        environment[SecureServer.logDirEnvironmentKey] = SecureServer.sandboxLogDir
        environment[SecureServer.cacheDirEnvironmentKey] = SecureServer.sandboxCacheDir
        return SecureServerLaunch(
            executable: runtime.executable.path,
            arguments: sandboxArguments(),
            environment: environment,
            port: port,
            accessPaths: sandboxAccessPaths(),
            writablePaths: sandboxWritablePaths(),
            settingsPath: basePath.appendingPathComponent("settings.json").path
        )
    }

    /// What under the base directory the server legitimately modifies: the settings
    /// files and the databases beside them. `settings.json` is on the list even though
    /// the server saves into its seeded copy now: the base directory's
    /// `add_file,delete_child` already lets a replace of any top-level entry, so the
    /// grant is what makes such a replace *recorded* — with an owner to hand back —
    /// rather than invisible. Nothing here narrows what the same server does as you.
    ///
    /// A save writes a temp file and renames over the target, and a measured replace
    /// needs `add_file,delete_child` on the containing directory. So while the mode is
    /// on the sandbox can replace or remove top-level entries of the oMLX data folder —
    /// which is what the same server already does running as you — and gets nothing
    /// else: not the rest of the home directory, not the right to create new
    /// directories. Every entry is recorded and revoked on the way out, and anything
    /// the sandbox ended up owning is handed back to its original owner.
    private func sandboxWritablePaths() -> [String] {
        var paths = [basePath.path]
        guard let entries = try? FileManager.default.contentsOfDirectory(atPath: basePath.path) else {
            return paths
        }
        let named: Set<String> = ["settings.json", "model_settings.json"]
        paths.append(contentsOf: entries
            .filter { named.contains($0) || $0.hasSuffix(".db") || $0.hasSuffix(".sqlite") }
            .map { basePath.appendingPathComponent($0).path })
        return paths
    }

    /// The sandboxed server is pinned to loopback whatever the settings say.
    ///
    /// `omlx serve` resolves its bind address as CLI > env > settings.json, and the
    /// normal spawn passes no `--host` at all, so a `0.0.0.0` in settings would
    /// otherwise survive into secure mode and let any device on the network reach
    /// the server. Passing the flag makes the choice authoritative; the pf inbound
    /// block stays as the second line of defence for the case where the argument
    /// cannot be honoured.
    private func sandboxArguments() -> [String] {
        var args = makeArguments()
        guard let index = args.firstIndex(of: "--host"), args.indices.contains(index + 1) else {
            args.append(contentsOf: ["--host", "127.0.0.1"])
            return args
        }
        args[index + 1] = "127.0.0.1"
        return args
    }

    /// Everything the sandbox has to reach. Changing uid means losing the access
    /// the logged-in user had, and one `drwx------` ancestor anywhere on the path
    /// makes the server fail to exec rather than fail loudly: uv hides its
    /// interpreters behind `~/.local`, which is exactly such a directory.
    private func sandboxAccessPaths() -> [String] {
        var paths = [
            runtime.executable.path,
            runtime.executable.deletingLastPathComponent().path,
            basePath.path
        ]
        if let dev = ProcessInfo.processInfo.environment["OMLX_DEV_SERVER_SCRIPT"],
           !dev.isEmpty {
            paths.append(dev)
        }
        // Where Python finds its packages and caches, as the runtime says it.
        let pathKeys = [
            "PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME",
            "HF_HOME", "HF_HUB_CACHE", "XDG_CACHE_HOME", "MODELSCOPE_CACHE"
        ]
        let environment = runtime.makeEnvironment()
        for key in pathKeys {
            guard let value = environment[key], !value.isEmpty else { continue }
            paths.append(contentsOf: value.split(separator: ":").map(String.init))
        }
        return paths.filter { FileManager.default.fileExists(atPath: $0) }
    }

    /// Loss of the sandboxed server, reported by the health loop.
    private func handleSandboxedExit() {
        guard process == nil, let gone = sandboxedPID else { return }
        // The pid and the syscall's own verdict, captured before anything is cleared:
        // reading them afterwards reports nil/false by construction and cannot tell a
        // server that exited from a live one this check misreads.
        let probe = kill(gone, 0)
        let probeErrno = probe == 0 ? 0 : errno
        log.error(
            "sandboxed server \(gone, privacy: .public) judged exited: kill=\(probe, privacy: .public) errno=\(probeErrno, privacy: .public)"
        )
        let wasExpectingExit = expectingExit
        sandboxedPID = nil
        if wasExpectingExit {
            update(.stopped)
            return
        }
        tryAutoRestart(reason: "Sandboxed server exited")
    }

    private func handleProcessExit(code: Int32, pid: Int32) {
        // The handler is enqueued via DispatchQueue.main.async, while
        // stop()/forceRestart() only poll proc.isRunning — so a stale
        // handler from an OLD process can land after a new child has
        // already been spawned and assigned to `process`. Without this
        // guard it would null the new process reference, close the new
        // log handle, and (if the new process is still .starting)
        // tryAutoRestart a second time — orphaning the first child, which
        // the menubar's stop/forceRestart can no longer reach, typically
        // surfacing as a port conflict (§F2).
        // Only a *different* live child makes this handler stale. With no current
        // process there is nothing to clobber, so the exit must still be handled —
        // stop() and the harness both reach here with `process` already nil.
        if let current = process, current.processIdentifier != pid { return }
        cancelHealthLoop()
        lastAuxiliaryHealthyAt = nil
        let wasExpectingExit = expectingExit
        expectingExit = false
        process = nil
        closeLog()
        removeLeftoverGPUPowerCap(serverPID: pid)
        Task { @MainActor [weak self] in self?.presentStartupNotice() }

        if wasExpectingExit {
            update(.stopped)
            return
        }

        switch state {
        case .starting:
            tryAutoRestart(reason: "Server exited with code \(code) during startup")
        case .running, .unresponsive:
            tryAutoRestart(reason: "Server exited with code \(code)")
        default:
            // Unexpected — log and stop.
            update(.stopped)
        }
    }

    /// A killed server cannot remove its system-wide GPU power cap. While a
    /// cap is set the server keeps this marker with its PID
    /// (omlx/utils/gpu_power.py); a marker from another server is left alone.
    private func removeLeftoverGPUPowerCap(serverPID: Int32) {
        let marker = basePath.appendingPathComponent("gpu_power_cap.active")
        guard let owner = try? String(contentsOf: marker, encoding: .utf8),
              Int32(owner.trimmingCharacters(in: .whitespacesAndNewlines)) == serverPID
        else { return }
        let service = IOServiceGetMatchingService(
            kIOMainPortDefault, IOServiceMatching("AGXAccelerator"))
        guard service != 0 else { return }
        defer { IOObjectRelease(service) }
        let uncap: [String: Any] = ["SetMaxGPUAbsolutePower": true, "AbsoluteTarget": -1000]
        if IORegistryEntrySetCFProperties(service, uncap as CFDictionary) == KERN_SUCCESS {
            try? FileManager.default.removeItem(at: marker)
        }
    }

    private func tryAutoRestart(reason: String) {
        guard let attempt = autoRestartBudget.consumeRestart(at: Date()) else {
            update(.failed(
                message: "\(reason). Auto-restart failed after " +
                         "\(autoRestartBudget.maxAttempts) attempts."
            ))
            return
        }

        consecutiveFailures = 0
        lastAuxiliaryHealthyAt = nil
        let backoff = TimeInterval(5 * (1 << (attempt - 1)))   // 5, 10, 20s

        log.notice(
            "auto-restart \(attempt, privacy: .public)/\(self.autoRestartBudget.maxAttempts, privacy: .public) in \(Int(backoff), privacy: .public)s — \(reason, privacy: .public) [sandboxedPID=\(self.sandboxedPID.map(String.init) ?? "nil", privacy: .public) alive=\(String(self.sandboxedPIDAlive), privacy: .public) pid=\(self.pid.map(String.init) ?? "nil", privacy: .public)]"
        )
        update(.starting)

        Task { @MainActor [weak self] in
            try? await Task.sleep(for: .seconds(backoff))
            guard let self else { return }
            // If the user (or stop) intervened during backoff, abort.
            guard case .starting = self.state else { return }
            // Something may be answering again by now, and restarting is not free:
            // the helper's start() stops whatever it is running first. A healthy probe
            // ends the restart instead of killing the server that took the backoff to
            // come up.
            if await self.resolver.probeHealth().ok {
                log.notice("auto-restart \(attempt, privacy: .public) skipped: the server answers")
                self.markHealthy(pid: self.pid ?? 0, at: Date())
                self.startHealthCheckLoop()
                return
            }
            do {
                try self.doStart()
                log.notice("auto-restart \(attempt, privacy: .public) started from \(String(describing: self.state), privacy: .public)")
            } catch {
                self.update(.failed(message: "Auto-restart failed: \(error)"))
            }
        }
    }

    // MARK: - Internal — health check

    func prepareStartupNotice() -> URL {
        if let previous = startupNoticeURL {
            try? FileManager.default.removeItem(at: previous)
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("omlx-startup-\(UUID().uuidString).txt")
        startupNoticeURL = url
        return url
    }

    @MainActor
    func consumeStartupNotice() -> String? {
        guard let url = startupNoticeURL,
              let message = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        try? FileManager.default.removeItem(at: url)
        startupNoticeURL = nil
        bindAddress = "127.0.0.1"
        resolver = PortConflictResolver(host: host, port: port)
        NotificationCenter.default.post(name: Self.stateDidChangeNotification, object: self)
        return message
    }

    @MainActor
    private func presentStartupNotice() {
        guard let message = consumeStartupNotice() else { return }
        NSApp.activate(ignoringOtherApps: true)
        let alert = NSAlert()
        alert.alertStyle = .warning
        alert.messageText = "Server access is now limited to this Mac"
        alert.informativeText = message
        alert.addButton(withTitle: "OK")
        alert.window.level = .floating
        alert.runModal()
    }

    private func startHealthCheckLoop() {
        cancelHealthLoop()
        healthTask = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                await self.tickHealth()
                try? await Task.sleep(for: .seconds(self.healthCheckInterval))
            }
        }
    }

    private func cancelHealthLoop() {
        healthTask?.cancel()
        healthTask = nil
    }

    @MainActor
    private func tickHealth() async {
        switch state {
        case .starting, .running, .unresponsive:
            break
        default:
            return
        }

        // A sandboxed server that dies fires no termination handler, because the
        // process was never ours, so liveness has to be checked here.
        if process == nil, sandboxedPID != nil, !sandboxedPIDAlive {
            handleSandboxedExit()
            return
        }
        presentStartupNotice()
        // .starting also covers restart backoff, when no child exists yet. A sandboxed
        // server has no local Process at all; its liveness was checked above instead.
        let probedProcess = process
        if probedProcess == nil, sandboxedPID == nil { return }
        if let probedProcess, !probedProcess.isRunning { return }
        let probe = await resolver.probeHealth()
        // The child may have exited or been replaced while the probe was pending.
        guard !Task.isCancelled,
              process === probedProcess,
              probedProcess?.isRunning ?? sandboxedPIDAlive else { return }
        let now = Date()
        switch state {
        case .starting:
            if probe.ok || hasRecentAuxiliaryHealth(now: now) {
                let pid = pid ?? 0
                markHealthy(pid: pid, at: now)
            } else {
                logHealthProbeFailure(probe, failures: consecutiveFailures, suppressed: false)
            }
        case .running(let pid), .unresponsive(let pid):
            if probe.ok {
                markHealthy(pid: pid, at: now)
            } else if hasRecentAuxiliaryHealth(now: now) {
                logHealthProbeFailure(probe, failures: consecutiveFailures, suppressed: true)
                markHealthy(pid: pid, at: now)
            } else {
                consecutiveFailures += 1
                logHealthProbeFailure(probe, failures: consecutiveFailures, suppressed: false)
                if consecutiveFailures >= maxHealthFailures,
                   case .running = state {
                    update(.unresponsive(pid: pid))
                }
            }
        default:
            return
        }
    }

    @MainActor
    private func markHealthy(pid: Int32, at date: Date) {
        consecutiveFailures = 0
        autoRestartBudget.recordHealthy(at: date)
        switch state {
        case .starting, .unresponsive:
            update(.running(pid: pid))
        default:
            break
        }
    }

    private func hasRecentAuxiliaryHealth(now: Date) -> Bool {
        guard let lastAuxiliaryHealthyAt else { return false }
        return now.timeIntervalSince(lastAuxiliaryHealthyAt) <= auxiliaryHealthFreshness
    }

    private func logHealthProbeFailure(
        _ result: HealthProbeResult,
        failures: Int,
        suppressed: Bool
    ) {
        let status = result.statusCode.map(String.init) ?? "none"
        let error = result.errorDescription ?? "none"
        NSLog(
            "oMLX: health probe failed url=\(result.url) latency_ms=\(result.latencyMs) status=\(status) error=\(error) failures=\(failures) suppressed_by_recent_status=\(suppressed)"
        )
    }

    // MARK: - Internal — helpers

    private func makeArguments() -> [String] {
        let env = ProcessInfo.processInfo.environment
        if let dev = env["OMLX_DEV_SERVER_SCRIPT"], !dev.isEmpty {
            return [dev, "--host", bindAddress, "--port", String(port)]
        }
        return [
            "-m", "omlx.cli", "serve",
            "--base-path", basePath.path,
            "--port", String(port),
        ]
    }

    private func update(_ next: State) {
        guard state != next else { return }
        state = next
        // Observers (MenubarController.serverStateChanged, AppServices) are
        // `@MainActor`. We can be called from the cooperative executor pool
        // (via `await stop()` / `await forceRestart()` / `tickHealth`), so a
        // direct synchronous post trips Swift 6's actor-isolation check and
        // crashes the parent — see crash report 2026-05-09. Hop to main.
        let note = Self.stateDidChangeNotification
        if Thread.isMainThread {
            NotificationCenter.default.post(name: note, object: self)
        } else {
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                NotificationCenter.default.post(name: note, object: self)
            }
        }
    }

    private func postPortConflict(_ conflict: PortConflict) {
        let note = Self.portConflictNotification
        if Thread.isMainThread {
            NotificationCenter.default.post(name: note, object: self, userInfo: ["conflict": conflict])
        } else {
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                NotificationCenter.default.post(name: note, object: self, userInfo: ["conflict": conflict])
            }
        }
    }

    private func ensureDir(_ url: URL) throws {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    }

    private func closeLog() {
        try? logHandle?.close()
        logHandle = nil
    }

    static func defaultBasePath() -> URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".omlx", isDirectory: true)
    }

    static func defaultLogURL() -> URL {
        AppConfig.appSupportURL().appendingPathComponent("logs/server.log")
    }
}

// MARK: - Port conflict payload

struct PortConflict: Sendable, Equatable {
    let pid: pid_t?
    let isOMLX: Bool
}

// MARK: - Sandboxed launch hooks

/// How to run the server as somebody else. Filled in by whoever owns both the
/// server and the privileged helper, so that `ServerProcess` never has to name
/// either: while Secure Server mode is on these replace the local spawn, because
/// the pf rules match on the uid that owns the socket.
struct ServerSandbox: Sendable {
    var start: @Sendable (SecureServerLaunch) async throws -> Int32
    var stop: @Sendable () async -> Bool
    /// The pid the helper already has running, if any. Asked before `start` because
    /// `start` replaces whatever is running, and at launch the app has not yet
    /// adopted it.
    var runningPID: @Sendable () async -> Int32?
}

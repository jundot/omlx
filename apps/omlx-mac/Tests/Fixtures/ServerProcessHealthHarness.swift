// SPDX-License-Identifier: Apache-2.0
// Compiled after ServerProcess.swift by test_server_process_health.py.
// Dependencies below isolate settings, logs, HTTP and the child executable.

struct PythonRuntime {
    let executable = URL(fileURLWithPath: CommandLine.arguments[2])
    func makeEnvironment() -> [String: String] { [:] }
}

struct AppConfig {
    struct Saved {
        let bindAddress: String? = nil
        let port: Int? = nil
    }
    static func readSettings(basePath: String) throws -> Saved { Saved() }
    static func connectableHost(for host: String) -> String { host }
    static func appSupportURL() -> URL { URL(fileURLWithPath: CommandLine.arguments[1]) }
}

struct HealthProbeResult: Sendable {
    let ok: Bool
    let url: String
    let latencyMs: Int
    let statusCode: Int?
    let errorDescription: String?
}

struct PortConflictResolver: Sendable {
    let host: String
    let port: Int
    func isPortInUseSync() -> Bool { false }
    func findOwnerPIDSync() -> pid_t? { nil }
    func isOMLXOnPortSync() -> Bool { false }
    func probeHealth() async -> HealthProbeResult { await ProbeControl.probe() }
}

@MainActor
private enum ProbeControl {
    static var ok = false
    static var hold = false
    static var calls = 0
    static var continuation: CheckedContinuation<HealthProbeResult, Never>?

    static func result(_ ok: Bool) -> HealthProbeResult {
        HealthProbeResult(ok: ok, url: "mock://health", latencyMs: 0,
                          statusCode: ok ? 200 : nil,
                          errorDescription: ok ? nil : "connection refused")
    }

    static func probe() async -> HealthProbeResult {
        calls += 1
        if hold {
            return await withCheckedContinuation { continuation = $0 }
        }
        return result(ok)
    }

    static func waitUntilPending() async throws {
        let deadline = Date().addingTimeInterval(2)
        while continuation == nil, Date() < deadline {
            try await Task.sleep(for: .milliseconds(1))
        }
        try require(continuation != nil, "probe did not suspend")
    }

    static func complete() {
        let pending = continuation
        continuation = nil
        hold = false
        pending?.resume(returning: result(true))
    }
}

private struct RegressionFailure: Error, CustomStringConvertible {
    let description: String
}

private func require(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    if !condition() { throw RegressionFailure(description: message) }
}

extension ServerProcess {
    @MainActor
    fileprivate func runHealthRegression(_ scenario: String) async throws {
        var children: [Process] = []
        func child() throws -> Process {
            let child = Process()
            child.executableURL = URL(fileURLWithPath: "/bin/sleep")
            child.arguments = ["30"]
            try child.run()
            children.append(child)
            return child
        }
        defer {
            cancelHealthLoop()
            state = .stopped // Cancel any pending restart, including on assertion failure.
            if let process { children.append(process) }
            for child in children {
                if child.isRunning { child.terminate() }
                child.waitUntilExit()
            }
            if let startupNoticeURL { try? FileManager.default.removeItem(at: startupNoticeURL) }
            closeLog()
        }

        switch scenario {
        case "restart_after_stale_status", "stop_during_backoff", "exit_cancels_health_loop":
            // Queue the same restart as a clean web-triggered server exit.
            // Keep a handle to the previous loop to check actual cancellation.
            state = .running(pid: 42)
            let oldLoop = Task<Void, Never> { try? await Task.sleep(for: .seconds(60)) }
            healthTask = oldLoop
            defer { oldLoop.cancel() }
            lastAuxiliaryHealthyAt = Date()
            handleProcessExit(code: 0, pid: 42)

            if scenario == "exit_cancels_health_loop" {
                try require(oldLoop.isCancelled && healthTask == nil, "old health loop survived exit")
                try require(lastAuxiliaryHealthyAt == nil, "old auxiliary health survived exit")
                return
            }
            if scenario == "stop_during_backoff" {
                await stop()
            } else {
                // The first failed stats poll used to replay its cached success.
                recordAuxiliaryHealthSuccess(at: Date().addingTimeInterval(-2))
                await tickHealth()
            }
            // Exercise the production five-second restart task, not a copy.
            let deadline = Date().addingTimeInterval(7)
            if scenario == "stop_during_backoff" {
                try await Task.sleep(for: .milliseconds(5500))
                try require(state == .stopped && process == nil, "manual stop allowed a respawn")
            } else {
                while process == nil, Date() < deadline {
                    try await Task.sleep(for: .milliseconds(10))
                }
                try require(process?.isRunning == true, "stale status suppressed the scheduled restart: \(state)")
                try require((pid ?? 0) > 0, "replacement has no real PID")
            }

        case "no_child_health":
            state = .starting // Auto-restart backoff has no child yet.
            recordAuxiliaryHealthSuccess()
            try require(lastAuxiliaryHealthyAt == nil, "no-child status recorded as healthy")
            ProbeControl.ok = true
            await tickHealth()
            try require(state == .starting && pid == nil, "probe created a phantom running server")
            try require(ProbeControl.calls == 0, "backoff probed an endpoint without a child")

        case "dead_child_status":
            let exited = try child()
            exited.terminate()
            exited.waitUntilExit()
            process = exited // Termination callback has not run yet.
            state = .starting
            recordAuxiliaryHealthSuccess()
            try require(state == .starting && lastAuxiliaryHealthyAt == nil,
                        "status revived an already exited child")

        case "live_health", "live_status_fallback", "live_failure_and_recovery":
            let live = try child()
            process = live
            state = .starting
            let running = State.running(pid: live.processIdentifier)
            if scenario == "live_health" {
                ProbeControl.ok = true
                await tickHealth()
                try require(state == running, "successful probe did not mark the live PID running")
            } else if scenario == "live_status_fallback" {
                state = .unresponsive(pid: live.processIdentifier)
                recordAuxiliaryHealthSuccess()
                try require(state == running, "status did not recover a live child")
                for _ in 0..<3 { await tickHealth() }
                try require(state == running, "recent status did not suppress failed health probes")
            } else {
                state = running
                for _ in 0..<3 { await tickHealth() }
                try require(state == .unresponsive(pid: live.processIdentifier),
                            "consecutive failures did not mark the live child unresponsive")
                ProbeControl.ok = true
                await tickHealth()
                try require(state == running, "successful probe did not recover a live child")
            }

        case "probe_after_exit", "probe_after_replacement", "cancelled_probe":
            process = try child()
            state = .starting
            ProbeControl.hold = true
            let pending = Task { @MainActor in await tickHealth() }
            try await ProbeControl.waitUntilPending()
            if scenario == "probe_after_exit" {
                process?.terminate()
                process?.waitUntilExit()
                process = nil
            } else if scenario == "probe_after_replacement" {
                process = try child()
            } else {
                pending.cancel()
            }
            ProbeControl.complete()
            await pending.value
            try require(state == .starting, "late/cancelled probe changed lifecycle state: \(state)")

        default:
            throw RegressionFailure(description: "unknown scenario: \(scenario)")
        }
    }
}

@main
private struct ServerProcessHealthHarness {
    @MainActor
    static func main() async {
        let root = URL(fileURLWithPath: CommandLine.arguments[1])
        let scenario = CommandLine.arguments[3]
        do {
            let server = ServerProcess(runtime: PythonRuntime(), basePath: root)
            try await server.runHealthRegression(scenario)
            print("PASS \(scenario)")
        } catch {
            print("FAIL \(scenario): \(error)")
            exit(1)
        }
    }
}

// SPDX-License-Identifier: Apache-2.0
// Appended to ServerProcess.swift by test_server_process_health.py. Stubs its app dependencies.

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
    static var hold = false
    static var continuation: CheckedContinuation<HealthProbeResult, Never>?

    static func result(_ ok: Bool) -> HealthProbeResult {
        HealthProbeResult(ok: ok, url: "mock://health", latencyMs: 0,
                          statusCode: ok ? 200 : nil,
                          errorDescription: ok ? nil : "connection refused")
    }

    static func probe() async -> HealthProbeResult {
        if hold {
            return await withCheckedContinuation { continuation = $0 }
        }
        return result(false)
    }
}

private struct RegressionFailure: Error, CustomStringConvertible {
    let description: String
}

private func require(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    if !condition() { throw RegressionFailure(description: message) }
}

@MainActor
private func waitUntil(seconds: Double, _ condition: () -> Bool) async throws {
    let deadline = Date().addingTimeInterval(seconds)
    while !condition(), Date() < deadline {
        try await Task.sleep(for: .milliseconds(10))
    }
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
            state = .stopped // Cancels a pending auto-restart.
            if let process { children.append(process) }
            for child in children {
                if child.isRunning { child.terminate() }
                child.waitUntilExit()
            }
            if let startupNoticeURL { try? FileManager.default.removeItem(at: startupNoticeURL) }
            closeLog()
        }

        switch scenario {
        case "restart_after_stale_status":
            // A clean exit queues the 5 s auto-restart, as with the web Restart Server.
            state = .running(pid: 42)
            lastAuxiliaryHealthyAt = Date()
            handleProcessExit(code: 0, pid: 42)
            // The first failed stats poll reposts its cached success.
            recordAuxiliaryHealthSuccess(at: Date().addingTimeInterval(-2))
            await tickHealth()
            try await waitUntil(seconds: 7) { process != nil }
            try require(process?.isRunning == true, "stale status suppressed the restart: \(state)")

        case "probe_after_exit":
            process = try child()
            state = .starting
            ProbeControl.hold = true
            let pending = Task { @MainActor in await tickHealth() }
            try await waitUntil(seconds: 2) { ProbeControl.continuation != nil }
            try require(ProbeControl.continuation != nil, "probe did not suspend")
            process?.terminate()
            process?.waitUntilExit()
            process = nil
            ProbeControl.continuation?.resume(returning: ProbeControl.result(true))
            await pending.value
            try require(state == .starting, "late probe marked an exited child running: \(state)")

        case "live_status_fallback":
            // A busy server can miss /health while /api/status still answers.
            let live = try child()
            process = live
            state = .unresponsive(pid: live.processIdentifier)
            recordAuxiliaryHealthSuccess()
            for _ in 0..<3 { await tickHealth() }
            try require(state == .running(pid: live.processIdentifier),
                        "recent status did not keep a live child running: \(state)")

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

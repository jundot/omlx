// PR 5 — port-conflict probes used by ServerProcess.start() and the
// health-check loop. Mirrors server_manager.py:224-255.

import Foundation
import Darwin

struct HealthProbeResult: Sendable, Equatable {
    let ok: Bool
    let url: String
    let latencyMs: Int
    let statusCode: Int?
    let errorDescription: String?
}

struct PortConflictResolver: Sendable {
    let host: String
    let port: Int

    private let healthTimeout: TimeInterval = 5

    private var healthURL: URL {
        AppConfig.httpURL(host: host, port: port, path: "/health")!
    }

    // MARK: - Sync probes (cheap; called from start() before spawn)

    /// `connect()` to host:port; if it succeeds, port is in use. Times out
    /// at 1s on connect_refused — fast enough to be sync.
    func isPortInUseSync() -> Bool {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { return false }
        defer { close(fd) }

        // Non-blocking mode via fcntl (FIONBIO isn't exposed by Swift's Darwin).
        let flags = fcntl(fd, F_GETFL, 0)
        _ = fcntl(fd, F_SETFL, flags | O_NONBLOCK)

        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = in_port_t(port).bigEndian
        addr.sin_addr.s_addr = inet_addr(host)
        if addr.sin_addr.s_addr == INADDR_NONE {
            // Non-literal hosts are not expected after AppConfig normalization.
            return false
        }

        let connectResult = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { ptr in
                connect(fd, ptr, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        if connectResult == 0 { return true }
        if errno == EISCONN { return true }
        if errno == EINPROGRESS || errno == EWOULDBLOCK || errno == EALREADY {
            // Wait briefly for the connection to resolve.
            var fdset = fd_set()
            __darwin_fd_zero(&fdset)
            __darwin_fd_set(fd, &fdset)
            var tv = timeval(tv_sec: 1, tv_usec: 0)
            let sel = select(fd + 1, nil, &fdset, nil, &tv)
            if sel <= 0 { return false }
            var soError: Int32 = 0
            var len = socklen_t(MemoryLayout<Int32>.size)
            getsockopt(fd, SOL_SOCKET, SO_ERROR, &soError, &len)
            return soError == 0
        }
        return false   // ECONNREFUSED → free
    }

    /// `lsof -ti :<port> -sTCP:LISTEN`. Sync because it runs at start().
    /// Returns nil if lsof isn't available, the port isn't held, or output
    /// can't be parsed.
    func findOwnerPIDSync() -> pid_t? {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/sbin/lsof")
        proc.arguments = ["-ti", ":\(port)", "-sTCP:LISTEN"]
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = Pipe()
        do {
            try proc.run()
        } catch {
            return nil
        }
        let deadline = Date().addingTimeInterval(2)
        while proc.isRunning && Date() < deadline {
            usleep(50_000)
        }
        if proc.isRunning {
            proc.terminate()
            return nil
        }
        let data = (try? pipe.fileHandleForReading.readToEnd()) ?? Data()
        let text = String(data: data, encoding: .utf8) ?? ""
        let line = text.split(separator: "\n").first.map(String.init) ?? ""
        return pid_t(line.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    /// Sync /health probe: 5s timeout, single shot.
    func isOMLXOnPortSync() -> Bool {
        var req = URLRequest(url: healthURL)
        req.timeoutInterval = healthTimeout
        req.httpMethod = "GET"

        let semaphore = DispatchSemaphore(value: 0)
        let result = SendableBox(false)
        URLSession.shared.dataTask(with: req) { _, response, _ in
            if let http = response as? HTTPURLResponse, http.statusCode == 200 {
                result.value = true
            }
            semaphore.signal()
        }.resume()
        let waitMs = Int((healthTimeout + 1) * 1000)
        _ = semaphore.wait(timeout: .now() + .milliseconds(waitMs))
        return result.value
    }

    // MARK: - Async probes (used by health-check loop)

    func probeHealth() async -> HealthProbeResult {
        let url = healthURL
        let started = Date()
        var req = URLRequest(url: healthURL)
        req.timeoutInterval = healthTimeout
        req.httpMethod = "GET"
        req.cachePolicy = .reloadIgnoringLocalCacheData
        do {
            let (_, response) = try await URLSession.shared.data(for: req)
            let latencyMs = Self.latencyMs(since: started)
            let statusCode = (response as? HTTPURLResponse)?.statusCode
            return HealthProbeResult(
                ok: statusCode == 200,
                url: url.absoluteString,
                latencyMs: latencyMs,
                statusCode: statusCode,
                errorDescription: nil
            )
        } catch {
            return HealthProbeResult(
                ok: false,
                url: url.absoluteString,
                latencyMs: Self.latencyMs(since: started),
                statusCode: nil,
                errorDescription: String(describing: error)
            )
        }
    }

    // MARK: - Kill external owner

    /// SIGTERM, wait up to `timeout` seconds, then SIGKILL.
    /// Returns true if the process is gone afterward.
    @discardableResult
    func killExternal(_ pid: pid_t, timeout: TimeInterval = 5) async -> Bool {
        guard kill(pid, SIGTERM) == 0 else { return false }
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if kill(pid, 0) != 0 { return true }
            try? await Task.sleep(for: .milliseconds(100))
        }
        kill(pid, SIGKILL)
        try? await Task.sleep(for: .milliseconds(500))
        return kill(pid, 0) != 0
    }

    private static func latencyMs(since started: Date) -> Int {
        max(0, Int(Date().timeIntervalSince(started) * 1000))
    }
}

// MARK: - Concurrency-safe scratch box for sync URLSession bridging.

private final class SendableBox<T>: @unchecked Sendable {
    var value: T
    init(_ value: T) { self.value = value }
}

// MARK: - fd_set helpers (stdlib doesn't expose FD_ZERO/FD_SET portably)

private func __darwin_fd_zero(_ set: inout fd_set) {
    set = fd_set()
}

private func __darwin_fd_set(_ fd: Int32, _ set: inout fd_set) {
    let intOffset = Int(fd / 32)
    let bitOffset = fd % 32
    let mask: Int32 = 1 << bitOffset
    withUnsafeMutablePointer(to: &set.fds_bits) { ptr in
        ptr.withMemoryRebound(to: Int32.self, capacity: 32) { arr in
            arr[intOffset] |= mask
        }
    }
}

import Foundation
import Darwin

@MainActor
protocol AppControlHandling: AnyObject {
    func handleAppControl(_ command: AppControlServer.Command) async -> AppControlServer.Response
}

final class AppControlServer: @unchecked Sendable {
    enum Command: String, Sendable {
        case start
        case stop
        case restart
        case status
    }

    struct Response: Encodable, Sendable {
        let ok: Bool
        let status: String
        let state: String
        let pid: Int32?
        let host: String
        let port: Int
        let message: String?

        static func success(
            status: String,
            state: ServerProcess.State,
            server: ServerProcess?,
            message: String? = nil
        ) -> Response {
            Response(
                ok: true,
                status: status,
                state: AppControlServer.describe(state),
                pid: server?.pid,
                host: server?.host ?? "127.0.0.1",
                port: server?.port ?? 8000,
                message: message
            )
        }

        static func failure(
            status: String,
            state: ServerProcess.State,
            server: ServerProcess?,
            message: String
        ) -> Response {
            Response(
                ok: false,
                status: status,
                state: AppControlServer.describe(state),
                pid: server?.pid,
                host: server?.host ?? "127.0.0.1",
                port: server?.port ?? 8000,
                message: message
            )
        }
    }

    private struct Request: Decodable {
        let command: String
    }

    weak var handler: AppControlHandling?
    private let socketURL: URL
    private let queue = DispatchQueue(label: "app.omlx.control")
    private var listenFD: Int32 = -1
    private var running = false

    init(socketURL: URL = AppControlServer.defaultSocketURL()) {
        self.socketURL = socketURL
    }

    deinit {
        stop()
    }

    func start() throws {
        guard !running else { return }
        try FileManager.default.createDirectory(
            at: socketURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try? FileManager.default.setAttributes(
            [.posixPermissions: 0o700],
            ofItemAtPath: socketURL.deletingLastPathComponent().path
        )
        try? FileManager.default.removeItem(at: socketURL)

        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { throw POSIXError(.init(rawValue: errno) ?? .EIO) }

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let path = socketURL.path
        let maxPath = MemoryLayout.size(ofValue: addr.sun_path)
        guard path.utf8.count < maxPath else {
            close(fd)
            throw POSIXError(.ENAMETOOLONG)
        }
        withUnsafeMutableBytes(of: &addr.sun_path) { rawBuffer in
            let raw = rawBuffer.baseAddress!.assumingMemoryBound(to: CChar.self)
            _ = path.withCString { cstr in
                strncpy(raw, cstr, maxPath - 1)
            }
        }

        let bindResult = withUnsafePointer(to: &addr) { ptr -> Int32 in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard bindResult == 0 else {
            let err = errno
            close(fd)
            throw POSIXError(.init(rawValue: err) ?? .EIO)
        }
        chmod(path, S_IRUSR | S_IWUSR)

        guard listen(fd, 8) == 0 else {
            let err = errno
            close(fd)
            throw POSIXError(.init(rawValue: err) ?? .EIO)
        }

        listenFD = fd
        running = true
        queue.async { [weak self] in
            self?.acceptLoop()
        }
    }

    func stop() {
        guard running else { return }
        running = false
        if listenFD >= 0 {
            shutdown(listenFD, SHUT_RDWR)
            close(listenFD)
            listenFD = -1
        }
        try? FileManager.default.removeItem(at: socketURL)
    }

    private func acceptLoop() {
        while running {
            let client = accept(listenFD, nil, nil)
            if client < 0 {
                if running { usleep(50_000) }
                continue
            }
            handle(clientFD: client)
        }
    }

    private func handle(clientFD: Int32) {
        defer { close(clientFD) }
        let data = readRequest(fd: clientFD)
        let response: Response
        do {
            let req = try JSONDecoder().decode(Request.self, from: data)
            guard let command = Command(rawValue: req.command) else {
                response = Response(
                    ok: false,
                    status: "error",
                    state: "unknown",
                    pid: nil,
                    host: "127.0.0.1",
                    port: 8000,
                    message: "Unknown command: \(req.command)"
                )
                writeResponse(response, fd: clientFD)
                return
            }
            let semaphore = DispatchSemaphore(value: 0)
            let box = ResponseBox()
            Task { @MainActor [weak self] in
                if let handler = self?.handler {
                    box.value = await handler.handleAppControl(command)
                } else {
                    box.value = Response(
                        ok: false,
                        status: "error",
                        state: "unknown",
                        pid: nil,
                        host: "127.0.0.1",
                        port: 8000,
                        message: "App control handler unavailable"
                    )
                }
                semaphore.signal()
            }
            _ = semaphore.wait(timeout: .now() + 30)
            response = box.value ?? Response(
                ok: false,
                status: "timeout",
                state: "unknown",
                pid: nil,
                host: "127.0.0.1",
                port: 8000,
                message: "Command timed out"
            )
        } catch {
            response = Response(
                ok: false,
                status: "error",
                state: "unknown",
                pid: nil,
                host: "127.0.0.1",
                port: 8000,
                message: "Invalid request: \(error)"
            )
        }
        writeResponse(response, fd: clientFD)
    }

    private func readRequest(fd: Int32) -> Data {
        var out = Data()
        var buffer = [UInt8](repeating: 0, count: 4096)
        while true {
            let n = read(fd, &buffer, buffer.count)
            if n <= 0 { break }
            if let newline = buffer[..<n].firstIndex(of: 10) {
                out.append(contentsOf: buffer[..<newline])
                break
            }
            out.append(buffer, count: n)
            if out.count > 65536 { break }
        }
        return out
    }

    private func writeResponse(_ response: Response, fd: Int32) {
        guard let data = try? JSONEncoder().encode(response) else { return }
        var bytes = [UInt8](data)
        bytes.append(10)
        _ = bytes.withUnsafeBytes {
            Darwin.write(fd, $0.baseAddress, bytes.count)
        }
    }

    static func defaultSocketURL() -> URL {
        AppConfig.appSupportURL().appendingPathComponent("control.sock")
    }

    static func describe(_ state: ServerProcess.State) -> String {
        switch state {
        case .stopped: return "stopped"
        case .starting: return "starting"
        case .running: return "running"
        case .stopping: return "stopping"
        case .unresponsive: return "unresponsive"
        case .failed: return "failed"
        }
    }
}

private final class ResponseBox: @unchecked Sendable {
    var value: AppControlServer.Response?
}

// oMLX privileged helper — installed once by SMJobBless, then reached over XPC.
// The command surface is deliberately tiny and fixed: one validated wired-limit
// integer. No shell is involved.

import Foundation

@objc(OMLXHelperProtocol)
protocol OMLXHelperProtocol {
    func setWiredLimit(_ mb: Int, withReply reply: @escaping (Bool, String) -> Void)
}

final class OMLXHelper: NSObject, OMLXHelperProtocol {
    func setWiredLimit(_ mb: Int, withReply reply: @escaping (Bool, String) -> Void) {
        guard mb > 0, mb <= 1_048_576 else {
            reply(false, "wired limit must be between 1 and 1048576 MiB")
            return
        }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/sbin/sysctl")
        process.arguments = ["iogpu.wired_limit_mb=\(mb)"]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        do {
            try process.run()
            process.waitUntilExit()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            let message = String(data: data, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            reply(process.terminationStatus == 0, message)
        } catch {
            reply(false, error.localizedDescription)
        }
    }
}

final class ListenerDelegate: NSObject, NSXPCListenerDelegate {
    func listener(
        _ listener: NSXPCListener,
        shouldAcceptNewConnection newConnection: NSXPCConnection
    ) -> Bool {
        newConnection.exportedInterface = NSXPCInterface(with: OMLXHelperProtocol.self)
        newConnection.exportedObject = OMLXHelper()
        newConnection.resume()
        return true
    }
}

let listener = NSXPCListener(machServiceName: "app.omlx.helper")
let delegate = ListenerDelegate()
listener.delegate = delegate
listener.resume()
RunLoop.current.run()

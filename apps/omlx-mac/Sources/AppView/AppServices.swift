// PR 7 — wires AppDelegate-owned runtime objects (ServerProcess, AppConfig)
// to the SwiftUI side. AppView mounts a single instance via `.environmentObject`
// so screens can pull whatever they need without prop drilling. The screens
// keep their own data + polling state in their own view models.
//
// `serverState` republishes ServerProcess.State on every state change so a
// view can `@EnvironmentObject` AppServices and use it as a SwiftUI source
// of truth. ServerProcess itself stays NSNotification-driven (no Combine
// retrofit).

import Foundation
import SwiftUI

@MainActor
final class AppServices: NSObject, ObservableObject {
    @Published var config: AppConfig
    @Published var serverState: ServerProcess.State = .stopped

    let client: OMLXClient
    let updates: UpdateController

    private weak var server: ServerProcess?

    init(config: AppConfig = .default, server: ServerProcess? = nil) {
        self.config = config
        self.client = OMLXClient(host: config.host, port: config.port, apiKey: config.apiKey)
        self.updates = UpdateController()
        super.init()
        self.bind(server: server)
    }

    func bind(server: ServerProcess?) {
        // Detach from the previous server (if any) before re-attaching.
        if self.server != nil {
            NotificationCenter.default.removeObserver(
                self,
                name: ServerProcess.stateDidChangeNotification,
                object: nil
            )
        }
        self.server = server
        if let server {
            self.serverState = server.state
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(serverStateDidChange(_:)),
                name: ServerProcess.stateDidChangeNotification,
                object: server
            )
        }
    }

    @objc private func serverStateDidChange(_ note: Notification) {
        guard let proc = note.object as? ServerProcess, proc === server else { return }
        // ServerProcess posts on the main queue (via DispatchQueue.main.async
        // in terminationHandler / @MainActor health-check Task), so we're
        // already on the main thread here.
        serverState = proc.state
    }

    func updateConfig(_ next: AppConfig) {
        self.config = next
        client.configure(host: next.host, port: next.port, apiKey: next.apiKey)
    }

    // MARK: - Server lifecycle (proxied to ServerProcess)

    var hasServer: Bool { server != nil }

    @discardableResult
    func startServer() throws -> ServerProcess.StartResult? {
        try server?.start()
    }

    func stopServer() async {
        await server?.stop()
    }

    func restartServer() async throws {
        await server?.stop()
        _ = try server?.start()
    }

    func forceRestartServer() async throws {
        _ = try await server?.forceRestart()
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
    }
}

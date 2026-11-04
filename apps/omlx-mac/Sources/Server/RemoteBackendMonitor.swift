// RemoteBackendMonitor — the app's view of an oMLX server it does not run.
//
// When the install points at another Mac (`AppConfig.isRemote`), there is no
// `ServerProcess` to observe, so this polls the remote server's `/health`
// the way the managed process's health loop does and publishes a state the
// Server screen, the menubar and the control socket can render. It never
// starts or stops anything: the remote server is managed on its own Mac.

import Foundation

@MainActor
final class RemoteBackendMonitor {
    enum State: Equatable, Sendable {
        /// Not polling yet.
        case idle
        /// `remote.url` in settings.json is empty or not an address.
        case invalidAddress
        /// First probe in flight.
        case checking
        case reachable(latencyMs: Int)
        /// `/health` answers but the key does not open the API.
        case unauthorized
        case unreachable(message: String)

        var isReachable: Bool {
            if case .reachable = self { return true }
            return false
        }
    }

    static let stateDidChangeNotification = Notification.Name("RemoteBackendMonitor.stateDidChange")
    static let pollInterval: Duration = .seconds(5)

    let endpoint: OMLXEndpoint?
    private(set) var state: State = .idle {
        didSet {
            guard state != oldValue else { return }
            NotificationCenter.default.post(name: Self.stateDidChangeNotification, object: self)
        }
    }
    private var pollTask: Task<Void, Never>?

    init(endpoint: OMLXEndpoint?) {
        self.endpoint = endpoint
    }

    func start() {
        guard pollTask == nil else { return }
        guard let endpoint else {
            state = .invalidAddress
            return
        }
        state = .checking
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                let next = await Self.probe(endpoint)
                guard let self, !Task.isCancelled else { return }
                self.state = next
                try? await Task.sleep(for: Self.pollInterval)
            }
        }
    }

    func stop() {
        pollTask?.cancel()
        pollTask = nil
        state = .idle
    }

    /// Re-run the probe now instead of waiting for the next tick.
    func refresh() async {
        guard let endpoint else { return }
        state = await Self.probe(endpoint)
    }

    /// `/health` first (reachability), then `/v1/models/status` with the key
    /// (authorization). The managed process uses the same health probe.
    static func probe(_ endpoint: OMLXEndpoint) async -> State {
        let latency: Int
        do {
            latency = try await endpoint.checkHealth()
        } catch {
            return .unreachable(message: error.omlxDescription)
        }
        do {
            _ = try await endpoint.models()
        } catch OMLXClientError.unauthenticated {
            return .unauthorized
        } catch OMLXClientError.http(let status, _) where status == 401 || status == 403 {
            return .unauthorized
        } catch {
            // The server is up; a model-list hiccup is not an outage.
            return .reachable(latencyMs: latency)
        }
        return .reachable(latencyMs: latency)
    }
}

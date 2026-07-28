import SwiftUI

@MainActor
@Observable
final class ContextBenchScreenVM {
    // Form state — defaults mirror the HTML admin panel.
    var selectedModelId: String = ""
    var targetTokens: Int = 131072

    // Server state
    private(set) var models: [ModelDTO] = []
    private(set) var running: Bool = false
    private(set) var phase: String = ""
    private(set) var progress: Double = 0
    private(set) var message: String = ""
    private(set) var result: ContextBenchResultDTO?
    private(set) var currentBenchId: String?
    var lastError: String?

    @ObservationIgnored
    private weak var client: OMLXClient?
    @ObservationIgnored
    private var pollTask: Task<Void, Never>?

    /// Server-validated whitelist for `target_tokens`.
    static let targetOptions: [Int] = [16384, 32768, 65536, 131072, 262144, 524288]

    // MARK: Derived

    var canRun: Bool {
        !selectedModelId.isEmpty && !running
    }

    // MARK: Lifecycle

    /// Idempotent: refreshes the model list every time the screen appears
    /// but never touches the running-bench state or poll task, so an
    /// in-flight bench survives navigating away and back.
    func start(client: OMLXClient) async {
        self.client = client
        await loadModels()
    }

    /// Manual teardown hook for future disconnect flows — not wired to
    /// `.onDisappear` (the bench survives screen unloads).
    func stop() {
        pollTask?.cancel()
        pollTask = nil
    }

    // MARK: Loaders

    private func loadModels() async {
        guard let client else { return }
        do {
            let resp = try await client.listModels()
            // Loaded-first, then case-insensitive by id (accuracy pattern),
            // excluding virtual entries that cannot be benchmarked.
            self.models = resp.models
                .filter { !($0.virtual ?? false) }
                .sorted { a, b in
                    if a.loaded != b.loaded { return a.loaded && !b.loaded }
                    return a.id.localizedCaseInsensitiveCompare(b.id) == .orderedAscending
                }
        } catch {
            self.lastError = error.omlxDescription
        }
    }

    // MARK: Actions

    func runBenchmark(client: OMLXClient) {
        guard canRun else { return }
        let body = ContextBenchStartRequest(
            modelId: selectedModelId,
            targetTokens: targetTokens
        )
        result = nil
        lastError = nil
        phase = ""
        progress = 0
        message = ""
        running = true

        Task { [weak self] in
            do {
                let resp = try await client.startContextBench(body)
                await MainActor.run {
                    guard let self else { return }
                    self.currentBenchId = resp.benchId
                    self.pollStatus(client: client)
                }
            } catch {
                await MainActor.run {
                    guard let self else { return }
                    self.running = false
                    self.lastError = error.omlxDescription
                }
            }
        }
    }

    func cancelBenchmark(client: OMLXClient) {
        guard let benchId = currentBenchId else {
            running = false
            return
        }
        Task { [weak self] in
            do {
                _ = try await client.cancelContextBench(benchId: benchId)
            } catch {
                await MainActor.run { self?.lastError = error.omlxDescription }
            }
            // Keep the poll alive: the server flips status to "cancelled"
            // once its cleanup (model unload) finishes, and the poll loop
            // shuts the UI down from that terminal status.
        }
    }

    // MARK: Polling

    /// 1.5 s poll of GET /api/bench/context/{id}/results while running.
    /// The endpoint mirrors phase / progress / message server-side, so no
    /// SSE parsing is needed.
    private func pollStatus(client: OMLXClient) {
        pollTask?.cancel()
        guard let benchId = currentBenchId else { return }
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                do {
                    let resp = try await client.getContextBenchStatus(benchId: benchId)
                    let terminal = await MainActor.run { () -> Bool in
                        self.phase = resp.phase
                        self.progress = resp.progress
                        self.message = resp.message
                        if let r = resp.result {
                            self.result = r
                        }
                        if let err = resp.error, !err.isEmpty {
                            self.lastError = err
                        }
                        let status = resp.status.lowercased()
                        let done = (status == "completed"
                                    || status == "cancelled"
                                    || status == "error")
                        if done {
                            self.running = false
                        }
                        return done
                    }
                    if terminal { return }
                } catch {
                    // Transient failures (server restart, dropped socket)
                    // shouldn't kill the poll — log and try again.
                    await MainActor.run {
                        self.lastError = error.omlxDescription
                    }
                }
                try? await Task.sleep(for: .seconds(1.5))
            }
        }
    }
}

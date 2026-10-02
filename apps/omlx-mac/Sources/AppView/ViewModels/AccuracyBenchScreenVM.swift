import SwiftUI

@MainActor
@Observable
final class AccuracyBenchScreenVM {
    // Form state
    var selectedModelId: String = ""
    var selectedBenchmarks: Set<String> = []
    var sampleSizes: [String: Int] = [:]
    var batchSize: Int = 4
    var enableThinking: Bool = false

    // Server state
    private(set) var models: [ModelDTO] = []
    private(set) var status: AccuracyQueueStatus?
    private(set) var results: [AccuracyResultDTO] = []

    // UI state
    private(set) var isAdding: Bool = false
    var lastError: String?

    @ObservationIgnored
    private weak var client: OMLXClient?
    @ObservationIgnored
    private var pollTask: Task<Void, Never>?

    var canSubmit: Bool {
        !selectedModelId.isEmpty && !selectedBenchmarks.isEmpty
    }

    // MARK: Lifecycle

    func start(client: OMLXClient) async {
        self.client = client
        await loadModels()
        await pollOnce()
        startPolling()
    }

    func stop() {
        pollTask?.cancel()
        pollTask = nil
    }

    // MARK: Loaders

    private func loadModels() async {
        guard let client else { return }
        do {
            let resp = try await client.listModels()
            self.models = resp.models
        } catch {
            self.lastError = String(localized: "bench.accuracy.error.load_models",
                                    defaultValue: "Failed to load models: \(error.omlxDescription)",
                                    comment: "Accuracy Bench error when listing models fails; placeholder is the underlying error description")
        }
    }

    private func pollOnce() async {
        guard let client else { return }
        // Status and results are independent endpoints — fan them out so a
        // slow one doesn't block the other.
        async let statusFetch = client.getAccuracyQueueStatus()
        async let resultsFetch = client.listAccuracyResults()
        do {
            let s = try await statusFetch
            self.status = s
        } catch {
            // Status failures are transient — keep the previous snapshot so
            // the queue/running row doesn't flicker out during a hiccup.
        }
        do {
            let r = try await resultsFetch
            self.results = r.results
        } catch {
            // Same logic — last-known results stay visible.
        }
    }

    // MARK: Polling

    private func startPolling() {
        pollTask?.cancel()
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                let active = await MainActor.run { () -> Bool in
                    let running = self.status?.running == true
                    let queued = (self.status?.queue.isEmpty == false)
                    return running || queued
                }
                // Fast 2 s cadence while work is in flight; idle 8 s otherwise.
                try? await Task.sleep(for: .seconds(active ? 2 : 8))
                if Task.isCancelled { return }
                await self.pollOnce()
            }
        }
    }

    // MARK: Actions

    func addToQueue(client: OMLXClient) {
        guard canSubmit, !isAdding else { return }
        // Snapshot form state — the user can keep editing while the request
        // is in flight; we want the version they confirmed.
        let modelId = selectedModelId
        let benchmarks: [String: Int] = Dictionary(
            uniqueKeysWithValues: selectedBenchmarks.map { key in
                (key, sampleSizes[key] ?? 100)
            }
        )
        let body = AccuracyQueueAddRequest(
            modelId: modelId,
            benchmarks: benchmarks,
            batchSize: batchSize,
            enableThinking: enableThinking
        )
        isAdding = true
        lastError = nil
        Task { [weak self] in
            defer { Task { @MainActor [weak self] in self?.isAdding = false } }
            do {
                let s = try await client.addAccuracyQueue(body)
                await MainActor.run {
                    guard let self else { return }
                    self.status = s
                    // Reset selection on success so the user can stage another
                    // run without manually clearing the grid.
                    self.selectedBenchmarks = []
                }
                await self?.pollOnce()
            } catch {
                await MainActor.run {
                    self?.lastError = String(localized: "bench.accuracy.error.add_queue",
                                             defaultValue: "Failed to add to queue: \(error.omlxDescription)",
                                             comment: "Accuracy Bench error when adding to queue fails; placeholder is the underlying error")
                }
            }
        }
    }

    func removeFromQueue(client: OMLXClient, index: Int) {
        Task { [weak self] in
            do {
                let s = try await client.removeAccuracyQueue(index: index)
                await MainActor.run { self?.status = s }
            } catch {
                await MainActor.run {
                    self?.lastError = String(localized: "bench.accuracy.error.remove",
                                             defaultValue: "Failed to remove: \(error.omlxDescription)",
                                             comment: "Accuracy Bench error when removing a queue entry fails; placeholder is the underlying error")
                }
            }
        }
    }

    func cancelRunning(client: OMLXClient) {
        Task { [weak self] in
            do {
                _ = try await client.cancelAccuracyBench()
                await self?.pollOnce()
            } catch {
                await MainActor.run {
                    self?.lastError = String(localized: "bench.accuracy.error.cancel",
                                             defaultValue: "Failed to cancel: \(error.omlxDescription)",
                                             comment: "Accuracy Bench error when cancelling the running bench fails; placeholder is the underlying error")
                }
            }
        }
    }

    func resetResults(client: OMLXClient) {
        Task { [weak self] in
            do {
                _ = try await client.resetAccuracyResults()
                await MainActor.run { self?.results = [] }
                await self?.pollOnce()
            } catch {
                await MainActor.run {
                    self?.lastError = String(localized: "bench.accuracy.error.clear_results",
                                             defaultValue: "Failed to clear results: \(error.omlxDescription)",
                                             comment: "Accuracy Bench error when clearing accumulated results fails; placeholder is the underlying error")
                }
            }
        }
    }

}

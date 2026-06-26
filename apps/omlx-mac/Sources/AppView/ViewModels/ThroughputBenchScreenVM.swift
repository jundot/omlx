import SwiftUI

@MainActor
@Observable
final class ThroughputBenchScreenVM {
    // Form state — defaults mirror the HTML admin panel's pre-ticked options.
    var selectedModelId: String = ""
    var promptLengths: Set<Int> = [4096, 16384]
    var genLength: String = "128"
    var batchSizes: Set<Int> = [2, 4]
    var exportOpen: Bool = false

    // Server state
    private(set) var models: [ModelDTO] = []
    private(set) var device: DeviceInfoDTO?
    private(set) var running: Bool = false
    private(set) var singleResults: [BenchResultDTO] = []
    private(set) var batchResults: [BenchResultDTO] = []
    private(set) var currentBenchId: String?
    /// Server-side upload-to-leaderboard state, populated after the
    /// bench completes. Phases: "idle" (not yet started, or no upload
    /// because of experimental features detected later in the run) →
    /// "uploading" → "done" | "skipped". The poll loop keeps running
    /// past `status=completed` until this reaches a terminal phase so
    /// the user sees the leaderboard URL light up without manually
    /// refreshing.
    private(set) var uploadState: BenchUploadStateDTO?
    var lastError: String?

    @ObservationIgnored
    private weak var client: OMLXClient?
    @ObservationIgnored
    private var pollTask: Task<Void, Never>?
    /// Counts poll iterations spent waiting for the upload phase to
    /// terminate after the bench itself completes. Reset on each new
    /// run; capped at 120 (i.e. 2 min at 1 Hz) so a wedged upload
    /// doesn't hold the poll loop hostage forever.
    @ObservationIgnored
    private var postCompleteTicks: Int = 0

    // MARK: Derived

    var canRun: Bool {
        !selectedModelId.isEmpty
            && !running
            && !promptLengths.isEmpty
            && !batchSizes.isEmpty
            && (Int(genLength) ?? 0) > 0
    }

    /// Synthetic 1× baseline for the Batch Results table: the first single
    /// trial whose pp == 1024 (matches the JS admin panel's behaviour).
    var batchBaseline: BenchResultDTO? {
        singleResults.first(where: { $0.pp == 1024 })
            ?? singleResults.first
    }

    /// Monospaced two-table dump used by the Text export card.
    var exportText: String {
        var lines: [String] = []
        if !singleResults.isEmpty {
            lines.append("# Single request results")
            lines.append(
                ["Test", "TTFT(ms)", "TPOT(ms)", "ppTPS", "tgTPS",
                 "E2E(s)", "Throughput", "PeakMem"]
                    .joined(separator: "\t")
            )
            for r in singleResults {
                lines.append([
                    "pp \(r.pp ?? 0) / tg \(r.tg ?? 0)",
                    format1(r.ttftMs),
                    format1(r.tpotMs),
                    format1(r.processingTps),
                    format1(r.genTps),
                    format1(r.e2eLatencyS),
                    format1(r.totalThroughput),
                    formatPeakMem(r.peakMemoryBytes),
                ].joined(separator: "\t"))
            }
            lines.append("")
        }
        if !batchResults.isEmpty {
            lines.append("# Batch results")
            lines.append(
                ["Batch", "tgTPS", "ppTPS", "avgTTFT(ms)", "E2E(s)", "Speedup"]
                    .joined(separator: "\t")
            )
            let baselineTps = batchBaseline?.genTps ?? 0
            if let baseline = batchBaseline {
                lines.append([
                    "1x baseline",
                    format1(baseline.genTps),
                    format1(baseline.processingTps),
                    format1(baseline.ttftMs),
                    format1(baseline.e2eLatencyS),
                    "1.00x",
                ].joined(separator: "\t"))
            }
            for r in batchResults {
                let speedup: String = {
                    guard baselineTps > 0, let tg = r.tgTps else { return "—" }
                    return String(format: "%.2fx", tg / baselineTps)
                }()
                lines.append([
                    "\(r.batchSize ?? 0)x",
                    format1(r.tgTps),
                    format1(r.ppTps),
                    format1(r.avgTtftMs),
                    format1(r.e2eLatencyS),
                    speedup,
                ].joined(separator: "\t"))
            }
        }
        return lines.isEmpty ? "No results yet." : lines.joined(separator: "\n")
    }

    // MARK: Lifecycle

    /// Idempotent: called every time the screen appears. Refreshes the
    /// model + device lists (cheap, ~ms) but never touches the
    /// running-bench state, results table, or poll task. If the user
    /// navigated away during a run, the same poll task is still alive
    /// updating these observable properties — coming back just
    /// re-subscribes via SwiftUI's diffing.
    func start(client: OMLXClient) async {
        self.client = client
        await loadModels()
        await loadDevice()
    }

    /// Manually tear down the poll task. Not wired to the screen's
    /// `.onDisappear` — the bench survives screen unloads. Kept around
    /// for future "logout / disconnect" flows where the long-lived VM
    /// needs to be reset.
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
            // Surface so the user can recover; polling does not depend on this.
            self.lastError = error.omlxDescription
        }
    }

    private func loadDevice() async {
        guard let client else { return }
        do {
            self.device = try await client.getDeviceInfo()
        } catch {
            // Device chip is a "nice to have" — hide silently on failure
            // so a missing /api/device-info doesn't block running the bench.
            self.device = nil
        }
    }

    // MARK: Actions

    func runBenchmark(client: OMLXClient) {
        guard canRun else { return }
        let body = BenchStartRequest(
            modelId: selectedModelId,
            promptLengths: promptLengths.sorted(),
            generationLength: Int(genLength) ?? 128,
            batchSizes: batchSizes.sorted()
        )
        // Wipe the previous run's tables so a new run doesn't accumulate
        // across unrelated configurations.
        singleResults = []
        batchResults = []
        uploadState = nil
        postCompleteTicks = 0
        lastError = nil
        running = true

        Task { [weak self] in
            do {
                let resp = try await client.startThroughputBench(body)
                await MainActor.run {
                    guard let self else { return }
                    self.currentBenchId = resp.benchId
                    self.pollResults(client: client)
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
            // Nothing to cancel server-side — flip the UI back regardless
            // so we don't strand the screen in "Running…" forever.
            running = false
            return
        }
        Task { [weak self] in
            do {
                _ = try await client.cancelBench(benchId: benchId)
            } catch {
                await MainActor.run { self?.lastError = error.omlxDescription }
            }
            await MainActor.run {
                self?.running = false
                self?.pollTask?.cancel()
                self?.pollTask = nil
            }
        }
    }

    // MARK: Polling

    /// 1 Hz poll of GET /api/bench/{id}/results while running. Server
    /// returns the full `results` array — we append-dedupe per call so
    /// the in-progress tables don't flicker as new rows arrive.
    private func pollResults(client: OMLXClient) {
        pollTask?.cancel()
        guard let benchId = currentBenchId else { return }
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                do {
                    let resp = try await client.getBenchResults(benchId: benchId)
                    await MainActor.run {
                        self.absorb(results: resp.results)
                        if let err = resp.error, !err.isEmpty {
                            self.lastError = err
                        }
                        if let upload = resp.uploadState {
                            self.uploadState = upload
                        }
                        let status = resp.status.lowercased()
                        let terminal = (status == "completed"
                                        || status == "failed"
                                        || status == "cancelled")
                        if terminal {
                            self.running = false
                        }
                    }
                    // Keep polling past `status=completed` until the upload
                    // phase also terminates ("done" | "skipped"). The
                    // backend writes upload state on the same BenchmarkRun
                    // (benchmark.py:_upload_to_omlx_ai) and surfaces it
                    // via /results, so this is just one more tick or two.
                    // Cap with a 120 s safety net so a stuck upload
                    // doesn't keep the poll alive forever.
                    let (stillRunning, uploadDone, hitCap) = await MainActor.run {
                        () -> (Bool, Bool, Bool) in
                        let phase = self.uploadState?.phase ?? "idle"
                        let isTerminal = (phase == "done" || phase == "skipped")
                        self.postCompleteTicks += self.running ? 0 : 1
                        return (self.running, isTerminal,
                                self.postCompleteTicks >= 120)
                    }
                    if !stillRunning && (uploadDone || hitCap) {
                        await MainActor.run { self.postCompleteTicks = 0 }
                        return
                    }
                } catch {
                    // Transient failures (server restart, dropped socket)
                    // shouldn't kill the poll — log and try again.
                    await MainActor.run {
                        self.lastError = error.omlxDescription
                    }
                }
                try? await Task.sleep(for: .seconds(1))
            }
        }
    }

    /// Split the server's flat `results` array into single / batch buckets
    /// and merge against what we already have. Dedupe key:
    ///   • single  → "single::pp::tg"
    ///   • batch   → "batch::batchSize"
    /// Mirrors the JS panel: rows are unique per (testType, key).
    private func absorb(results: [BenchResultDTO]) {
        var singles: [BenchResultDTO] = []
        var batches: [BenchResultDTO] = []
        var seen = Set<String>()
        for r in results {
            switch r.testType {
            case "single":
                let key = "single::\(r.pp ?? 0)::\(r.tg ?? 0)"
                if seen.insert(key).inserted { singles.append(r) }
            case "batch":
                let key = "batch::\(r.batchSize ?? 0)"
                if seen.insert(key).inserted { batches.append(r) }
            default:
                continue
            }
        }
        // Sort for stable presentation regardless of arrival order.
        self.singleResults = singles.sorted { ($0.pp ?? 0, $0.tg ?? 0) < ($1.pp ?? 0, $1.tg ?? 0) }
        self.batchResults = batches.sorted { ($0.batchSize ?? 0) < ($1.batchSize ?? 0) }
    }

    private func format1(_ value: Double?) -> String {
        guard let v = value else { return "—" }
        return String(format: "%.1f", v)
    }

    private func formatPeakMem(_ bytes: Int64?) -> String {
        guard let b = bytes, b > 0 else { return "—" }
        return formatBytes(b)
    }

}

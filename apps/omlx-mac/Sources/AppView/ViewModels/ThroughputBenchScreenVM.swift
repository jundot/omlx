import SwiftUI

extension BenchmarkContextProfile {
    var localizedLabel: String {
        switch self {
        case .codePython:
            String(localized: "bench.throughput.context.code_python",
                   defaultValue: "Code (Python)",
                   comment: "Python code context profile for the throughput benchmark")
        case .codeMixed:
            String(localized: "bench.throughput.context.code_mixed",
                   defaultValue: "Code (Mixed)",
                   comment: "Mixed-language code context profile for the throughput benchmark")
        case .novelKorean:
            String(localized: "bench.throughput.context.novel_ko",
                   defaultValue: "Novel (Korean)",
                   comment: "Korean novel context profile for the throughput benchmark")
        case .novelEnglish:
            String(localized: "bench.throughput.context.novel_en",
                   defaultValue: "Novel (English)",
                   comment: "English novel context profile for the throughput benchmark")
        case .novelJapanese:
            String(localized: "bench.throughput.context.novel_ja",
                   defaultValue: "Novel (Japanese)",
                   comment: "Japanese novel context profile for the throughput benchmark")
        }
    }
}

@MainActor
@Observable
final class ThroughputBenchScreenVM {
    // Form state — defaults mirror the HTML admin panel's pre-ticked options.
    var selectedModelId: String = ""
    var contextProfile: BenchmarkContextProfile = .codePython
    var warmupMode: BenchmarkWarmupMode = .quick
    var alignPromptToAne: Bool = false
    var promptLengths: Set<Int> = [1024, 4096]
    var genLength: String = "128"
    var batchSizes: Set<Int> = [2, 4]
    var forceLmEngine: Bool = false
    var externalEnabled: Bool = false
    var externalBaseUrl: String = ""
    var externalApiKey: String = ""
    var externalModel: String = ""
    var exportOpen: Bool = false

    // Keep credentials out of the snapshot used to identify results and exports.
    private var runTarget: (modelId: String, baseUrl: String?, forceLmEngine: Bool,
                            contextProfile: BenchmarkContextProfile)?

    // Server state
    private(set) var models: [ModelDTO] = []
    private(set) var device: DeviceInfoDTO?
    private(set) var running: Bool = false
    private(set) var singleResults: [BenchResultDTO] = []
    private(set) var batchResults: [BenchResultDTO] = []
    private(set) var currentBenchId: String?
    /// Server-side upload-to-leaderboard state, populated after the
    /// bench completes. Phases: "idle" (not yet started) → "uploading" →
    /// "done" | "skipped". Only external-endpoint runs skip now;
    /// accelerated runs upload with their flags attached. The poll loop
    /// keeps running
    /// past `status=completed` until this reaches a terminal phase so
    /// the user sees the leaderboard URL light up without manually
    /// refreshing.
    private(set) var uploadState: BenchUploadStateDTO?
    var lastError: String?

    @ObservationIgnored
    private weak var client: OMLXClient?
    @ObservationIgnored
    private var pollTask: Task<Void, Never>?
    @ObservationIgnored
    private var cancelRequested = false

    // MARK: Derived

    var canRun: Bool {
        !running && startRequest != nil
    }

    var startRequest: BenchStartRequest? {
        guard !promptLengths.isEmpty, !batchSizes.isEmpty,
              let generationLength = Int(genLength), generationLength > 0 else { return nil }
        let external: BenchExternalEndpoint?
        if externalEnabled {
            let baseUrl = externalBaseUrl.trimmingCharacters(in: .whitespacesAndNewlines)
            let model = externalModel.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let url = URL(string: baseUrl),
                  ["http", "https"].contains(url.scheme ?? ""),
                  let host = url.host, !host.isEmpty, !model.isEmpty else { return nil }
            external = BenchExternalEndpoint(baseUrl: baseUrl, apiKey: externalApiKey, model: model)
        } else {
            guard !selectedModelId.isEmpty else { return nil }
            external = nil
        }
        return BenchStartRequest(
            modelId: external?.model ?? selectedModelId,
            contextProfile: contextProfile,
            warmupMode: warmupMode,
            alignPromptToAne: alignPromptToAne,
            promptLengths: promptLengths.sorted(),
            generationLength: generationLength,
            batchSizes: batchSizes.sorted(),
            forceLmEngine: externalEnabled ? false : forceLmEngine,
            external: external
        )
    }

    var runDescription: String? {
        guard let target = runTarget else { return nil }
        let engine: String
        if target.baseUrl != nil {
            engine = String(localized: "bench.throughput.engine.external",
                            defaultValue: "External OpenAI API",
                            comment: "Engine label identifying an external throughput benchmark result")
        } else if target.forceLmEngine {
            engine = String(localized: "bench.throughput.engine.forced",
                            defaultValue: "Forced mlx-lm",
                            comment: "Engine label identifying a throughput benchmark with force mlx-lm enabled")
        } else {
            engine = String(localized: "bench.throughput.engine.auto",
                            defaultValue: "Automatic engine",
                            comment: "Engine label identifying a throughput benchmark using automatic engine selection")
        }
        return [target.modelId, target.baseUrl, engine].compactMap { $0 }.joined(separator: " · ")
    }

    /// Acceleration features the selected model has enabled, so the user knows
    /// before starting that the run will be tagged on the leaderboard.
    /// Derived from settings already fetched by `loadModels()` — no extra
    /// request. Mirrors `_FEATURE_FLAG_SPECS` in omlx/admin/benchmark.py.
    var pendingFeatureFlags: [String] {
        guard !externalEnabled,
              let s = models.first(where: { $0.id == selectedModelId })?.settings else {
            return []
        }
        var flags: [String] = []
        if s.dflashEnabled == true { flags.append("DFlash") }
        if s.specprefillEnabled == true { flags.append("SpecPrefill") }
        if s.turboquantKvEnabled == true {
            let scheme = s.turboquantKvScheme ?? "turboquant"
            let name = scheme == "affine4"
                ? "Affine4 KV" : scheme == "affine8" ? "Affine8 KV" : "TurboQuant KV"
            if let bits = s.turboquantKvBits {
                let text = bits == bits.rounded()
                    ? String(Int(bits))
                    : String(format: "%g", bits)
                flags.append("\(name) \(text)-bit")
            } else {
                flags.append(name)
            }
        }
        if s.mtpEnabled == true { flags.append("Lightning MTP") }
        if s.vlmMtpEnabled == true { flags.append("VLM MTP") }
        return flags
    }

    /// Synthetic 1× baseline for the Batch Results table: the first single
    /// trial whose pp == 1024 (matches the JS admin panel's behaviour).
    var batchBaseline: BenchResultDTO? {
        singleResults.first(where: { $0.pp == 1024 })
            ?? singleResults.first
    }

    /// Monospaced two-table dump used by the Text export card.
    var exportText: String {
        var lines = ["# Context: \((runTarget?.contextProfile ?? contextProfile).localizedLabel)"]
        if let runDescription { lines.append("# \(runDescription)") }
        lines.append("")
        if !singleResults.isEmpty {
            lines.append("# Single request results")
            lines.append(
                ["Test", "TTFT(ms)", "TPOT(ms)", "PP TPS", "TG TPS",
                 "E2E(s)", "Throughput", "PeakMem"]
                    .joined(separator: "\t")
            )
            for r in singleResults {
                lines.append([
                    "PP \(r.pp ?? 0) / TG \(r.tg ?? 0)",
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
                ["Batch", "TG TPS", "PP TPS", "avgTTFT(ms)", "E2E(s)", "Speedup"]
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
        guard !running, let body = startRequest else { return }
        stop()
        currentBenchId = nil
        cancelRequested = false
        var endpoint = body.external.flatMap { URLComponents(string: $0.baseUrl) }
        endpoint?.user = nil
        endpoint?.password = nil
        endpoint?.query = nil
        endpoint?.fragment = nil
        runTarget = (body.modelId, endpoint?.string, body.forceLmEngine, body.contextProfile)
        // Wipe the previous run's tables so a new run doesn't accumulate
        // across unrelated configurations.
        singleResults = []
        batchResults = []
        uploadState = nil
        lastError = nil
        running = true

        Task { [weak self] in
            do {
                let resp = try await client.startThroughputBench(body)
                guard let self else { return }
                self.currentBenchId = resp.benchId
                self.pollResults(client: client)
                if self.cancelRequested { self.cancelBenchmark(client: client) }
            } catch {
                self?.running = false
                self?.lastError = error.omlxDescription
            }
        }
    }

    func cancelBenchmark(client: OMLXClient) {
        guard running else { return }
        cancelRequested = true
        // A pending start must supply its ID before we can cancel the server run.
        guard let benchId = currentBenchId else { return }
        Task { [weak self] in
            do {
                _ = try await client.cancelBench(benchId: benchId)
                guard let self, self.currentBenchId == benchId else { return }
                self.running = false
                self.lastError = nil
                self.stop()
            } catch {
                guard let self, self.currentBenchId == benchId else { return }
                self.lastError = error.omlxDescription
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
            var postCompleteTicks = 0
            while !Task.isCancelled {
                guard let self else { return }
                do {
                    let resp = try await client.getBenchResults(benchId: benchId)
                    guard !Task.isCancelled, self.currentBenchId == benchId else { return }
                    self.absorb(results: resp.results)
                    if let err = resp.error, !err.isEmpty {
                        self.lastError = err
                    }
                    if let upload = resp.uploadState {
                        self.uploadState = upload
                    }
                    switch resp.status.lowercased() {
                    case "error", "failed", "cancelled":
                        self.running = false
                        return
                    case "completed":
                        self.running = false
                        // Wait for uploads, capped at 120 completed polls (about two minutes).
                        postCompleteTicks += 1
                        let phase = self.uploadState?.phase ?? "idle"
                        if phase == "done" || phase == "skipped" || postCompleteTicks >= 120 {
                            return
                        }
                    default:
                        break
                    }
                } catch {
                    // Transient failures (server restart, dropped socket)
                    // shouldn't kill the poll — log and try again.
                    guard !Task.isCancelled, self.currentBenchId == benchId else { return }
                    self.lastError = error.omlxDescription
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

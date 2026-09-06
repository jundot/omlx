import SwiftUI

@MainActor
@Observable
final class StorageRooflineScreenVM {
    // Form state
    var selectedModelId: String = ""
    var tokPerCycleText: String = "1.79"
    var verifyMultText: String = "2.3"
    var measuredBaseText: String = ""

    // Server state
    private(set) var models: [ModelDTO] = []
    private(set) var running: Bool = false
    private(set) var currentJobId: String?
    private(set) var progressPhase: String = ""
    private(set) var progressDone: Int = 0
    private(set) var progressTotal: Int = 0
    /// Completed-run report (also repopulated on screen load from the
    /// predict endpoint so past runs survive navigation/restart).
    private(set) var report: StorageRooflineReportDTO?
    /// Derived verdict params for the selected model (auto-params endpoint).
    private(set) var autoParams: StorageAutoParamsDTO?
    /// Recent measurement reports (newest first) with delta rendering.
    private(set) var history: [OMLXClient.StorageHistoryEntry] = []
    var lastError: String?

    @ObservationIgnored
    private weak var client: OMLXClient?
    @ObservationIgnored
    private var pollTask: Task<Void, Never>?

    // MARK: Derived

    var parsedTokPerCycle: Double? {
        Double(tokPerCycleText.trimmingCharacters(in: .whitespaces))
    }

    var parsedVerifyMult: Double? {
        Double(verifyMultText.trimmingCharacters(in: .whitespaces))
    }

    var parsedMeasuredBase: Double? {
        let t = measuredBaseText.trimmingCharacters(in: .whitespaces)
        return t.isEmpty ? nil : Double(t)
    }

    var canRun: Bool {
        !selectedModelId.isEmpty
            && !running
            && (parsedTokPerCycle ?? 0) > 0
            && (parsedVerifyMult ?? 0) > 0
    }

    var canPredict: Bool {
        !selectedModelId.isEmpty && !running
    }

    // MARK: Lifecycle

    func start(client: OMLXClient) async {
        self.client = client
        await loadModels()
        // Auto-fill the verdict params from the model's derived values so
        // the first prediction needs no manual input; the user can still
        // edit them afterwards (which flips to explicit params).
        if !selectedModelId.isEmpty {
            await loadAutoParams()
            await predict()
        }
        await loadHistory()
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
            if selectedModelId.isEmpty, let first = resp.models.first {
                selectedModelId = first.id
            }
        } catch {
            self.lastError = error.omlxDescription
        }
    }

    @MainActor
    func loadHistory() async {
        guard let client else { return }
        do {
            history = try await client.getStorageHistory()
        } catch {
            // History is a nice-to-have; never surface as an error.
        }
    }

    @MainActor
    func loadAutoParams() async {
        guard let client, !selectedModelId.isEmpty else { return }
        do {
            let auto = try await client.getStorageAutoParams(modelId: selectedModelId)
            self.autoParams = auto
            if auto.available == true {
                if let tok = auto.tokPerCycle {
                    tokPerCycleText = String(format: "%.2f", tok)
                }
                if let mult = auto.verifyByteMult {
                    verifyMultText = String(format: "%.2f", mult)
                }
            }
        } catch {
            // No derived params yet: defaults stay, badge says so. The
            // server returns 200-empty here; anything else is not fatal.
        }
    }

    // MARK: Actions

    func runBenchmark(client: OMLXClient) {
        let body = StorageBenchStartRequest(
            modelId: selectedModelId,
            fileGb: 2.0,
            samples: 256,
            readMb: 2
        )
        report = nil
        lastError = nil
        running = true
        progressPhase = ""
        progressDone = 0
        progressTotal = 0

        Task { [weak self] in
            do {
                let resp = try await client.startStorageBench(body)
                await MainActor.run {
                    guard let self else { return }
                    self.currentJobId = resp.jobId
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
        // tok/mult are for the post-run prediction — hold them for the poll
        // completion handler without storing as properties the UI binds to.
        // Empty fields mean "use the server's auto-derived params".
        pendingTokPerCycle = parsedTokPerCycle
        pendingVerifyMult = parsedVerifyMult
    }

    @ObservationIgnored
    private var pendingTokPerCycle: Double?
    @ObservationIgnored
    private var pendingVerifyMult: Double?

    func predict() async {
        guard canPredict, let client else { return }
        do {
            report = try await client.getStoragePrediction(
                modelId: selectedModelId,
                tokPerCycle: parsedTokPerCycle,
                verifyMult: parsedVerifyMult,
                measuredBaseTokS: parsedMeasuredBase
            )
            lastError = nil
        } catch {
            // 404 with "no measurement yet" is the expected first-run
            // state; anything else is a real error.
            let desc = error.omlxDescription
            if !desc.lowercased().contains("no completed storage measurement") {
                lastError = desc
            }
        }
    }

    // MARK: Polling

    private func pollResults(client: OMLXClient) {
        pollTask?.cancel()
        guard let jobId = currentJobId else { return }
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                do {
                    let resp = try await client.getStorageBenchResults(jobId: jobId)
                    let terminal = await MainActor.run { () -> Bool in
                        if let prog = resp.progress {
                            self.progressPhase = prog.phase
                            self.progressDone = prog.done
                            self.progressTotal = prog.total
                        }
                        if let report = resp.report {
                            self.report = report
                        }
                        if let err = resp.error, !err.isEmpty {
                            self.lastError = err
                        }
                        if resp.isTerminal {
                            self.running = false
                            // A fresh bench pair may have changed the
                            // derived params; refresh badge + fields, then
                            // re-predict with the freshest numbers.
                            Task { [weak self] in
                                await self?.loadAutoParams()
                                await self?.loadHistory()
                                await self?.predict()
                            }
                        }
                        return resp.isTerminal
                    }
                    if terminal { return }
                } catch {
                    await MainActor.run {
                        self.lastError = error.omlxDescription
                    }
                }
                try? await Task.sleep(for: .seconds(1))
            }
        }
    }
}

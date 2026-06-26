import SwiftUI

@MainActor
@Observable
final class QuantizationScreenVM {
    // Form state
    var selectedModelPath: String = ""
    var sensitivityModelPath: String = ""
    var oqLevel: Double = 4
    var textOnly: Bool = false
    var preserveMtp: Bool = false
    var dtype: String = "bfloat16"
    var advancedOpen: Bool = false

    // Server state
    private(set) var models: [OQModelInfo] = []
    private(set) var allModels: [OQModelInfo] = []
    private(set) var modelsLoaded: Bool = false
    private(set) var tasks: [OQTaskDTO] = []
    private(set) var estimate: OQEstimateResponse?

    // Upload state — covers the sheet + the Upload Tasks section. The token
    // is hydrated from Keychain on `start()` and re-written after a
    // successful `validateHFUploadToken` round-trip. We hold it in plain
    // memory while the screen is mounted so the sheet's SecureField stays
    // bound; it never gets persisted anywhere except the Keychain.
    var uploadTasks: [HFUploadTaskDTO] = []
    var uploadTarget: OQTaskDTO?
    var uploadCandidateModels: [HFUploadModelInfo] = []
    var uploadToken: String = ""
    var uploadValidatedUsername: String?
    var uploadOrgs: [HFOrgInfo] = []
    var uploadNamespace: String = ""
    var isValidatingToken: Bool = false
    var lastUploadError: String?

    // UI state
    private(set) var isStarting: Bool = false
    var lastError: String?
    var lastSuccess: String?

    @ObservationIgnored
    private weak var client: OMLXClient?
    @ObservationIgnored
    private var pollTask: Task<Void, Never>?
    @ObservationIgnored
    private var estimateDebounceTask: Task<Void, Never>?
    @ObservationIgnored
    private var successClearTask: Task<Void, Never>?

    // Settings (no Codable persistence — form lives only while screen is open).
    private static let groupSize = 64

    // MARK: Derived

    /// True iff the source model offers sensible sensitivity candidates
    /// (same model family at lower precision, etc.). The HTML hides the
    /// dropdown entirely when this is empty.
    var sensitivityCandidates: [OQModelInfo] {
        guard let source = models.first(where: { $0.path == selectedModelPath })
        else { return [] }
        let prefix = source.name.split(separator: "-").prefix(2).joined(separator: "-")
        return allModels.filter { m in
            m.path != selectedModelPath
            && m.isQuantized
            && m.name.hasPrefix(prefix)
        }
    }

    var selectedIsVLM: Bool {
        models.first(where: { $0.path == selectedModelPath })?.isVlm ?? false
    }

    var selectedHasMTP: Bool {
        models.first(where: { $0.path == selectedModelPath })?.hasMtpHeads ?? false
    }

    /// Estimate strip — memory pill. Mirrors `oqEstimatedMemory` in JS:
    /// if a sensitivity model is picked memory ≈ sens.size × 1.5 + 5 GB,
    /// else the `memory_streaming_formatted` from the API, else the source
    /// model's static `memory_streaming.peak_formatted`.
    var memoryText: String {
        if let est = estimate {
            if !sensitivityModelPath.isEmpty,
               let sens = allModels.first(where: { $0.path == sensitivityModelPath }) {
                let bytes = Int64(Double(sens.size) * 1.5) + 5 * 1024 * 1024 * 1024
                return formatBytes(bytes)
            }
            if let m = est.memoryStreamingFormatted, !m.isEmpty { return m }
        }
        return models.first(where: { $0.path == selectedModelPath })?
            .memoryStreaming?.peakFormatted ?? ""
    }

    var bpwText: String {
        guard let est = estimate else { return "" }
        return String(format: "%.1f", est.effectiveBpw)
    }

    var outputSizeText: String {
        estimate?.outputSizeFormatted ?? ""
    }

    // MARK: Lifecycle

    func start(client: OMLXClient) async {
        self.client = client
        // Hydrate the HF token from Keychain. Silent on miss — the sheet
        // shows an empty SecureField and the user can paste a new token.
        if let stored = Keychain.read(), !stored.isEmpty {
            self.uploadToken = stored
        }
        await loadModels()
        await loadUploadCandidates()
        await loadTasks()
        await loadUploadTasks()
        startPollingIfNeeded()
    }

    func stop() {
        pollTask?.cancel(); pollTask = nil
        estimateDebounceTask?.cancel(); estimateDebounceTask = nil
        successClearTask?.cancel(); successClearTask = nil
    }

    // MARK: Loaders

    private func loadModels() async {
        guard let client else { return }
        do {
            let resp = try await client.listOQModels()
            self.models = resp.models
            self.allModels = resp.allModels
            self.modelsLoaded = true
        } catch {
            self.modelsLoaded = true
            self.lastError = String(localized: "quant.error.load_models",
                                    defaultValue: "Failed to load models: \(error)",
                                    comment: "Banner error message when listing OQ models fails. Placeholder is the underlying error")
        }
    }

    private func loadTasks() async {
        guard let client else { return }
        do {
            let resp = try await client.listOQTasks()
            // If a task just transitioned from active → completed, refresh
            // the model list (so the new quantized model shows up as a
            // sensitivity candidate) and the upload candidate list (so the
            // README picker can copy from it). No manual reload required.
            let hadActive = self.tasks.contains(where: { $0.isActive })
            let hasActiveNow = resp.tasks.contains(where: { $0.isActive })
            self.tasks = resp.tasks
            if hadActive && !hasActiveNow {
                await loadModels()
                await loadUploadCandidates()
            }
        } catch {
            // Polling failure is expected during server restarts — don't
            // clobber the user-facing banner with transient errors.
        }
    }

    /// Loads local oQ models that can serve as a README source when the user
    /// picks "Copy from <model>" in the upload sheet. Filtered to oQ output
    /// (matching the HTML panel's `oq_models` slot) so the dropdown stays
    /// short.
    func loadUploadCandidates() async {
        guard let client else { return }
        do {
            let resp = try await client.listHFUploadModels()
            self.uploadCandidateModels = resp.oqModels
        } catch {
            // Soft-fail — the auto-generate path still works without
            // candidates, so we don't block the sheet on this.
        }
    }

    private func loadUploadTasks() async {
        guard let client else { return }
        do {
            let resp = try await client.listHFUploadTasks()
            self.uploadTasks = resp.tasks
        } catch {
            // Polling failure: stay quiet (same rationale as loadTasks).
        }
    }

    // MARK: Polling

    private func startPollingIfNeeded() {
        pollTask?.cancel()
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                let hasActive = await MainActor.run {
                    self.tasks.contains(where: { $0.isActive })
                    || self.uploadTasks.contains(where: { $0.isActive })
                }
                if hasActive {
                    try? await Task.sleep(for: .seconds(2))
                    if Task.isCancelled { return }
                    await self.loadTasks()
                    await self.loadUploadTasks()
                } else {
                    // Idle poll cadence — 6 s while no work is queued.
                    try? await Task.sleep(for: .seconds(6))
                    if Task.isCancelled { return }
                    await self.loadTasks()
                    await self.loadUploadTasks()
                }
            }
        }
    }

    // MARK: Estimate (debounced)

    /// Schedules a 300 ms debounced fetch — matches the JS dashboard. Each
    /// call cancels the previous timer so rapid changes (typing in a select,
    /// keyboard arrows) collapse to a single network round-trip.
    func scheduleEstimateRefresh(client: OMLXClient) {
        estimateDebounceTask?.cancel()
        if selectedModelPath.isEmpty {
            estimate = nil
            return
        }
        let path = selectedModelPath
        let level = oqLevel
        let preserve = selectedHasMTP && preserveMtp
        estimateDebounceTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(300))
            if Task.isCancelled { return }
            do {
                let est = try await client.estimateOQ(
                    modelPath: path,
                    oqLevel: level,
                    preserveMtp: preserve
                )
                await MainActor.run {
                    guard let self else { return }
                    // Drop the result if the user has moved on to a different
                    // model since this request was kicked off.
                    if self.selectedModelPath == path { self.estimate = est }
                }
            } catch {
                // Silent — the strip will read "Calculating…" which is fine
                // for a transient estimate failure.
            }
        }
    }

    // MARK: Actions

    func startQuantization(client: OMLXClient) {
        guard !selectedModelPath.isEmpty, !isStarting else { return }
        isStarting = true
        lastError = nil
        lastSuccess = nil
        let body = OQStartRequest(
            modelPath: selectedModelPath,
            oqLevel: oqLevel,
            groupSize: Self.groupSize,
            sensitivityModelPath: sensitivityModelPath,
            textOnly: textOnly,
            dtype: dtype,
            preserveMtp: selectedHasMTP && preserveMtp
        )
        let displayName = models.first(where: { $0.path == selectedModelPath })?.name
            ?? selectedModelPath
        let levelLabel = (oqLevel.rounded() == oqLevel)
            ? "oQ\(Int(oqLevel))" : "oQ\(oqLevel)"
        Task { [weak self] in
            defer { Task { @MainActor [weak self] in self?.isStarting = false } }
            do {
                let resp = try await client.startOQQuantization(body)
                await MainActor.run {
                    guard let self else { return }
                    if resp.success {
                        self.lastSuccess = String(localized: "quant.success.started",
                                                  defaultValue: "Quantization started: \(displayName) → \(levelLabel)",
                                                  comment: "Success banner after a quant job starts. Placeholders: source model name, target oQ level")
                        self.scheduleSuccessClear()
                    } else {
                        self.lastError = String(localized: "quant.error.server_refused",
                                                defaultValue: "Server refused the request",
                                                comment: "Banner error when the server returned success=false for a quant start")
                    }
                }
                await self?.loadTasks()
            } catch {
                await MainActor.run {
                    self?.lastError = String(localized: "quant.error.start_failed",
                                             defaultValue: "Failed to start: \(error)",
                                             comment: "Banner error when starting a quant job throws. Placeholder is the underlying error")
                }
            }
        }
    }

    func cancelTask(taskId: String, client: OMLXClient) {
        Task { [weak self] in
            do {
                _ = try await client.cancelOQTask(taskId: taskId)
                await self?.loadTasks()
            } catch {
                await MainActor.run {
                    self?.lastError = String(localized: "quant.error.cancel_failed",
                                             defaultValue: "Cancel failed: \(error)",
                                             comment: "Banner error when cancelling a quant task throws. Placeholder is the underlying error")
                }
            }
        }
    }

    func removeTask(taskId: String, client: OMLXClient) {
        Task { [weak self] in
            do {
                _ = try await client.removeOQTask(taskId: taskId)
                await self?.loadTasks()
            } catch {
                await MainActor.run {
                    self?.lastError = String(localized: "quant.error.remove_failed",
                                             defaultValue: "Remove failed: \(error)",
                                             comment: "Banner error when removing a quant task throws. Placeholder is the underlying error")
                }
            }
        }
    }

    private func scheduleSuccessClear() {
        successClearTask?.cancel()
        successClearTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(5))
            if Task.isCancelled { return }
            await MainActor.run { self?.lastSuccess = nil }
        }
    }

    // MARK: Upload actions

    /// Validates the current `uploadToken` against `/api/upload/validate-token`.
    /// On success the token is persisted to the Keychain so the next session
    /// skips this round-trip, and the namespace defaults to the returned
    /// username (with orgs available via the Popup in the sheet).
    func validateUploadToken(client: OMLXClient) async {
        let token = uploadToken.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !token.isEmpty else {
            lastUploadError = String(localized: "quant.upload.error.empty_token",
                                     defaultValue: "Token is empty",
                                     comment: "Validation error when the HF token field is empty before validation")
            return
        }
        isValidatingToken = true
        lastUploadError = nil
        defer { isValidatingToken = false }
        do {
            let resp = try await client.validateHFUploadToken(hfToken: token)
            self.uploadValidatedUsername = resp.username
            self.uploadOrgs = resp.orgs
            self.uploadNamespace = resp.username
            Keychain.write(token)
        } catch {
            self.uploadValidatedUsername = nil
            self.uploadOrgs = []
            self.uploadNamespace = ""
            self.lastUploadError = String(localized: "quant.upload.error.validate_failed",
                                          defaultValue: "Validate failed: \(error.omlxDescription)",
                                          comment: "Error message when HF token validation throws. Placeholder is the underlying error description")
        }
    }

    /// Submits a configured upload job. The caller (the sheet) clears
    /// `uploadTarget` on success; on failure we surface the message via
    /// `lastUploadError` and leave the sheet open so the user can correct
    /// the body and retry without losing their inputs.
    func startUpload(body: HFUploadStartRequest, client: OMLXClient) async {
        lastUploadError = nil
        do {
            let resp = try await client.startHFUpload(body)
            if resp.success == false {
                lastUploadError = String(localized: "quant.upload.error.server_refused",
                                         defaultValue: "Server refused the request",
                                         comment: "Error when the server returned success=false for an upload start")
            }
            await loadUploadTasks()
            // Make sure the polling loop picks up the new active task even
            // if nothing else was running before this submission.
            startPollingIfNeeded()
        } catch {
            lastUploadError = String(localized: "quant.upload.error.start_failed",
                                     defaultValue: "Upload failed: \(error.omlxDescription)",
                                     comment: "Error when an upload start request throws. Placeholder is the underlying error description")
        }
    }

    func cancelUpload(taskId: String, client: OMLXClient) {
        Task { [weak self] in
            do {
                _ = try await client.cancelHFUploadTask(taskId: taskId)
                await self?.loadUploadTasks()
            } catch {
                await MainActor.run {
                    self?.lastUploadError = String(localized: "quant.upload.error.cancel_failed",
                                                   defaultValue: "Cancel failed: \(error)",
                                                   comment: "Error when cancelling an upload task throws. Placeholder is the underlying error")
                }
            }
        }
    }

    func removeUpload(taskId: String, client: OMLXClient) {
        Task { [weak self] in
            do {
                _ = try await client.removeHFUploadTask(taskId: taskId)
                await self?.loadUploadTasks()
            } catch {
                await MainActor.run {
                    self?.lastUploadError = String(localized: "quant.upload.error.remove_failed",
                                                   defaultValue: "Remove failed: \(error)",
                                                   comment: "Error when removing an upload task throws. Placeholder is the underlying error")
                }
            }
        }
    }

}

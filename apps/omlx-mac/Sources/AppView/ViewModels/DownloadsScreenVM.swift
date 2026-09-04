import SwiftUI

@MainActor
@Observable
final class DownloadsScreenVM {
    // MARK: - Active source

    /// Currently selected download source. Drives which form, task list,
    /// and recommended set the view shows. Switching the source kicks a
    /// fresh load of that source's tasks + recommended on first activation.
    var source: DownloadSource = .hf {
        didSet { sourceDidChange() }
    }

    /// `true` when the server reports the modelscope SDK is importable. The
    /// switcher disables the MS option when false so we never start a flow
    /// that will only ever 503.
    private(set) var msAvailable: Bool = false

    // MARK: - HF state (pre-existing)

    var repoText: String = ""
    private(set) var tasks: [HFTaskDTO] = []
    private(set) var recommended: [HFModelInfo] = []

    /// Configured HF mirror endpoint. Empty when using the HF default
    /// (huggingface.co). Loaded once on screen start, kept in sync with
    /// PATCH /admin/api/global-settings (`hf_endpoint`).
    private(set) var mirrorEndpoint: String = ""
    var isEditingMirror: Bool = false
    var mirrorDraft: String = ""
    private(set) var mirrorBusy: Bool = false

    /// Auto-complete suggestions for the manual repo input. Cleared when
    /// the input is empty, exactly matches a chosen repo, or the user
    /// dismisses the dropdown with Esc.
    private(set) var searchResults: [HFModelInfo] = []
    private(set) var searchLoading: Bool = false
    var searchDismissed: Bool = false
    @ObservationIgnored
    private var searchTask: Task<Void, Never>?
    @ObservationIgnored
    private var lastSearchQuery: String = ""

    // MARK: - MS state (Phase 2)

    var msRepoText: String = ""
    private(set) var msTasks: [MSTaskDTO] = []
    private(set) var msRecommended: [MSModelInfo] = []

    /// Configured MS mirror endpoint. Empty = ModelScope default
    /// (modelscope.cn). Kept in sync with PATCH /admin/api/global-settings
    /// (`ms_endpoint`).
    private(set) var msMirrorEndpoint: String = ""
    var isEditingMsMirror: Bool = false
    var msMirrorDraft: String = ""
    private(set) var msMirrorBusy: Bool = false

    private(set) var msSearchResults: [MSModelInfo] = []
    private(set) var msSearchLoading: Bool = false
    var msSearchDismissed: Bool = false
    @ObservationIgnored
    private var msSearchTask: Task<Void, Never>?
    @ObservationIgnored
    private var lastMsSearchQuery: String = ""

    // MARK: - Cross-source

    private(set) var isStarting: Bool = false
    private(set) var recommendedLoading: Bool = false
    var recommendedSort: SuggestedSort = .downloads
    var lastError: String?

    /// Target for the model-card sheet. Non-nil while the sheet is open;
    /// `ModelCardSheet` updates it back to nil on dismiss via `.sheet(item:)`.
    /// Identifiable on `repoId+source`, so re-tapping the same row while a
    /// sheet for a different row was open re-fires the fetch task.
    var modelCardTarget: ModelCardTarget?

    /// Open the model-card sheet for a row. Source is resolved from the
    /// active tab — `.hf` rows always belong to the HF tab and vice versa.
    func showModelCard(repoId: String) {
        let resolvedSource: ModelCardSource = (source == .hf) ? .huggingFace : .modelScope
        modelCardTarget = ModelCardTarget(repoId: repoId, source: resolvedSource)
    }

    /// Host the user sees on the Downloads screen. Strips scheme so the
    /// inline label reads like `huggingface.co` / `hf-mirror.com` per design.
    var mirrorHost: String { hostString(mirrorEndpoint, fallback: "huggingface.co") }
    var mirrorIsCustom: Bool {
        !mirrorEndpoint.trimmingCharacters(in: .whitespaces).isEmpty
    }

    var msMirrorHost: String { hostString(msMirrorEndpoint, fallback: "modelscope.cn") }
    var msMirrorIsCustom: Bool {
        !msMirrorEndpoint.trimmingCharacters(in: .whitespaces).isEmpty
    }

    private func hostString(_ raw: String, fallback: String) -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespaces)
        if trimmed.isEmpty { return fallback }
        if let url = URL(string: trimmed), let host = url.host { return host }
        return trimmed
    }

    /// Re-sorted view of the active source's recommended list. Always
    /// descending; entries missing the sort key fall to the bottom so
    /// they don't shove valid models out of the top 15 the section shows.
    var sortedRecommended: [HFModelInfo] {
        let pool = (source == .hf) ? recommended : msRecommended
        switch recommendedSort {
        case .downloads:
            return pool.sorted { ($0.downloads ?? -1) > ($1.downloads ?? -1) }
        case .params:
            return pool.sorted { ($0.params ?? -1) > ($1.params ?? -1) }
        case .size:
            return pool.sorted { ($0.size ?? -1) > ($1.size ?? -1) }
        }
    }

    @ObservationIgnored
    private weak var client: OMLXClient?
    @ObservationIgnored
    private var pollTask: Task<Void, Never>?
    @ObservationIgnored
    private var hasLoadedHFRecommended = false
    @ObservationIgnored
    private var hasLoadedMSRecommended = false

    var activeTasks: [HFTaskDTO] {
        (source == .hf ? tasks : msTasks).filter { $0.isActive }
    }
    var terminalTasks: [HFTaskDTO] {
        (source == .hf ? tasks : msTasks).filter { !$0.isActive }
    }

    func start(client: OMLXClient) async {
        self.client = client
        pollTask?.cancel()
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                await self.refreshTasks()
                try? await Task.sleep(for: .seconds(1))
            }
        }
        await refreshMirrors(client: client)
        await refreshMsAvailability(client: client)
        await loadActiveRecommendedIfNeeded(client: client)
    }

    /// Source-switch hook. When the user picks the other source, kick a
    /// task refresh + lazy recommended load for that source. Mirror state
    /// already lives in the VM, so the new form's preview values are ready
    /// instantly.
    private func sourceDidChange() {
        guard let client else { return }
        Task { [weak self] in
            await self?.refreshTasks()
            await self?.loadActiveRecommendedIfNeeded(client: client)
        }
    }

    private func loadActiveRecommendedIfNeeded(client: OMLXClient) async {
        switch source {
        case .hf:
            if !hasLoadedHFRecommended {
                hasLoadedHFRecommended = true
                await loadRecommended(client: client)
            }
        case .ms:
            if !hasLoadedMSRecommended {
                hasLoadedMSRecommended = true
                await loadRecommended(client: client)
            }
        }
    }

    private func refreshMirrors(client: OMLXClient) async {
        // Load both mirror endpoints once so the inactive source's form
        // reads correctly the moment the user switches.
        do {
            let settings = try await client.getGlobalSettings()
            self.mirrorEndpoint = settings.huggingface?.endpoint ?? ""
            self.msMirrorEndpoint = settings.modelscope?.endpoint ?? ""
        } catch {
            // Non-fatal — leave mirrors as defaults.
        }
    }

    private func refreshMsAvailability(client: OMLXClient) async {
        do {
            let resp = try await client.getMSStatus()
            self.msAvailable = resp.available
            if !resp.available && source == .ms {
                self.source = .hf
            }
        } catch {
            self.msAvailable = false
        }
    }

    func saveMirror(client: OMLXClient) {
        let draft = mirrorDraft.trimmingCharacters(in: .whitespaces)
        // Server treats empty string as "reset to default" — pass it through.
        Task { [weak self] in
            guard let self else { return }
            self.mirrorBusy = true
            defer { Task { @MainActor [weak self] in self?.mirrorBusy = false } }
            do {
                _ = try await client.updateGlobalSettings(
                    GlobalSettingsPatch(hfEndpoint: draft)
                )
                self.mirrorEndpoint = draft
                self.isEditingMirror = false
                self.mirrorDraft = ""
                self.lastError = nil
            } catch {
                self.lastError = error.omlxDescription
            }
        }
    }

    func resetMirror(client: OMLXClient) {
        mirrorDraft = ""
        saveMirror(client: client)
    }

    func saveMsMirror(client: OMLXClient) {
        let draft = msMirrorDraft.trimmingCharacters(in: .whitespaces)
        Task { [weak self] in
            guard let self else { return }
            self.msMirrorBusy = true
            defer { Task { @MainActor [weak self] in self?.msMirrorBusy = false } }
            do {
                _ = try await client.updateGlobalSettings(
                    GlobalSettingsPatch(msEndpoint: draft)
                )
                self.msMirrorEndpoint = draft
                self.isEditingMsMirror = false
                self.msMirrorDraft = ""
                self.lastError = nil
            } catch {
                self.lastError = error.omlxDescription
            }
        }
    }

    func resetMsMirror(client: OMLXClient) {
        msMirrorDraft = ""
        saveMsMirror(client: client)
    }

    // MARK: Autocomplete

    /// Driven by .onChange(of: vm.repoText) in the view. Cancels any in-
    /// flight search, debounces 300 ms, then fires GET /admin/api/hf/search.
    /// Stays quiet when input is < 2 chars or matches the previous result
    /// (avoids hammering the API for trivial keystrokes).
    func updateSearch(query rawQuery: String, client: OMLXClient) {
        let q = rawQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        searchTask?.cancel()
        if q.isEmpty {
            searchResults = []
            searchLoading = false
            searchDismissed = false
            lastSearchQuery = ""
            return
        }
        if q == lastSearchQuery && !searchResults.isEmpty {
            return
        }
        if q.count < 2 {
            // Too short to be useful — wait for more characters.
            return
        }
        searchDismissed = false
        searchTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(300))
            if Task.isCancelled { return }
            guard let self else { return }
            self.searchLoading = true
            defer { Task { @MainActor [weak self] in self?.searchLoading = false } }
            do {
                let resp = try await client.searchHFModels(query: q, limit: 20)
                if Task.isCancelled { return }
                self.searchResults = resp.models
                self.lastSearchQuery = q
            } catch is CancellationError {
                return
            } catch {
                // Treat search failures as soft — keep the input usable for
                // direct repo-id paste, just don't surface the error in the
                // download lastError slot.
                self.searchResults = []
            }
        }
    }

    func pickSearchResult(_ model: HFModelInfo) {
        repoText = model.repoId
        // Picking a result means we don't want a popup again for the same
        // string — store it as the satisfied query.
        lastSearchQuery = model.repoId
        searchResults = []
        searchDismissed = true
    }

    func dismissSearch() {
        searchTask?.cancel()
        searchResults = []
        searchDismissed = true
    }

    // MARK: - MS autocomplete

    /// Mirror of `updateSearch` for the ModelScope source. Same debounce,
    /// same min-length, same cancel semantics — only the endpoint changes.
    func updateMsSearch(query rawQuery: String, client: OMLXClient) {
        let q = rawQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        msSearchTask?.cancel()
        if q.isEmpty {
            msSearchResults = []
            msSearchLoading = false
            msSearchDismissed = false
            lastMsSearchQuery = ""
            return
        }
        if q == lastMsSearchQuery && !msSearchResults.isEmpty { return }
        if q.count < 2 { return }
        msSearchDismissed = false
        msSearchTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(300))
            if Task.isCancelled { return }
            guard let self else { return }
            self.msSearchLoading = true
            defer { Task { @MainActor [weak self] in self?.msSearchLoading = false } }
            do {
                let resp = try await client.searchMSModels(query: q, limit: 20)
                if Task.isCancelled { return }
                self.msSearchResults = resp.models
                self.lastMsSearchQuery = q
            } catch is CancellationError {
                return
            } catch {
                // Soft fail — keep input usable for direct repo-id paste.
                self.msSearchResults = []
            }
        }
    }

    func pickMsSearchResult(_ model: MSModelInfo) {
        msRepoText = model.repoId
        lastMsSearchQuery = model.repoId
        msSearchResults = []
        msSearchDismissed = true
    }

    func dismissMsSearch() {
        msSearchTask?.cancel()
        msSearchResults = []
        msSearchDismissed = true
    }

    func stop() {
        pollTask?.cancel()
        pollTask = nil
    }

    // MARK: - Source-routed CRUD

    /// Starts a download against the active source. `repo` lets the
    /// suggested-models grid pass a repo id without having to first stuff
    /// it into the input box.
    func startDownload(repo: String? = nil, client: OMLXClient) {
        let target = (repo ?? (source == .hf ? repoText : msRepoText))
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !target.isEmpty else { return }
        isStarting = true
        let activeSource = source
        Task { [weak self] in
            defer { Task { @MainActor in self?.isStarting = false } }
            do {
                switch activeSource {
                case .hf:
                    _ = try await client.startHFDownload(repoId: target)
                    if repo == nil { self?.repoText = "" }
                case .ms:
                    _ = try await client.startMSDownload(modelId: target)
                    if repo == nil { self?.msRepoText = "" }
                }
                await self?.refreshTasks()
            } catch {
                guard let self else { return }
                self.lastError = error.omlxDescription
            }
        }
    }

    func cancel(taskId: String, client: OMLXClient) {
        let activeSource = source
        Task { [weak self] in
            do {
                switch activeSource {
                case .hf: _ = try await client.cancelHFDownload(taskId: taskId)
                case .ms: _ = try await client.cancelMSDownload(taskId: taskId)
                }
                await self?.refreshTasks()
            } catch {
                guard let self else { return }
                self.lastError = error.omlxDescription
            }
        }
    }

    func retry(taskId: String, client: OMLXClient) {
        let activeSource = source
        Task { [weak self] in
            do {
                switch activeSource {
                case .hf: _ = try await client.retryHFDownload(taskId: taskId)
                case .ms: _ = try await client.retryMSDownload(taskId: taskId)
                }
                await self?.refreshTasks()
            } catch {
                guard let self else { return }
                self.lastError = error.omlxDescription
            }
        }
    }

    func remove(taskId: String, client: OMLXClient) {
        let activeSource = source
        Task { [weak self] in
            do {
                switch activeSource {
                case .hf: _ = try await client.removeHFTask(taskId: taskId)
                case .ms: _ = try await client.removeMSTask(taskId: taskId)
                }
                await self?.refreshTasks()
            } catch {
                guard let self else { return }
                self.lastError = error.omlxDescription
            }
        }
    }

    func loadRecommended(client: OMLXClient) async {
        self.recommendedLoading = true
        defer { self.recommendedLoading = false }
        do {
            // Trending-first, then popular, deduped by repoId. Mirrors how
            // the original dashboard surfaces both lists side-by-side.
            switch source {
            case .hf:
                let resp = try await client.getHFRecommended()
                self.recommended = Self.merge(trending: resp.trending, popular: resp.popular)
            case .ms:
                let resp = try await client.getMSRecommended()
                self.msRecommended = Self.merge(trending: resp.trending, popular: resp.popular)
            }
            self.lastError = nil
        } catch {
            // 502/504 are common (mirror unreachable, dev offline). Surface
            // but keep UI usable.
            self.lastError = error.omlxDescription
        }
    }

    private static func merge(trending: [HFModelInfo], popular: [HFModelInfo]) -> [HFModelInfo] {
        var seen = Set<String>()
        var merged: [HFModelInfo] = []
        for m in trending + popular where seen.insert(m.repoId).inserted {
            merged.append(m)
        }
        return merged
    }

    private func refreshTasks() async {
        guard let client else { return }
        do {
            switch source {
            case .hf: self.tasks   = try await client.listHFTasks().tasks
            case .ms: self.msTasks = try await client.listMSTasks().tasks
            }
            self.lastError = nil
        } catch {
            self.lastError = error.omlxDescription
        }
    }

}
